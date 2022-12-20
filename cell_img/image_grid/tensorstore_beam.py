"""Tensorstore beam methods."""

import asyncio
import concurrent
import queue
from typing import Tuple, Type, Union

import apache_beam as beam
from apache_beam.transforms import window
from apache_beam.utils import windowed_value
import tensorstore as ts

_NAMESPACE = 'ts'
_MAX_ATTEMPTS = 3
_TIMEOUT_SEC = 300
_MAX_WORKERS = 32
_EXCEPTIONS_TO_CATCH = (asyncio.CancelledError,)

_tensorstore_context = ts.Context()


class RetryableExecutor(concurrent.futures.ThreadPoolExecutor):
  """Like ThreadPoolExecutor but allows retry of failed tasks."""

  def __init__(self,
               max_attempts: int,
               exceptions_to_catch: Union[Type[Exception],
                                          Tuple[Type[Exception], ...]],
               max_workers=None):
    if max_attempts < 1:
      raise ValueError('max_attempts must be >= 1 but got %d' % max_attempts)
    self._max_attempts = max_attempts
    self._exceptions_to_catch = exceptions_to_catch
    super().__init__(max_workers)

  def submit(self, fn, *args, **kwargs):
    retry_fn = self._add_retry(fn)
    return super().submit(retry_fn, *args, **kwargs)

  def is_shutdown(self):
    return self._shutdown

  def _add_retry(self, func):
    """Returns a function that wraps the input func with retries."""

    def _func(*args, **kwargs):
      last_err = None
      for _ in range(self._max_attempts):
        try:
          return func(*args, **kwargs)
        except self._exceptions_to_catch as e:
          beam.metrics.Metrics.counter(_NAMESPACE,
                                       f'retry_{type(e).__name__}').inc()
          last_err = e
      if last_err is not None:
        raise last_err

    return _func


def _write_array_to_view(view, array):
  view.write(array).result(timeout=_TIMEOUT_SEC)
  beam.metrics.Metrics.counter(_NAMESPACE, 'write_to_tensorstore').inc()


class _WriteTensorStoreDoFn(beam.DoFn):
  """DoFn to write data to tensorstore."""

  def __init__(self, tensorstore_spec: ts.Spec, create_tensorstore):
    super(_WriteTensorStoreDoFn, self).__init__()
    self._tensorstore_spec = tensorstore_spec

    if create_tensorstore:
      # Open the tensorstore to create it.
      self._output_store = ts.open(
          self._tensorstore_spec,
          open=False,
          create=True,  # Will fail if already exists;Remove if restarts happen.
          context=_tensorstore_context).result()

  def setup(self):
    self._output_store = ts.open(
        self._tensorstore_spec,
        write=True,
        open=True,
        context=_tensorstore_context).result()
    self._executor = RetryableExecutor(_MAX_ATTEMPTS, _EXCEPTIONS_TO_CATCH,  # pytype: disable=wrong-arg-types  # py39-upgrade
                                       _MAX_WORKERS)
    beam.metrics.Metrics.counter(_NAMESPACE, 'write_setup').inc()

  def process(self, view_slice_and_array):
    beam.metrics.Metrics.counter(_NAMESPACE, 'write_process').inc()
    view_slice, array = view_slice_and_array

    output_view = self._output_store[view_slice]
    if output_view.dtype.numpy_dtype != array.dtype:
      beam.metrics.Metrics.counter(_NAMESPACE,
                                   'ERROR-dtype=%s' % str(array.dtype)).inc()
    typed_array = array.astype(output_view.dtype.numpy_dtype)

    if self._executor.is_shutdown():
      beam.metrics.Metrics.counter(_NAMESPACE,
                                   'Warning: executor already shut down.').inc()
      return
    self._executor.submit(_write_array_to_view, output_view, typed_array)
    yield view_slice

  def teardown(self):
    # Wait for all writes to complete, and raise any errors that occurred.
    self._executor.shutdown()
    beam.metrics.Metrics.counter(_NAMESPACE, 'write_teardown').inc()


class WriteTensorStore(beam.PTransform):
  """PTransform to write data to tensorstore."""

  def __init__(self, tensorstore_spec: ts.Spec, create_tensorstore=False):
    super(WriteTensorStore, self).__init__()
    self._tensorstore_spec = tensorstore_spec
    self._create_tensorstore = create_tensorstore

  def expand(self, p_view_slice_and_array: beam.pvalue.PCollection) -> None:
    return p_view_slice_and_array | beam.ParDo(
        _WriteTensorStoreDoFn(self._tensorstore_spec, self._create_tensorstore))


def _read_array_from_view(view, view_slice, passthrough, results_queue):
  array = view.read().result(timeout=_TIMEOUT_SEC)
  beam.metrics.Metrics.counter(_NAMESPACE,
                               'read-shape=%s' % str(array.shape)).inc()
  results_queue.put((view_slice, array) + passthrough)


class _ReadTensorStoreDoFn(beam.DoFn):
  """DoFn to read data from tensorstore."""

  def __init__(self, tensorstore_spec: ts.Spec):
    super(_ReadTensorStoreDoFn, self).__init__()
    # Check that the tensorstore can be opened for read.
    store = ts.open(
        tensorstore_spec,
        open=True,
        write=False,
        create=False,
        context=_tensorstore_context).result()
    self._tensorstore_spec = store.spec()

  def setup(self):
    self._input_store = ts.open(
        self._tensorstore_spec,
        open=True,
        write=False,
        create=False,
        context=_tensorstore_context).result()
    self._executor = RetryableExecutor(_MAX_ATTEMPTS, _EXCEPTIONS_TO_CATCH,  # pytype: disable=wrong-arg-types  # py39-upgrade
                                       _MAX_WORKERS)
    beam.metrics.Metrics.counter(_NAMESPACE, 'read_setup').inc()

  def start_bundle(self):
    self._read_results = queue.Queue(maxsize=_MAX_WORKERS)
    beam.metrics.Metrics.counter(_NAMESPACE, 'read_start_bundle').inc()

  def process(self, slice_and_passthrough):
    beam.metrics.Metrics.counter(_NAMESPACE, 'read_process').inc()
    view_slice = slice_and_passthrough[0]
    passthrough = slice_and_passthrough[1:]
    view = self._input_store[view_slice]

    while not self._read_results.empty():
      yield self._read_results.get()
    self._executor.submit(_read_array_from_view, view, view_slice, passthrough,
                          self._read_results)

  def finish_bundle(self):
    # Wait for all reads to complete, and raise any errors that occurred.
    results_to_yield = []
    while not self._read_results.empty():
      results_to_yield.append(self._read_results.get())
    self._executor.shutdown()
    while not self._read_results.empty():
      results_to_yield.append(self._read_results.get())

    for result in results_to_yield:
      # Yield WindowedValues since that is the only return type allowed for
      # finish_bundle.
      yield windowed_value.WindowedValue(result, -1, [window.GlobalWindow()])
    beam.metrics.Metrics.counter(_NAMESPACE, 'read_finish_bundle').inc()


class ReadTensorStore(beam.PTransform):
  """PTransform to read data from tensorstore.

  This optionally allows extra data to be passed through the PTransform and kept
  with the data that is read from tensorstore.

  Input PCollection should have tuples where the first element is a tuple of
  slices defining the coordinates to read data from. The remaining elements of
  the input tuple are passed through.

  Output PCollection will have tuples where the first element is the tuple of
  slices from the input, the second element is the array data that was read, and
  the remaining elements are the passed through part of the input tuple.
  """

  def __init__(self, tensorstore_spec: ts.Spec):
    super(ReadTensorStore, self).__init__()
    self._tensorstore_spec = tensorstore_spec

  def expand(
      self, p_slice_and_passthrough: beam.pvalue.PCollection
  ) -> beam.pvalue.PCollection:
    p_slice_and_array_and_passthrough = (
        p_slice_and_passthrough
        | beam.ParDo(_ReadTensorStoreDoFn(self._tensorstore_spec)))
    return p_slice_and_array_and_passthrough

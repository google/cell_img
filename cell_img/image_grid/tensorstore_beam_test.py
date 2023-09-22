"""Tests for cell_img.image_grid.tensorstore_beam."""

import asyncio
import tempfile

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
from cell_img.image_grid import tensorstore_beam
import numpy as np
import tensorstore as ts


def valid_tensorstore_spec(dir_path):
  return ts.Spec({
      'driver': 'n5',
      'kvstore': {
          'driver': 'file',
          'path': dir_path,
      },
      'metadata': {
          'compression': {
              'type': 'gzip'
          },
          'dataType': 'uint16',
          'dimensions': [1000, 20000],
          'blockSize': [10, 10],
      }
  })


class TensorstoreBeamTest(absltest.TestCase):

  def testWriteTensorStore(self):
    with tempfile.TemporaryDirectory() as dir_path:
      ts_spec = valid_tensorstore_spec(dir_path)

      # The location and data to be written in beam.
      view_slice = (slice(80, 82), slice(99, 102))
      orig_array = np.array([[1, 2, 3], [4, 5, 6]]).astype('uint16')

      with TestPipeline(beam.runners.direct.DirectRunner()) as p:
        p_view_slice_and_array = p | beam.Create([(view_slice, orig_array)])
        _ = p_view_slice_and_array = (
            p_view_slice_and_array | tensorstore_beam.WriteTensorStore(
                ts_spec, create_tensorstore=True))

      # Now read back the data with the serial API and assert it matches.
      dataset = ts.open(ts_spec).result()
      actual_array = dataset[view_slice].read().result()
      np.testing.assert_equal(orig_array, actual_array)

  def testReadTensorStore(self):
    with tempfile.TemporaryDirectory() as dir_path:
      ts_spec = valid_tensorstore_spec(dir_path)

      # First write the data to the location using the serial API.
      view_slice = (slice(80, 82), slice(99, 102))
      orig_array = np.array([[1, 2, 3], [4, 5, 6]]).astype('uint16')
      dataset = ts.open(ts_spec, create=True).result()
      dataset[view_slice] = orig_array

      with TestPipeline(beam.runners.direct.DirectRunner()) as p:
        p_slice_and_passthrough = p | beam.Create(
            [(view_slice, 'pass', 'through')])
        p_slice_and_array_and_passthrough = (
            p_slice_and_passthrough | tensorstore_beam.ReadTensorStore(ts_spec))

        # There should be one element in the resulting pcollection.
        p_count = (
            p_slice_and_array_and_passthrough | beam.combiners.Count.Globally())
        assert_that(p_count, equal_to([1]))

        # Assert that the content of the resulting pcollection is as expected.
        def do_assert(element):
          actual_slice = element[0]
          self.assertEqual(actual_slice, view_slice)

          actual_array = element[1]
          np.testing.assert_equal(orig_array, actual_array)

          actual_passthrough = element[2:]
          np.testing.assert_equal(('pass', 'through'), actual_passthrough)

        _ = p_slice_and_array_and_passthrough | beam.Map(do_assert)

  def testWriteThenReadTensorStore(self):
    with tempfile.TemporaryDirectory() as dir_path:
      ts_spec = valid_tensorstore_spec(dir_path)

      # The location and data to be written in beam.
      view_slice = (slice(80, 82), slice(99, 102))
      orig_array = np.array([[1, 2, 3], [4, 5, 6]]).astype('uint16')

      with TestPipeline(beam.runners.direct.DirectRunner()) as p:
        p_view_slice_and_array = (
            p | 'create_write' >> beam.Create([(view_slice, orig_array)]))
        _ = p_view_slice_and_array = (
            p_view_slice_and_array | tensorstore_beam.WriteTensorStore(
                ts_spec, create_tensorstore=True))

      # Only try reading data after the pipeline to write it has finished.
      with TestPipeline(beam.runners.direct.DirectRunner()) as p:
        p_slice = p | 'create_read' >> beam.Create([(view_slice,)])
        p_slice_and_array = (
            p_slice | tensorstore_beam.ReadTensorStore(ts_spec))

        # There should be one element in the resulting pcollection.
        p_count = (p_slice_and_array | beam.combiners.Count.Globally())
        assert_that(p_count, equal_to([1]))

        # Assert that the content of the resulting pcollection is as expected.
        def do_assert(element):
          self.assertLen(element, 2)
          actual_slice = element[0]
          self.assertEqual(actual_slice, view_slice)

          actual_array = element[1]
          np.testing.assert_equal(orig_array, actual_array)

        _ = p_slice_and_array | beam.Map(do_assert)

  def testScheduleAfterShutdown(self):
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as dir_path:
      ts_spec = valid_tensorstore_spec(dir_path)

      # The location and data to be written in beam.
      view_slice = (slice(80, 82), slice(99, 102))
      orig_array = np.array([[1, 2, 3], [4, 5, 6]]).astype('uint16')

      write_do_fn = tensorstore_beam._WriteTensorStoreDoFn(
          ts_spec, create_tensorstore=True)
      write_do_fn.setup()
      write_do_fn.teardown()
      # Submitting tasks after shutdown should not raise an error.
      _ = list(write_do_fn.process((view_slice, orig_array)))

  def testWritingTheWrongSizeRaises(self):
    with tempfile.TemporaryDirectory() as dir_path:
      ts_spec = valid_tensorstore_spec(dir_path)

      # The location and data to be written in beam.
      view_slice = (slice(80, 82), slice(99, 102))
      # Make an array that is too big for this view slice
      big_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).astype('uint16')

      with self.assertRaisesRegex(ValueError,
                                  '.*Error aligning dimensions.*'):
        with TestPipeline(beam.runners.direct.DirectRunner()) as p:
          p_view_slice_and_array = p | beam.Create([(view_slice, big_array)])

          _ = p_view_slice_and_array = (
              p_view_slice_and_array | tensorstore_beam.WriteTensorStore(
                  ts_spec, create_tensorstore=True))


class RetryableExecutorTest(absltest.TestCase):

  def test_constructor_one_exception(self):
    tensorstore_beam.RetryableExecutor(
        max_attempts=3, exceptions_to_catch=Exception, max_workers=4)

  def test_constructor_multiple_exceptions(self):
    tensorstore_beam.RetryableExecutor(
        max_attempts=3,
        exceptions_to_catch=(KeyError, ValueError),
        max_workers=4)

  def test_constructor_raises_on_zero_max_attempts(self):
    with self.assertRaisesRegex(ValueError,
                                'max_attempts must be >= 1 but got 0'):
      tensorstore_beam.RetryableExecutor(
          max_attempts=0, exceptions_to_catch=Exception)

  def test_succeeds_when_no_errors(self):
    executor = tensorstore_beam.RetryableExecutor(
        max_attempts=1, exceptions_to_catch=Exception)
    my_future = executor.submit(lambda x: x, 1234)
    self.assertEqual(1234, my_future.result())

  def test_succeeds_when_ten_attempts_and_nine_timeouts(self):
    executor = tensorstore_beam.RetryableExecutor(  # pytype: disable=wrong-arg-types  # py39-upgrade
        max_attempts=10, exceptions_to_catch=asyncio.CancelledError)
    num_timeouts = 9
    exceptions_to_raise = [asyncio.CancelledError] * num_timeouts
    func_with_timeouts = FuncWithExceptions(exceptions_to_raise)
    my_future = executor.submit(func_with_timeouts.func, 1234)
    self.assertEqual(1234, my_future.result())

  def test_raises_when_ten_attempts_and_ten_timeouts(self):
    executor = tensorstore_beam.RetryableExecutor(  # pytype: disable=wrong-arg-types  # py39-upgrade
        max_attempts=10, exceptions_to_catch=asyncio.CancelledError)
    num_timeouts = 10
    exceptions_to_raise = [asyncio.CancelledError] * num_timeouts
    func_with_timeouts = FuncWithExceptions(exceptions_to_raise)
    my_future = executor.submit(func_with_timeouts.func, 1234)
    with self.assertRaises(asyncio.CancelledError):
      my_future.result()

  def test_multiple_error_types(self):
    executor = tensorstore_beam.RetryableExecutor(  # pytype: disable=wrong-arg-types  # py39-upgrade
        max_attempts=3,
        exceptions_to_catch=(asyncio.CancelledError, KeyError),
        max_workers=1)
    func_with_exceptions = FuncWithExceptions(
        [asyncio.CancelledError, KeyError])
    my_future = executor.submit(func_with_exceptions.func, 1234)
    self.assertEqual(1234, my_future.result())

  def test_multiple_error_types_raises_on_uncaught_error(self):
    executor = tensorstore_beam.RetryableExecutor(  # pytype: disable=wrong-arg-types  # py39-upgrade
        max_attempts=3,
        exceptions_to_catch=asyncio.CancelledError,
        max_workers=1)
    func_with_exceptions = FuncWithExceptions(
        [asyncio.CancelledError, KeyError])
    my_future = executor.submit(func_with_exceptions.func, 1234)
    with self.assertRaises(KeyError):
      my_future.result()


class FuncWithExceptions(object):

  def __init__(self, exceptions_to_raise):
    self._exceptions_to_raise = exceptions_to_raise

  def func(self, my_arg):
    if self._exceptions_to_raise:
      to_raise = self._exceptions_to_raise.pop()
      raise to_raise
    return my_arg


if __name__ == '__main__':
  absltest.main()

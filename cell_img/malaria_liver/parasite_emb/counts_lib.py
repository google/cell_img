"""This module has beam functions to aggregate and output counts.

Aggregating and counting inside of a beam job can be tricky. This module
assumes you have a collection of elements that contain the information within
each element that can provide a unique key indicating the aggregation group,
along with the count values for all the metrics to be counted.

In this case, we want to aggregate to the level of a single well in the
experiment, and we want to count the number of different types of objects
(cells) found in each well. So our aggregation group is the well, and our
metrics are each type of object we want to count. Our elements each represent a
single identified object, equal to one putative cell. From the
element, we extract a well key that uniquely identifies the well that this
object should be counted toward (e.g. plate 1, well A04). For each metric, we
yield a count of 1 if this object counts towards that metric.

Specifically, we want to count every object detected in each well (each patch),
and we also want to count the number of parasites at each lifecycle stage.
So a patch that is a hypnozoite would yield once for being a patch and again
for being a hypnozoite.

All the elements yielded for each well are then aggregated and summed, and
the results are written to CSV. One CSV file is created for each metric,
containing the fields needed for aggregation (e.g. the plate and well values
that create the well unique identifier), and the metric (e.g. the count of
hypnozoites). These can easily be joined on the well key information to create
a single wide table as needed.

This module can also create weighted sums instead of counts. For example, if
you had a classifier that output the probability of an object belonging to
a specific class, you could yield the float probability for each object
instead of yielding a full 1 for each object that was predicted to belong
to the class. The aggregated metric would then be the sum
of all the predictions per well.

Within your beam pipeline, this module can be used as follows:
_ = (
        p_objects | 'object_counts' >> counts_lib.WellCountAggregator(
            output_dir=count_csv_dir,
            count_fn=counts_lib.yield_metrics_per_element))

where p_objects is a PCollection where each element is a dictionary representing
one patch. This will write CSVs to the given path, one for each metric.

After the beam pipeline, you can merge into a single csv:
  counts_lib.merge_count_csvs(
      count_csv_dir_path=count_csv_dir,
      output_filename=count_csv_filename)

To count different metrics, change the yield_metrics_per_element function to
yield the metrics you want to sum.

To aggregate to different values, change the NamedTuples to whichever
values you want to aggregate on and then change the associated code to create
keys using these fields.
"""

import os
from typing import Any, Callable, Dict, Generator, Iterable, NamedTuple, Tuple

import apache_beam as beam
from cell_img.common import io_lib
from cell_img.malaria_liver.parasite_emb import config


WELL_INDEX_COLS = ['batch', 'plate', 'well']

_METRICS_NAMESPACE = 'CellCounts'
# pylint: disable=invalid-name
WellKey = NamedTuple('WellKey', [
    ('batch', str),
    ('plate', str),
    ('well', str),
])

# pylint: disable=invalid-name
AggrKey = NamedTuple('AggrKey', [
    ('batch', str),
    ('plate', str),
    ('well', str),
    ('metric_name', str),
])

# pylint: disable=invalid-name
CsvKey = NamedTuple('CsvKey', [
    ('metric_name', str),
])

# pylint: disable=invalid-name
CsvRow = NamedTuple('CsvRow', [
    ('batch', str),
    ('plate', str),
    ('well', str),
    ('metric_value', float)
])


def _try_to_get(element, key):
  if key in element:
    return element[key]
  else:
    return 'missing_%s' % key


def yield_metrics_per_element(
    element: Dict[Any, Any]) -> Generator[Tuple[WellKey, str, int], None, None]:
  """Extracts metrics for a single element.

  Args:
    element:  One element to examine for metrics. An element is expected to
      be a dictionary containing fields providing information for the
      aggregation key (i.e. well_key) and the metrics to be counted.

  Yields:
    3-tuples where the first element is a WellKey, the second is the metric
    name, and the third is the metric value.
  Raises:
    ValueError if the element does not have the required keys.
  """
  beam.metrics.Metrics.counter(_METRICS_NAMESPACE, 'count_per_elem').inc()

  well_key = WellKey(
      batch=_try_to_get(element, 'batch'),
      plate=_try_to_get(element, 'plate'),
      well=_try_to_get(element, 'well')
    )

  # every row is a patch
  yield (well_key, 'num_obj', 1)
  # and yield one for the predicted stage
  if config.STAGE_RESULT in element:
    yield (well_key, 'num_' + element[config.STAGE_RESULT], 1)
  else:
    yield (well_key, 'num_UNSTAGED', 1)


class WellCountAggregator(beam.PTransform):
  """Saves the counts to CSV."""

  def __init__(self, output_dir: str,
               count_fn: Callable[[Dict[Any, Any]], Any]):
    self._output_dir = output_dir
    self._count_fn = count_fn

  def expand(self, p_parasite: beam.pvalue.PCollection) -> None:
    p_wellkey_metricname_metricvalue = (
        p_parasite
        | 'extract_count_per_well' >> beam.ParDo(
            self._count_fn))
    p_aggrkey_metricvalue = (
        p_wellkey_metricname_metricvalue | beam.Map(_to_aggr_key))
    p_aggrkey_summetricvalue = (
        p_aggrkey_metricvalue
        | beam.CombinePerKey(sum))
    p_csv_key_and_row = (
        p_aggrkey_summetricvalue | beam.Map(_to_csv_key_and_row))
    p_csv_key_and_rows = p_csv_key_and_row | beam.GroupByKey()
    _ = p_csv_key_and_rows | beam.Map(_save_csv, self._output_dir)


def _csv_key_to_path(csv_key: CsvKey, output_dir: str) -> str:
  # The name here should include every value in the csv_key, to avoid
  # overwriting files. Currently only the metric name is used, meaning all
  # batch/plate/wells will be saved to a single file. If this becomes
  # too large to join efficiently, it'd be easy to divide by batch.
  # To do that, add batch to the csv_key named tuple.
  # If that is added, the merge code will need to be updated to merge multiple
  # csvs per metric before merging across metrics.
  return os.path.join(output_dir, csv_key.metric_name + '.csv')


def _save_csv(element: Tuple[CsvKey, Iterable[CsvRow]],
              output_dir: str) -> None:
  """Saves a single csv with all the corresponding rows.

  Args:
    element: A 2-tuple where the first value is the CsvKey and the second is an
      iterable of all the corresponding CsvRows.
    output_dir: A string of the base directory to output the csv to.
      Subdirectories will be created based on the values in the CsvKey.
  """
  beam.metrics.Metrics.counter(_METRICS_NAMESPACE, 'save_csv').inc()
  csv_key, csv_rows = element

  output_path = _csv_key_to_path(csv_key, output_dir)
  csv_row_map_list = [csv_row._asdict() for csv_row in csv_rows]
  io_lib.write_maps_to_csv(csv_row_map_list, output_path)


def _to_aggr_key(element: Tuple[WellKey, str, int]) -> Tuple[AggrKey, int]:
  """Prepares the metrics for aggregation.

  Args:
    element: A 3-tuple where the first element is a WellKey, the second is the
      metric name, and the third is the count value.

  Returns:
    A 2-tuple ready to be aggregated where the first value is the AggrKey, and
    the second is the count value.
  """
  beam.metrics.Metrics.counter(_METRICS_NAMESPACE, 'to_aggr_key').inc()
  well_key, metric_name, metric_value = element
  aggr_key = AggrKey(
      batch=well_key.batch,
      plate=well_key.plate,
      well=well_key.well,
      metric_name=metric_name)
  return (aggr_key, metric_value)


def _to_csv_key_and_row(element: Tuple[AggrKey, int]) -> Tuple[CsvKey, CsvRow]:
  """Prepares the metrics for writing to csv.

  Args:
    element: A 2-tuple where the first element is an AggrKey and the second is
      the aggregated metric value.

  Returns:
    A 2-tuple where the first value is the CsvKey, and the second is the CsvRow.
  """
  beam.metrics.Metrics.counter(_METRICS_NAMESPACE, 'to_csv_key_and_row').inc()
  aggr_key, metric_value = element
  csv_key = CsvKey(
      metric_name=aggr_key.metric_name)
  csv_row = CsvRow(
      batch=aggr_key.batch,
      plate=aggr_key.plate,
      well=aggr_key.well,
      metric_value=metric_value)
  return csv_key, csv_row


def _read_one_count_csv(csv_path):
  """Reads one count csv into a dataframe, fixing column names."""

  df = io_lib.read_csv(csv_path)

  # validate that we have batch/plate/well columns
  if not set(WELL_INDEX_COLS).issubset(df.columns):
    raise ValueError(
        'Count csv "%s" does not have expected columns %s.\n'
        'Found columns: %s.' % (
            csv_path, WELL_INDEX_COLS, list(df.columns)))
  df['plate'] = config.format_plate_strings(df['plate'])

  # rename the metric column to the csv filename
  csv_filename = os.path.splitext(os.path.basename(csv_path))[0]
  new_col_name = '%s' % csv_filename
  df.rename(columns={'metric_value': new_col_name}, inplace=True)

  return df, new_col_name


def merge_count_csvs(count_csv_dir_path: str, output_filename: str):
  """Reads csvs with count information, merges and saves the merged csv.

  Args:
    count_csv_dir_path: String path to the directory with the csv files. It is
      assumed that csv_dir_path/*.csv will yield only count csvs files, and
      will yield all the count csv files. Each csv should have columns
      for batch, plate, well as well as one metric_value column with a count.
      The resulting dataframe will have one column for each csv found, where
      the col title will be the title of the csv + "_count".
    output_filename: String full path filename for the merged csv output.
  """
  glob_path = os.path.join(count_csv_dir_path, '*.csv')
  csv_paths = io_lib.glob(glob_path)
  if len(csv_paths) < 1:
    raise ValueError('Did not find any csv files to build the count_df.\n'
                     'filepath: %s' % (glob_path))

  count_cols = []
  count_df, next_count_col = _read_one_count_csv(csv_paths[0])
  count_cols.append(next_count_col)
  for csv_path in csv_paths[1:]:
    next_count_df, next_count_col = _read_one_count_csv(csv_path)
    count_cols.append(next_count_col)
    count_df = count_df.merge(next_count_df, on=WELL_INDEX_COLS,
                              how='outer')

  cols_to_fill = {}
  for col in count_cols:
    cols_to_fill[col] = 0
  count_df.fillna(cols_to_fill, inplace=True)

  io_lib.write_csv(count_df, output_filename)

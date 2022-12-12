r"""Write images to tensorstore.

Sample commands:

Set up a python environment:
python3 -m venv env
source env/bin/activate

Install the cell_img package:
git clone https://github.com/google/cell_img
pip install -e cell_img

Run the image grid pipeline to create a new tensorstore:
python cell_img/cell_img/image_grid/ts_write_main.py \
--tensorstore_path=gs://BUCKET/PATH/TO/OUTPUT/TENSORSTORE' \
--image_metadata_path=gs://BUCKET/PATH/TO/INPUT/image_metadata.csv \
--image_path_col=image_path \
--create_new_tensorstore=True \
--axes=Y,X,stain \
--x_axis_wrap=plate,well_col,site_col \
--y_axis_wrap=batch,well_row,site_row

Expand the existing tensorstore and write a new batch and plate:
python cell_img/cell_img/image_grid/ts_write_main.py \
--tensorstore_path=gs://BUCKET/PATH/TO/OUTPUT/TENSORSTORE' \
--image_metadata_path=gs://BUCKET/PATH/TO/INPUT/image_metadata.csv \
--image_path_col=image_path \
--allow_expansion_of_tensorstore=True

Add these flags to run the pipeline on cloud dataflow:
--project=PROJECT \
--bucket=STAGING_BUCKET \
--region=REGION \
--run_on_cloud=True
"""

import copy
from typing import Sequence

from absl import app
from absl import flags

from apache_beam.options import pipeline_options
from cell_img.image_grid import ts_write_main_lib

flags.DEFINE_string('tensorstore_path', None, 'Path for the tensorstore.')
flags.DEFINE_boolean(
    'create_new_tensorstore', False,
    'Creates a new tensorstore if True. If one already exists at'
    'the given path it will be deleted.')
flags.DEFINE_boolean(
    'allow_expansion_of_tensorstore', False,
    'Allow the existing tensorstore expanded if needed to write the images.')
flags.DEFINE_list('image_metadata_path', None,
                  'The list of paths for the image metadata csv.')
flags.DEFINE_string('override_coord_arrays_path', None,
                    'Path for the coordinate arrays override json file.')
flags.DEFINE_string(
    'image_path_col', None,
    'Column name for the image path in the image metadata csv.')
flags.DEFINE_list('axes', None, 'Ordered list of axes for a new tensorstore.')
flags.DEFINE_list('x_axis_wrap', None, 'Ordered list of axis to wrap in x.')
flags.DEFINE_list('y_axis_wrap', None, 'Ordered list of axis to wrap in y.')
flags.DEFINE_boolean('run_on_cloud', False, 'Run on cloud dataflow.')
flags.DEFINE_string('project', None, 'Cloud project.')
flags.DEFINE_string('bucket', None, 'Cloud bucket.')
flags.DEFINE_string('region', None, 'Cloud region e.g. us-west1.')


def get_pipeline_options(project, bucket, region):
  """Returns cloud dataflow pipeline options."""
  options = pipeline_options.PipelineOptions(flags=[
      '--setup_file',
      'cell_img/cell_img/image_grid/setup.py',
      '--runner',
      'DataflowRunner',
      # Flag use_runner_v2 avoids a segfault when worker pool starts.
      # Probably not needed long term.
      '--experiments',
      'use_runner_v2',
  ])
  options.view_as(pipeline_options.SetupOptions).save_main_session = False
  options.view_as(pipeline_options.GoogleCloudOptions).project = project
  options.view_as(pipeline_options.GoogleCloudOptions).region = region
  dataflow_gcs_location = 'gs://%s/dataflow' % bucket
  options.view_as(pipeline_options.GoogleCloudOptions
                 ).staging_location = '%s/staging' % dataflow_gcs_location
  options.view_as(pipeline_options.GoogleCloudOptions
                 ).temp_location = '%s/temp' % dataflow_gcs_location
  return options


def main(argv: Sequence[str]):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # The main session will be pickled by beam and sent to the dataflow workers.
  # In the absl library you can't pickle FlagValues. Therefore, we have make a
  # deepcopy of each one. We also can't have FLAGS = flags.FLAGS in the outer
  # scope.
  tensorstore_path = copy.deepcopy(flags.FLAGS.tensorstore_path)
  create_new_tensorstore = copy.deepcopy(flags.FLAGS.create_new_tensorstore)
  allow_expansion_of_tensorstore = copy.deepcopy(
      flags.FLAGS.allow_expansion_of_tensorstore)
  image_metadata_path = copy.deepcopy(flags.FLAGS.image_metadata_path)
  override_coord_arrays_path = copy.deepcopy(
      flags.FLAGS.override_coord_arrays_path)
  image_path_col = copy.deepcopy(flags.FLAGS.image_path_col)
  axes = copy.deepcopy(flags.FLAGS.axes)
  x_axis_wrap = copy.deepcopy(flags.FLAGS.x_axis_wrap)
  y_axis_wrap = copy.deepcopy(flags.FLAGS.y_axis_wrap)
  run_on_cloud = copy.deepcopy(flags.FLAGS.run_on_cloud)
  project = copy.deepcopy(flags.FLAGS.project)
  bucket = copy.deepcopy(flags.FLAGS.bucket)
  region = copy.deepcopy(flags.FLAGS.region)

  options = get_pipeline_options(project, bucket,
                                 region) if run_on_cloud else None

  pipeline_result = ts_write_main_lib.run_write_to_ts_pipeline(
      tensorstore_path, create_new_tensorstore, allow_expansion_of_tensorstore,
      image_metadata_path, image_path_col, axes, x_axis_wrap, y_axis_wrap,
      override_coord_arrays_path, options)

  error_counters = ''
  for metric_result in pipeline_result.metrics().query()['counters']:
    counter_name = metric_result.key.metric.name
    namespace = metric_result.key.metric.namespace
    value = metric_result.result
    counter_string = '%s.%s: %s\n' % (namespace, counter_name, value)
    if 'ERROR' in counter_name:
      error_counters += counter_string
  if error_counters:
    raise ValueError(
        'Pipeline finished running but had ERROR counters: %s' %
        error_counters)


if __name__ == '__main__':
  app.run(main)

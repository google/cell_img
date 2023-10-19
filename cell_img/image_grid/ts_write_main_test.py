"""Tests for cell_img.image_grid.ts_write_main."""

import os

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from cell_img.common import io_lib
from cell_img.image_grid import downsample_lib
from cell_img.image_grid import test_data
from cell_img.image_grid import ts_index_lib
from cell_img.image_grid import ts_write_lib
from cell_img.image_grid import ts_write_main
import numpy as np
import tensorstore as ts

FLAGS = flags.FLAGS

FULL_IMAGE_METADATA_DF = test_data.generate_image_metadata_df()


class TsWriteMainTest(absltest.TestCase):

  def setUp(self):
    self.saved_flags = flagsaver.save_flag_values()
    self.tensorstore_dir = self.create_tempdir('tensorstore')
    # Force a tiny min size to ensure downsampling code is exercised.
    downsample_lib.YX_DOWNSAMPLE_MIN_SIZE = 5
    super().setUp()

  def tearDown(self):
    flagsaver.restore_flag_values(self.saved_flags)
    super().tearDown()

  def test_create_new_tensorstore(self):
    self.run_main(
        'batch=="batch1" and plate=="plate1"',
        create_new_tensorstore=True,
        allow_expansion_of_tensorstore=False)

  def test_enlarge_existing_tensorstore(self):
    # Create a tensorstore.
    # B1P1
    self.run_main(
        'batch=="batch1" and plate=="plate1"',
        create_new_tensorstore=True,
        allow_expansion_of_tensorstore=False)
    # Enlarge the existing tensorstore.
    # B1P1, XXXX
    # XXXX, B2P2
    self.run_main(
        'batch=="batch2" and plate=="plate2"',
        create_new_tensorstore=False,
        allow_expansion_of_tensorstore=True)

  def test_write_new_data_without_enlarging(self):
    # Create a tensorstore.
    # B1P1, XXXX
    # XXXX, B2P2
    self.run_main(
        '(batch=="batch1" and plate=="plate1") or '
        '(batch=="batch2" and plate=="plate2")',
        create_new_tensorstore=True,
        allow_expansion_of_tensorstore=False)
    # Write to the existing tensorstore without allowing expansion.
    # B1P1, XXXX
    # B2P1, B2P2
    self.run_main(
        'batch=="batch2" and plate=="plate1"',
        create_new_tensorstore=False,
        allow_expansion_of_tensorstore=False)

  def test_write_new_data_without_enlarging_fails(self):
    # Create a tensorstore.
    # B1P1
    self.run_main(
        'batch=="batch1" and plate=="plate1"',
        create_new_tensorstore=True,
        allow_expansion_of_tensorstore=False)
    with self.assertRaisesRegex(
        ValueError,
        'Cannot write the new images without expanding the tensorstore first.*'
    ):
      self.run_main(
          'batch=="batch2" and plate=="plate1"',
          create_new_tensorstore=False,
          allow_expansion_of_tensorstore=False)

  def test_create_new_with_coord_array_override(self):
    # Change the default order of stains and leave room for an extra stain.
    coordinate_arrays_override = {'stain': ['stain2', 'extra_stain', 'stain1']}
    self.run_main(
        'batch=="batch1" and plate=="plate1"',
        create_new_tensorstore=True,
        allow_expansion_of_tensorstore=False,
        coordinate_arrays_override=coordinate_arrays_override)
    tensorstore_metadata = read_s0_metadata(FLAGS.tensorstore_path)
    self.assertSameStructure(['stain2', 'extra_stain', 'stain1'],
                             tensorstore_metadata['coordinateArrays']['stain'])

  def test_write_existing_with_coord_array_override_fails(self):
    # Check that you cannot write to an existing tensorstore and provide an
    # coordinate array override if you don't allow expansion of the tensorstore.

    # Create a tensorstore.
    self.run_main(
        '(batch=="batch1" and plate=="plate1") or '
        '(batch=="batch2" and plate=="plate2")',
        create_new_tensorstore=True,
        allow_expansion_of_tensorstore=False)
    # Write to the existing tensorstore (without allowing expansion) and provide
    # coordinate_arrays_override.
    coordinate_arrays_override = {'stain': ['stain1', 'stain2', 'new_stain']}
    with self.assertRaisesRegex(
        ValueError, 'Flag override_coord_arrays_path must only be set if '
        'create_new_tensorstore is True or both create_new_tensorstore is '
        'False and allow_expansion_of_tensorstore is True.'):
      self.run_main(
          'batch=="batch2" and plate=="plate1"',
          create_new_tensorstore=False,
          allow_expansion_of_tensorstore=False,
          coordinate_arrays_override=coordinate_arrays_override)

  def test_enlarge_existing_with_coord_array_override(self):
    # Create a tensorstore.
    # B1P1
    self.run_main(
        'batch=="batch1" and plate=="plate1"',
        create_new_tensorstore=True,
        allow_expansion_of_tensorstore=False)
    # Enlarge the existing tensorstore.
    # Use an override to specify the order of the two new plates.
    # Also use the override to leave a gap for an extra plate
    # B1P1, XXXX, XXXX, XXXX
    # XXXX, B2P3, XXXX, B2P2
    coordinate_arrays_override = {
        'plate': ['plate1', 'plate3', 'extra_plate', 'plate2']
    }
    self.run_main(
        'batch=="batch2" and (plate=="plate2" or plate == "plate3")',
        create_new_tensorstore=False,
        allow_expansion_of_tensorstore=True,
        coordinate_arrays_override=coordinate_arrays_override)
    tensorstore_metadata = read_s0_metadata(FLAGS.tensorstore_path)
    self.assertSameStructure(['plate1', 'plate3', 'extra_plate', 'plate2'],
                             tensorstore_metadata['coordinateArrays']['plate'])

  def test_without_axes_wrap(self):
    # Create a tensorstore where exerything is a separate dimension.
    FLAGS.axes = [
        'Y', 'X', 'stain', 'plate', 'well_col', 'site_col', 'batch', 'well_row',
        'site_row'
    ]
    FLAGS.x_axis_wrap = []
    FLAGS.y_axis_wrap = []

    input_subset = 'batch=="batch1" and plate=="plate1"'
    image_metadata_df = FULL_IMAGE_METADATA_DF.query(input_subset)
    image_metadata_csv_path = test_data.write_image_metadata_csv_and_images(
        self.create_tempdir().full_path, image_metadata_df)

    FLAGS.image_metadata_path = image_metadata_csv_path
    FLAGS.tensorstore_path = self.tensorstore_dir.full_path
    FLAGS.image_path_col = test_data.IMAGE_PATH_COL
    FLAGS.create_new_tensorstore = True
    FLAGS.allow_expansion_of_tensorstore = False

    ts_write_main.main(argv=[])

    self.assert_image_data_in_tensorstore(image_metadata_df,
                                          FLAGS.tensorstore_path)

    # We do not expect downsampled levels since YX dimensions are small.
    downsampling_factors = downsample_lib.read_downsampling_factors(
        FLAGS.tensorstore_path)
    self.assertLen(downsampling_factors, 1)

  def test_behavior_on_missing_images(self):
    with self.assertRaisesRegex(
        ValueError, 'Pipeline finished running but had ERROR counters.*'
        'ERROR-get_view_slice_and_read_image-SEE_LOGS: 1'):
      self.run_main(
          'batch=="batch1" and plate=="plate1"',
          create_new_tensorstore=True,
          allow_expansion_of_tensorstore=False,
          delete_an_input_image=True)

  def run_main(self,
               input_subset,
               create_new_tensorstore,
               allow_expansion_of_tensorstore,
               coordinate_arrays_override=None,
               delete_an_input_image=False):
    # Write csv and images input data to temp dir.
    image_metadata_df = FULL_IMAGE_METADATA_DF.query(input_subset)
    image_metadata_csv_path = test_data.write_image_metadata_csv_and_images(
        self.create_tempdir().full_path, image_metadata_df)
    if delete_an_input_image:
      image_path = image_metadata_df[test_data.IMAGE_PATH_COL].iloc[-1]
      os.remove(image_path)
      to_check_image_metadata_df = image_metadata_df.iloc[:-1, :]
    else:
      to_check_image_metadata_df = image_metadata_df

    if coordinate_arrays_override:
      override_coord_arrays_path = self.create_tempfile().full_path
      io_lib.write_json_file(coordinate_arrays_override,
                             override_coord_arrays_path)
    else:
      override_coord_arrays_path = None

    FLAGS.image_metadata_path = image_metadata_csv_path
    FLAGS.tensorstore_path = self.tensorstore_dir.full_path
    FLAGS.image_path_col = test_data.IMAGE_PATH_COL
    FLAGS.create_new_tensorstore = create_new_tensorstore
    FLAGS.allow_expansion_of_tensorstore = allow_expansion_of_tensorstore
    FLAGS.override_coord_arrays_path = override_coord_arrays_path

    if create_new_tensorstore:
      FLAGS.axes = test_data.AXES
      FLAGS.x_axis_wrap = test_data.AXES_WRAP['X']
      FLAGS.y_axis_wrap = test_data.AXES_WRAP['Y']
    else:
      FLAGS.axes = None
      FLAGS.x_axis_wrap = None
      FLAGS.y_axis_wrap = None

    ts_write_main.main(argv=[])

    self.assert_image_data_in_tensorstore(to_check_image_metadata_df,
                                          FLAGS.tensorstore_path)

    # Check that all the downsampled data matches.
    downsampling_factors = downsample_lib.read_downsampling_factors(
        FLAGS.tensorstore_path)
    self.assertGreater(len(downsampling_factors), 1)
    delta_factors = downsample_lib.get_delta_factors(downsampling_factors)
    spec_without_metadata = ts_write_lib.create_spec_from_path(
        downsample_lib.join_downsample_level_to_path(FLAGS.tensorstore_path, 0))
    prev_spec = spec_without_metadata
    prev_store = ts_write_lib.open_existing_tensorstore(spec_without_metadata)
    for delta_factor in delta_factors:
      downsampled = ts.downsample(
          prev_store, delta_factor, method=downsample_lib.DOWNSAMPLING_METHOD)
      next_spec = downsample_lib.inc_downsample_level_in_spec(prev_spec)
      next_store = ts_write_lib.open_existing_tensorstore(next_spec)

      np.testing.assert_equal(downsampled.read().result(),
                              next_store.read().result())

      prev_spec = next_spec
      prev_store = next_store

  def assert_image_data_in_tensorstore(self, image_metadata_df,
                                       tensorstore_path):
    # Check that an input image matches the data read from tensorstore.
    image_metadata = image_metadata_df.iloc[-1, :]
    image_path = image_metadata[test_data.IMAGE_PATH_COL]
    input_image = io_lib.read_image(image_path)

    # Open the s0 level tensorstore to see if the data matches.
    path_s0 = downsample_lib.join_downsample_level_to_path(tensorstore_path, 0)
    spec_without_metadata = ts_write_lib.create_spec_from_path(path_s0)
    tensorstore_dataset = ts_write_lib.open_existing_tensorstore(
        spec_without_metadata)
    ts_index = ts_index_lib.index_from_spec(
        tensorstore_dataset.spec().to_json())

    whole_image_slice = ts_index.get_whole_image_slice(image_metadata)
    read_result = tensorstore_dataset[whole_image_slice].read().result()
    np.testing.assert_array_equal(input_image, read_result)


def read_s0_metadata(tensorstore_root_path):
  path_s0 = downsample_lib.join_downsample_level_to_path(
      tensorstore_root_path, 0)
  spec_without_metadata = ts_write_lib.create_spec_from_path(path_s0)
  tensorstore_dataset = ts_write_lib.open_existing_tensorstore(
      spec_without_metadata)
  return tensorstore_dataset.spec().to_json()['metadata']


if __name__ == '__main__':
  FLAGS.eventmanager_default_stack_size = 512 * 1024
  absltest.main()

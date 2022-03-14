"""Tests for cell_img.image_grid.ts_metadata_lib."""

import copy

from absl.testing import absltest
from cell_img.image_grid import test_data
from cell_img.image_grid import ts_index_lib
from cell_img.image_grid import ts_metadata_lib
import pandas as pd


class TsMetadataLibTest(absltest.TestCase):

  def test_coordinate_arrays_from_dataframe(self):
    image_metadata_df = test_data.generate_image_metadata_df()
    coordinate_arrays = ts_metadata_lib.coordinate_arrays_from_dataframe(
        image_metadata_df, test_data.AXES, test_data.AXES_WRAP)
    self.assertCountEqual(coordinate_arrays['batch'], test_data.BATCH_LIST)
    self.assertCountEqual(coordinate_arrays['plate'], test_data.PLATE_LIST)
    self.assertCountEqual(coordinate_arrays['well_row'],
                          test_data.WELL_ROW_LIST)
    self.assertCountEqual(coordinate_arrays['well_col'],
                          test_data.WELL_COL_LIST)
    self.assertCountEqual(coordinate_arrays['site_row'],
                          test_data.SITE_ROW_LIST)
    self.assertCountEqual(coordinate_arrays['site_col'],
                          test_data.SITE_COL_LIST)
    self.assertCountEqual(coordinate_arrays['stain'], test_data.STAIN_LIST)

  def test_coordinate_arrays_from_dataframe_raises_when_not_string(self):
    image_metadata_df = test_data.generate_image_metadata_df()
    image_metadata_df['well_col'] = image_metadata_df['well_col'].astype(int)
    with self.assertRaisesRegex(
        ValueError, 'Values of a coordinate array should all be strings.*'):
      ts_metadata_lib.coordinate_arrays_from_dataframe(image_metadata_df,
                                                       test_data.AXES,
                                                       test_data.AXES_WRAP)

  def test_create_tensorstore_metadata(self):
    image_metadata_df = test_data.generate_image_metadata_df()
    coordinate_arrays = ts_metadata_lib.coordinate_arrays_from_dataframe(
        image_metadata_df, test_data.AXES, test_data.AXES_WRAP)

    tensorstore_metadata = ts_metadata_lib.create_tensorstore_metadata(
        test_data.AXES, test_data.AXES_WRAP, coordinate_arrays,
        test_data.UNIT_SIZES, test_data.IMAGE_DTYPE)
    self.assertCountEqual(tensorstore_metadata.keys(), [
        'axes', 'axesWrap', 'coordinateArrays', 'unitSizes', 'dimensions',
        'dataType', 'blockSize'
    ])

  def test_get_new_coordinate_values_add_new_batch(self):
    all_image_metadata_df = test_data.generate_image_metadata_df()
    batch_1_df = all_image_metadata_df.query('batch == "batch1"')

    tensorstore_metadata = tensorstore_metadata_from_df(batch_1_df)

    batch_2_df = all_image_metadata_df.query('batch == "batch2"')
    axis_to_values_to_add = ts_metadata_lib.get_new_coordinate_values(
        tensorstore_metadata, batch_2_df)
    self.assertDictEqual(axis_to_values_to_add, {'batch': ['batch2']})

  def test_get_new_coordinate_values_add_new_stain(self):
    all_image_metadata_df = test_data.generate_image_metadata_df()
    stain_1_df = all_image_metadata_df.query('stain == "stain1"')

    tensorstore_metadata = tensorstore_metadata_from_df(stain_1_df)

    stain_2_df = all_image_metadata_df.query('stain == "stain2"')
    axis_to_values_to_add = ts_metadata_lib.get_new_coordinate_values(
        tensorstore_metadata, stain_2_df)
    self.assertDictEqual(axis_to_values_to_add, {'stain': ['stain2']})

  def test_get_new_coordinate_values_add_new_plates(self):
    all_image_metadata_df = test_data.generate_image_metadata_df()
    plate_1_df = all_image_metadata_df.query('plate == "plate1"')

    tensorstore_metadata = tensorstore_metadata_from_df(plate_1_df)

    plate_2_and_3_df = all_image_metadata_df.query('plate != "plate1"')
    axis_to_values_to_add = ts_metadata_lib.get_new_coordinate_values(
        tensorstore_metadata, plate_2_and_3_df)
    self.assertDictEqual(axis_to_values_to_add, {'plate': ['plate2', 'plate3']})

  def test_get_new_coordinate_values_empty_when_nothing_to_add(self):
    image_metadata_df = test_data.generate_image_metadata_df()
    tensorstore_metadata = tensorstore_metadata_from_df(image_metadata_df)
    axis_to_values_to_add = ts_metadata_lib.get_new_coordinate_values(
        tensorstore_metadata, image_metadata_df)
    self.assertEmpty(axis_to_values_to_add)

  def test_get_new_coordinate_values_raises_when_cols_missing(self):
    image_metadata_df = test_data.generate_image_metadata_df()
    tensorstore_metadata = tensorstore_metadata_from_df(image_metadata_df)
    bad_image_metadata_df = image_metadata_df.drop(columns=['plate', 'batch'])
    with self.assertRaisesRegex(
        ValueError, 'The following columns were missing '
        r"from the dataframe: \['batch', 'plate'\]"):
      ts_metadata_lib.get_new_coordinate_values(tensorstore_metadata,
                                                bad_image_metadata_df)

  def test_expand_tensorstore_metadata_new_stain(self):
    # Test that we can expand a tensorstore to accommodate new stains.
    tensorstore_metadata = tensorstore_metadata_from_df(
        test_data.generate_image_metadata_df())
    axis_to_values_to_add = {'stain': ['new_stainA', 'new_stainB']}
    new_tensorstore_metadata = ts_metadata_lib.expand_tensorstore_metadata(
        tensorstore_metadata, axis_to_values_to_add)

    # Assert that the stain dimensions size has increased by 2.
    self.assertEqual(new_tensorstore_metadata['axes'][2], 'stain')
    orig_stain_dim_size = tensorstore_metadata['dimensions'][2]
    new_stain_dim_size = new_tensorstore_metadata['dimensions'][2]
    self.assertEqual(new_stain_dim_size - orig_stain_dim_size, 2)

    # Assert that the new stains are appended to the end.
    orig_stain_values = tensorstore_metadata['coordinateArrays']['stain']
    self.assertEqual(orig_stain_values, ['stain1', 'stain2'])
    new_stain_values = new_tensorstore_metadata['coordinateArrays']['stain']
    self.assertEqual(new_stain_values,
                     ['stain1', 'stain2', 'new_stainA', 'new_stainB'])

    # The tensorstore metadata should be otherwise identical.
    subset_tensorstore_metadata = copy.deepcopy(tensorstore_metadata)
    del subset_tensorstore_metadata['dimensions']
    del subset_tensorstore_metadata['coordinateArrays']['stain']
    subset_new_tensorstore_metadata = copy.deepcopy(new_tensorstore_metadata)
    del subset_new_tensorstore_metadata['dimensions']
    del subset_new_tensorstore_metadata['coordinateArrays']['stain']
    self.assertSameStructure(subset_tensorstore_metadata,
                             subset_new_tensorstore_metadata)

  def test_expand_tensorstore_metadata_raises_when_inner_wrapped_axis(self):
    tensorstore_metadata = tensorstore_metadata_from_df(
        test_data.generate_image_metadata_df())
    for axis_name in ['well_col', 'site_col', 'well_row', 'site_row']:
      with self.assertRaisesRegex(
          ValueError,
          'Cannot add new values to inner wrapped coordinate arrays.*'):
        axis_to_values_to_add = {axis_name: ['new_value']}
        ts_metadata_lib.expand_tensorstore_metadata(tensorstore_metadata,
                                                    axis_to_values_to_add)

  def test_expand_tensorstore_metadata_raises_when_non_existent(self):
    tensorstore_metadata = tensorstore_metadata_from_df(
        test_data.generate_image_metadata_df())
    with self.assertRaisesRegex(
        ValueError,
        'Cannot add new values to non-existent coordinate arrays.*'):
      axis_to_values_to_add = {'non_existent_axis': ['new_value']}
      ts_metadata_lib.expand_tensorstore_metadata(tensorstore_metadata,
                                                  axis_to_values_to_add)

  def test_expand_tensorstore_metadata_new_batches_and_plates(self):
    # Test we can expand a tensorstore to accommodate new batches and plates.
    tensorstore_metadata = tensorstore_metadata_from_df(
        test_data.generate_image_metadata_df())
    axis_to_values_to_add = {
        'batch': ['batch3', 'batch4', 'batch5'],
        'plate': ['plate4', 'plate5']
    }
    new_tensorstore_metadata = ts_metadata_lib.expand_tensorstore_metadata(
        tensorstore_metadata, axis_to_values_to_add)

    # Assert that the stain dimensions size has increased appropriately.
    self.assertEqual(tensorstore_metadata['axes'], ['Y', 'X', 'stain'])
    orig_dim_y, orig_dim_x, orig_dim_stain = tensorstore_metadata['dimensions']
    new_dim_y, new_dim_x, new_dim_stain = new_tensorstore_metadata['dimensions']
    self.assertEqual(new_dim_x / orig_dim_x, 5 / 3)  # batch
    self.assertEqual(new_dim_y / orig_dim_y, 5 / 2)  # plate
    self.assertEqual(new_dim_stain / orig_dim_stain, 1)

    # Assert that the new values are appended to the end.
    self.assertEqual(new_tensorstore_metadata['coordinateArrays']['batch'],
                     ['batch1', 'batch2', 'batch3', 'batch4', 'batch5'])
    self.assertEqual(new_tensorstore_metadata['coordinateArrays']['plate'],
                     ['plate1', 'plate2', 'plate3', 'plate4', 'plate5'])

    # The tensorstore metadata should be otherwise identical.
    subset_tensorstore_metadata = copy.deepcopy(tensorstore_metadata)
    del subset_tensorstore_metadata['dimensions']
    del subset_tensorstore_metadata['coordinateArrays']['batch']
    del subset_tensorstore_metadata['coordinateArrays']['plate']
    subset_new_tensorstore_metadata = copy.deepcopy(new_tensorstore_metadata)
    del subset_new_tensorstore_metadata['dimensions']
    del subset_new_tensorstore_metadata['coordinateArrays']['batch']
    del subset_new_tensorstore_metadata['coordinateArrays']['plate']
    self.assertSameStructure(subset_tensorstore_metadata,
                             subset_new_tensorstore_metadata)

  def test_coords_unchanged_after_expansion(self):
    all_image_metadata_df = test_data.generate_image_metadata_df()

    # Make a tensorstore with batch 2 and remember the coords for each image.
    batch2_df = all_image_metadata_df.query('batch == "batch2"')
    orig_tensorstore_metadata = tensorstore_metadata_from_df(batch2_df)
    orig_index = ts_index_lib.index_from_tensorstore_metadata(
        orig_tensorstore_metadata)
    orig_coords_df = pd.DataFrame.from_dict(
        orig_index.get_coords_dict(batch2_df))

    # Enlarge the existing tensorstore to add batch 1.
    batch1_df = all_image_metadata_df.query('batch == "batch1"')
    axis_to_values_to_add = ts_metadata_lib.get_new_coordinate_values(
        orig_tensorstore_metadata, batch1_df)
    new_tensorstore_metadata = ts_metadata_lib.expand_tensorstore_metadata(
        orig_tensorstore_metadata, axis_to_values_to_add)
    # The later batch is appended and should not affect the coordinates of the
    # existing data.
    self.assertEqual(['batch2', 'batch1'],
                     new_tensorstore_metadata['coordinateArrays']['batch'])

    # Assert that the coordinates are the same in the enlarged tensorstore.
    new_index = ts_index_lib.index_from_tensorstore_metadata(
        new_tensorstore_metadata)
    new_coords_df = pd.DataFrame.from_dict(new_index.get_coords_dict(batch2_df))
    pd.testing.assert_frame_equal(orig_coords_df, new_coords_df)

  def test_expand_tensorstore_simple(self):
    orig_tensorstore_metadata = {
        'axes': ['my_axis'],
        'axesWrap': {},
        'unitSizes': {
            'my_axis': 100
        },
        'coordinateArrays': {
            'my_axis': ['one', 'two']
        },
        'dimensions': [100],
    }
    axis_to_values_to_add = {'my_axis': ['three', 'four']}
    new_tensorstore_metadata = ts_metadata_lib.expand_tensorstore_metadata(
        orig_tensorstore_metadata, axis_to_values_to_add)
    self.assertSameStructure(
        new_tensorstore_metadata, {
            'axes': ['my_axis'],
            'axesWrap': {},
            'unitSizes': {
                'my_axis': 100
            },
            'coordinateArrays': {
                'my_axis': ['one', 'two', 'three', 'four']
            },
            'dimensions': [400],
        })

  def test_expand_tensorstore_metadata_with_coordinate_arrays_override(self):
    orig_tensorstore_metadata = {
        'axes': ['my_axis'],
        'axesWrap': {},
        'unitSizes': {
            'my_axis': 100
        },
        'coordinateArrays': {
            'my_axis': ['one', 'two']
        },
        'dimensions': [100],
    }
    axis_to_values_to_add = {'my_axis': ['three', 'four']}
    # Override should be able to change the order for values to add as well as
    # put in extra values.
    override_coord_arrays = {
        'my_axis': ['one', 'two', 'extra', 'four', 'three']
    }

    new_tensorstore_metadata = ts_metadata_lib.expand_tensorstore_metadata(
        orig_tensorstore_metadata, axis_to_values_to_add, override_coord_arrays)
    self.assertSameStructure(
        new_tensorstore_metadata, {
            'axes': ['my_axis'],
            'axesWrap': {},
            'unitSizes': {
                'my_axis': 100
            },
            'coordinateArrays': {
                'my_axis': ['one', 'two', 'extra', 'four', 'three']
            },
            'dimensions': [500],
        })

  def test_expand_tensorstore_metadata_raises_override_tries_reordering(self):
    orig_tensorstore_metadata = {
        'axes': ['my_axis'],
        'axesWrap': {},
        'unitSizes': {
            'my_axis': 100
        },
        'coordinateArrays': {
            'my_axis': ['one', 'two']  # Existing order cannot be changed.
        },
        'dimensions': [100],
    }
    axis_to_values_to_add = {'my_axis': ['three', 'four']}
    # Override tries to change the order for existing values.
    override_coord_arrays = {
        'my_axis': ['two', 'one', 'extra', 'four', 'three']
    }
    with self.assertRaisesRegex(
        ValueError,
        'When expanding and overriding coordinate arrays, you cannot change '
        'the order of existing values in any coordinate arrays.*'):
      ts_metadata_lib.expand_tensorstore_metadata(orig_tensorstore_metadata,
                                                  axis_to_values_to_add,
                                                  override_coord_arrays)

  def test_apply_coordinate_arrays_override_appends_new_values(self):
    # You should be able to use an override to add coordinate values that do
    # not exist in the original coordinate array.
    orig_coord_arrays = {'a': ['a1', 'a2']}
    override_coord_arrays = {'a': ['a1', 'a2', 'new_value_not_present_in_orig']}
    res = ts_metadata_lib.apply_coordinate_arrays_override(
        orig_coord_arrays, override_coord_arrays)
    self.assertSameStructure(
        res, {'a': ['a1', 'a2', 'new_value_not_present_in_orig']})

  def test_apply_coordinate_arrays_override_reorders_values(self):
    # Default value order is alphabetical. We should be able to override that.
    orig_coord_arrays = {'a': ['one', 'three', 'two']}
    override_coord_arrays = {'a': ['one', 'two', 'three']}
    res = ts_metadata_lib.apply_coordinate_arrays_override(
        orig_coord_arrays, override_coord_arrays)
    self.assertSameStructure(res, {'a': ['one', 'two', 'three']})

  def test_apply_coordinate_arrays_override_ignores_non_overridden_axes(self):
    orig_coord_arrays = {
        'non_overridden': ['a1', 'a2'],
        'overridden': ['b1', 'b2']
    }
    override_coord_arrays = {'overridden': ['b2', 'b1']}
    res = ts_metadata_lib.apply_coordinate_arrays_override(
        orig_coord_arrays, override_coord_arrays)
    self.assertSameStructure(res, {
        'non_overridden': ['a1', 'a2'],
        'overridden': ['b2', 'b1']
    })

  def test_apply_coordinate_arrays_override_raises_on_missing_values(self):
    orig_coord_arrays = {'my_axis': ['all', 'must', 'be', 'present']}
    override_coord_arrays = {'my_axis': ['not', 'all', 'present']}
    with self.assertRaisesRegex(
        ValueError,
        'Override coordinate array values should contain all values in the '
        'original array.*'):
      ts_metadata_lib.apply_coordinate_arrays_override(orig_coord_arrays,
                                                       override_coord_arrays)

  def test_apply_coordinate_arrays_override_raises_on_extra_axes(self):
    orig_coord_arrays = {'my_axis': ['a1', 'a2']}
    override_coord_arrays = {'unknown_axis': ['b1', 'b2']}
    with self.assertRaisesRegex(
        ValueError,
        'Override coordinate array should only contain keys present in the '
        'original.*'):
      ts_metadata_lib.apply_coordinate_arrays_override(orig_coord_arrays,
                                                       override_coord_arrays)

  def test_apply_coordinate_arrays_override_raises_on_duplicates(self):
    orig_coord_arrays = {'my_axis': ['a1', 'a2']}
    override_coord_arrays = {'my_axis': ['a2', 'a1', 'duplicate', 'duplicate']}
    with self.assertRaisesRegex(
        ValueError, 'Coordinate array values cannot contain duplicates.*'):
      ts_metadata_lib.apply_coordinate_arrays_override(orig_coord_arrays,
                                                       override_coord_arrays)


def tensorstore_metadata_from_df(image_metadata_df):
  coordinate_arrays = ts_metadata_lib.coordinate_arrays_from_dataframe(
      image_metadata_df, test_data.AXES, test_data.AXES_WRAP)
  return ts_metadata_lib.create_tensorstore_metadata(test_data.AXES,
                                                     test_data.AXES_WRAP,
                                                     coordinate_arrays,
                                                     test_data.UNIT_SIZES,
                                                     test_data.IMAGE_DTYPE)


if __name__ == '__main__':
  absltest.main()

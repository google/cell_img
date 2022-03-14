"""Tests for cell_img.image_grid.ts_index_lib."""

import copy

from absl.testing import absltest
from cell_img.image_grid import ts_index_lib
import numpy as np
import pandas as pd


class Single1DTest(absltest.TestCase):

  def test_get_index_individually(self):
    lookup = ts_index_lib._Single1D(
        ['zero', 'one', 'two', 'three', 'four', 'five'])
    self.assertEqual(lookup.get_index('five'), 5)
    self.assertEqual(lookup.get_index('one'), 1)
    self.assertEqual(lookup.get_index('four'), 4)
    self.assertEqual(lookup.get_index('three'), 3)
    self.assertEqual(lookup.get_index('two'), 2)
    self.assertEqual(lookup.get_index('zero'), 0)

  def test_get_index_multiple(self):
    lookup = ts_index_lib._Single1D(
        ['zero', 'one', 'two', 'three', 'four', 'five'])
    result = lookup.get_index(['two', 'five'])
    np.testing.assert_equal(result, [2, 5])

    result = lookup.get_index(np.array(['two', 'five']))
    np.testing.assert_equal(result, [2, 5])

    result = lookup.get_index(pd.Series(['two', 'five']))
    np.testing.assert_equal(result, [2, 5])

  def test_get_index_non_existent_raises(self):
    lookup = ts_index_lib._Single1D(
        ['zero', 'one', 'two', 'three', 'four', 'five'])
    with self.assertRaisesRegex(
        ValueError, 'Not all values of the query are in the index.'):
      lookup.get_index('six')
    with self.assertRaisesRegex(
        ValueError, 'Not all values of the query are in the index.'):
      lookup.get_index(['two', 'five', 'six'])

  def test_get_value_individually(self):
    lookup = ts_index_lib._Single1D(
        ['zero', 'one', 'two', 'three', 'four', 'five'])
    self.assertEqual(lookup.get_value(5), 'five')

  def test_get_value_multiple(self):
    lookup = ts_index_lib._Single1D(
        ['zero', 'one', 'two', 'three', 'four', 'five'])
    np.testing.assert_equal(lookup.get_value([2, 5]), ['two', 'five'])

  def test_bad_dimensions_raises(self):
    with self.assertRaisesRegex(ValueError, 'Expected 1D input but got.*'):
      ts_index_lib._Single1D([['A1', 'A2'], ['B1', 'B2']])

  def test_empty_raises(self):
    with self.assertRaisesRegex(ValueError, 'Input cannot be empty.'):
      ts_index_lib._Single1D([])

  def test_dim_length_one_works(self):
    lookup = ts_index_lib._Single1D(['one_value'])
    self.assertEqual(lookup.get_index('one_value'), 0)
    self.assertEqual(lookup.get_value(0), 'one_value')
    np.testing.assert_equal(
        lookup.get_index(['one_value', 'one_value']), [0, 0])
    np.testing.assert_equal(
        lookup.get_value([0, 0]), ['one_value', 'one_value'])


class Multi1DTest(absltest.TestCase):

  def get_three_level_lookup(self):
    coordinate_arrays = {
        'plate': ['plate0', 'plate1', 'plate2'],
        'well_col': ['1', '2', '3', '4', '5'],
        'site_col': ['site_col0', 'site_col1'],
    }
    coordinate_names = ['plate', 'well_col', 'site_col']

    lookup = ts_index_lib._Multi1D(coordinate_names, coordinate_arrays)
    return lookup

  def test_valid_missing_coords(self):
    valid_missing_coord_names = [['site_col',], ['site_col', 'well_col']]
    lookup = self.get_three_level_lookup()
    for coord_names in valid_missing_coord_names:
      # does not raise
      _ = lookup._check_valid_missing_coords(coord_names)

  def test_raises_when_not_smallest_coord(self):
    lookup = self.get_three_level_lookup()
    invalid_missing_coord_names = [['well_col',], ['plate']]
    for coord_names in invalid_missing_coord_names:
      with self.assertRaises(KeyError):
        lookup._check_valid_missing_coords(coord_names)

  def test_raises_when_all_missing(self):
    lookup = self.get_three_level_lookup()
    invalid_missing_coords = ['site_col', 'well_col', 'plate']
    with self.assertRaises(KeyError):
      lookup._check_valid_missing_coords(invalid_missing_coords)

  def test_raises_when_not_consecutive(self):
    lookup = self.get_three_level_lookup()
    invalid_missing_coords = ['site_col', 'plate']
    with self.assertRaises(KeyError):
      lookup._check_valid_missing_coords(invalid_missing_coords)

  def test_missing_coord_names_site(self):
    lookup = self.get_three_level_lookup()
    self.assertEqual(
        lookup._get_missing_coords({
            'plate': ['value1',],
            'well_col': ['value1',]
        }), ['site_col',])

  def test_missing_coord_names_site_well(self):
    lookup = self.get_three_level_lookup()
    self.assertEqual(
        lookup._get_missing_coords({
            'plate': ['value1',],
        }), ['well_col', 'site_col'])

  def test_get_well_level_smallest_block_size(self):
    lookup = self.get_three_level_lookup()
    well_level_labels = {'plate': ['plate0',], 'well_col': ['1',]}
    expected_block_size = 2  # 2 sites per well
    self.assertEqual(
        lookup.get_smallest_block_size(well_level_labels),
        expected_block_size)

  def test_get_plate_level_smallest_block_size(self):
    lookup = self.get_three_level_lookup()
    plate_level_labels = {'plate': ['plate0',]}
    expected_block_size = 2 * 5  # 2 sites per well, 5 wells per plate
    self.assertEqual(
        lookup.get_smallest_block_size(plate_level_labels),
        expected_block_size)

  def test_get_index_two_level(self):
    names = ['year', 'month']
    name_to_values = {
        'year': ['2019', '2020', '2021'],
        'month': [
            'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep',
            'oct', 'nov', 'dec'
        ]
    }
    lookup = ts_index_lib._Multi1D(names, name_to_values)
    self.assertEqual(lookup.get_index({'year': '2019', 'month': 'jan'}), 0)
    self.assertEqual(lookup.get_index({'year': '2019', 'month': 'aug'}), 7)
    self.assertEqual(lookup.get_index({'year': '2020', 'month': 'aug'}), 19)
    self.assertEqual(lookup.get_index({'year': '2021', 'month': 'aug'}), 31)
    self.assertEqual(lookup.get_index({'year': '2021', 'month': 'dec'}), 35)

    np.testing.assert_equal(
        lookup.get_index({
            'year': ['2019', '2019', '2020', '2021', '2021'],
            'month': ['jan', 'aug', 'aug', 'aug', 'dec']
        }), [0, 7, 19, 31, 35])

  def test_get_index_non_existent_raises(self):
    names = ['year', 'month']
    name_to_values = {
        'year': ['2019', '2020', '2021'],
        'month': [
            'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep',
            'oct', 'nov', 'dec'
        ]
    }
    lookup = ts_index_lib._Multi1D(names, name_to_values)
    with self.assertRaisesRegex(
        ValueError, 'Not all values of the query are in the index.'):
      lookup.get_index({'year': '2021', 'month': 'NON_EXISTENT'})
    with self.assertRaisesRegex(
        ValueError, 'Not all values of the query are in the index.'):
      lookup.get_index({
          'year': ['2020', '2021'],
          'month': ['feb', 'NON_EXISTENT']
      })

  def test_get_index_individually(self):
    names = ['x', 'y', 'z']
    name_to_values = {
        'x': ['x0', 'x1', 'x2'],
        'y': ['y0', 'y1', 'y2', 'y3'],
        'z': ['z0', 'z1']
    }
    lookup = ts_index_lib._Multi1D(names, name_to_values)
    self.assertEqual(lookup.get_index({'x': 'x0', 'y': 'y0', 'z': 'z0'}), 0)
    self.assertEqual(lookup.get_index({'x': 'x1', 'y': 'y2', 'z': 'z1'}), 13)
    self.assertEqual(lookup.get_index({'x': 'x2', 'y': 'y3', 'z': 'z1'}), 23)

  def test_get_index_multiple(self):
    names = ['x', 'y', 'z']
    name_to_values = {
        'x': ['x0', 'x1', 'x2'],
        'y': ['y0', 'y1', 'y2', 'y3'],
        'z': ['z0', 'z1']
    }
    lookup = ts_index_lib._Multi1D(names, name_to_values)
    results = lookup.get_index({
        'x': ['x0', 'x1'],
        'y': ['y0', 'y2'],
        'z': ['z0', 'z1']
    })
    np.testing.assert_equal(results, [0, 13])

  def test_get_index_exhaustive(self):
    names = ['x', 'y', 'z']
    name_to_values = {
        'x': ['x0', 'x1', 'x2'],
        'y': ['y0', 'y1', 'y2', 'y3'],
        'z': ['z0', 'z1']
    }
    lookup = ts_index_lib._Multi1D(names, name_to_values)

    i = 0
    for x in name_to_values['x']:
      for y in name_to_values['y']:
        for z in name_to_values['z']:
          self.assertEqual(lookup.get_index({'x': x, 'y': y, 'z': z}), i)
          i += 1


class TensorstoreIndexTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    coordinate_arrays = {
        'batch': ['batch0', 'batch1'],
        'plate': ['plate0', 'plate1', 'plate2'],
        'well_row': ['A', 'B', 'C', 'D'],
        'well_col': ['1', '2', '3', '4', '5'],
        'site_row': ['site_row0', 'site_row1'],
        'site_col': ['site_col0', 'site_col1'],
        'stain': ['stain0', 'stain1']
    }
    axes = ['X', 'Y', 'stain']
    axes_wrap = {
        'X': ['plate', 'well_col', 'site_col'],
        'Y': ['batch', 'well_row', 'site_row']
    }
    image_shape = (480, 640)
    unit_sizes = {'X': image_shape[1], 'Y': image_shape[0]}

    self.lookup = ts_index_lib.TensorstoreIndex(axes, axes_wrap,
                                                coordinate_arrays, unit_sizes)

  def test_get_dimensions(self):
    self.assertEqual(self.lookup.get_dimensions(), [19200, 7680, 2])

  def test_get_coords_individually(self):
    query = {
        'batch': 'batch0',
        'plate': 'plate0',
        'well_row': 'A',
        'well_col': '1',
        'site_row': 'site_row0',
        'site_col': 'site_col0',
        'stain': 'stain0',
    }
    self.assertEqual(self.lookup.get_coords(query), [0, 0, 0])
    # Move along the stain coordinate by one.
    query['stain'] = 'stain1'
    self.assertEqual(self.lookup.get_coords(query), [0, 0, 1])
    # Move along the Y coordinate by one image by going to the second site_row.
    query['site_row'] = 'site_row1'
    self.assertEqual(self.lookup.get_coords(query), [0, 480, 1])
    # Move along the X coordinate by one image by going to the second site_col.
    query['site_col'] = 'site_col1'
    self.assertEqual(self.lookup.get_coords(query), [640, 480, 1])

  def test_get_coords_raises_when_missing_key(self):
    valid_query = {
        'batch': 'batch0',
        'plate': 'plate0',
        'well_row': 'A',
        'well_col': '1',
        'site_row': 'site_row0',
        'site_col': 'site_col0',
        'stain': 'stain0',
    }
    # valid_query should not raise.
    _ = self.lookup.get_coords(valid_query)
    # Assert that it raises when the query is missing a key
    invalid_missing_keys = ['batch', 'plate', 'well_row', 'well_col', 'stain']
    for k in invalid_missing_keys:
      query = copy.deepcopy(valid_query)
      del query[k]
      with self.assertRaises(KeyError):
        self.lookup.get_coords(query)

  def test_get_coords_valid_when_missing_key(self):
    valid_query = {
        'batch': 'batch0',
        'plate': 'plate0',
        'well_row': 'B',
        'well_col': '2',
        'site_row': 'site_row1',
        'site_col': 'site_col1',
        'stain': 'stain0',
    }
    # valid_query should not raise.
    valid_coords = np.array([640*3, 480*3, 0])
    self.assertEqual(self.lookup.get_coords(valid_query), list(valid_coords))
    # Assert that it doesn't raise when missing smallest key(s)
    # drop site_row
    query = copy.deepcopy(valid_query)
    del query['site_row']
    expected_coords = valid_coords - np.array([0, 480, 0])
    self.assertEqual(self.lookup.get_coords(query), list(expected_coords))

    # drop [well_col, site_ col]
    query = copy.deepcopy(valid_query)
    del query['site_col']
    del query['well_col']
    expected_coords = valid_coords - np.array([640*3, 0, 0])
    self.assertEqual(self.lookup.get_coords(query), list(expected_coords))

  def test_get_block_sizes(self):
    valid_query = {
        'batch': 'batch0',
        'plate': 'plate0',
        'well_row': 'B',
        'well_col': '2',
        'site_row': 'site_row1',
        'site_col': 'site_col1',
        'stain': 'stain0',
    }
    # All dimensions are present, size should be 1 * unit_sizes
    self.assertEqual(self.lookup.get_block_sizes(valid_query), [640, 480, 1])

  def test_get_block_sizes_missing_site(self):
    valid_query = {
        'batch': 'batch0',
        'plate': 'plate0',
        'well_row': 'B',
        'well_col': '2',
        'stain': 'stain0',
    }
    # Missing sites, so sizes should be num_sites_per_well * unit_sizes
    self.assertEqual(
        self.lookup.get_block_sizes(valid_query), [2 * 640, 2 * 480, 1])

  def test_get_block_sizes_missing_site_well(self):
    valid_query = {
        'batch': 'batch0',
        'plate': 'plate0',
        'stain': 'stain0',
    }
    # Missing sites and wells, so sizes should be
    # num_wells_per_plate * num_sites_per_well * unit_sizes
    self.assertEqual(
        self.lookup.get_block_sizes(valid_query), [10 * 640, 8 * 480, 1])

  def test_get_coords_with_dataframe(self):
    query_df = pd.DataFrame.from_dict({
        'batch': ['batch0', 'batch0'],
        'plate': ['plate0', 'plate0'],
        'well_row': ['A', 'A'],
        'well_col': ['1', '1'],
        'site_row': ['site_row0', 'site_row1'],
        'site_col': ['site_col0', 'site_col1'],
        'stain': ['stain0', 'stain1'],
    })
    coords_dict = self.lookup.get_coords_dict(query_df)
    self.assertSameElements(coords_dict.keys(), ['X', 'Y', 'stain'])
    np.testing.assert_equal(coords_dict['X'], [0, 640])
    np.testing.assert_equal(coords_dict['Y'], [0, 480])
    np.testing.assert_equal(coords_dict['stain'], [0, 1])


if __name__ == '__main__':
  absltest.main()

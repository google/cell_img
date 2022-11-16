"""Tests for observations."""

from absl.testing import absltest
from cell_img.common import plate_utils
from cell_img.dose_response import observations
import numpy.testing
import pandas as pd


class ObservationsTest(absltest.TestCase):

  def test_row_int_to_str(self):
    for well in ['A1', 'B2', 'Z26', 'AA27', 'AB28']:
      row_str = plate_utils.get_row_str(well, strict=False)
      row_int = plate_utils.get_row_int(well)
      self.assertEqual(row_str, observations.row_int_to_str(row_int))

  def test_from_dataframe_is_invertible(self):
    df = pd.DataFrame(
        {'plate': ['alpha', 'alpha', 'beta', 'beta'],
         'well': ['A1', 'A2', 'B1', 'B2'],
         'actives': ['infected_control', 'active_control', 'sample', 'sample'],
         'compound': ['DMSO', 'abc', 'def', 'ghi'],
         'concentration': [1., 2., 3., 4.],
         'n_hypnozoite': [10, 5, 8, 6],
         'n_hepatocyte': [100, 50, 80, 60],})
    obs = observations.from_dataframe(
        df,
        well_row_min=1, well_row_max=2,
        well_col_min=1, well_col_max=2,
        hypnozoite_col='n_hypnozoite',
        hepatocyte_col='n_hepatocyte',
        compound_col='compound')
    new_df = observations.to_dataframe(
        obs,
        hypnozoite_col='n_hypnozoite',
        hepatocyte_col='n_hepatocyte',
        compound_col='compound')
    for col in ['plate', 'well', 'actives', 'compound',
                'concentration', 'n_hypnozoite', 'n_hepatocyte']:
      numpy.testing.assert_array_equal(df[col], new_df[col])

if __name__ == '__main__':
  absltest.main()

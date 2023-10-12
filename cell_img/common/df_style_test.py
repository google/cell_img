"""Tests for cell_img.common.df_style."""

from absl.testing import absltest
from cell_img.common import df_style

import numpy as np
import numpy.testing as npt
import pandas as pd


class DfStyleTest(absltest.TestCase):

  def testColorizerWithInts(self):
    values = [1, 2, 3, 4]
    c = df_style.make_colorizer_from_series(pd.Series(values))

    for v in values:
      self.assertTrue(c(v).startswith('background-color: #'))

  def testColorizerWithStrings(self):
    values = ['a', 'b', 'cc', 'd']
    c = df_style.make_colorizer_from_series(pd.Series(values))

    for v in values:
      self.assertTrue(c(v).startswith('background-color: #'))

  def testColorizerWithMixedTypes(self):
    values = [1, 2.0, 'cc', 'd']
    c = df_style.make_colorizer_from_series(pd.Series(values))

    for v in values:
      self.assertTrue(c(v).startswith('background-color: #'))

  def testColorizerWithNan(self):
    values = [np.nan]
    c = df_style.make_colorizer_from_series(pd.Series(values))

    self.assertEqual(c(float(np.nan)), 'background-color: #cccccc')

  def testColorizerWithWrongValue(self):
    values = [0, 1, 2, 3]
    c = df_style.make_colorizer_from_series(pd.Series(values))
    self.assertEqual(c(float(np.nan)), 'background-color: #cccccc')


class DfStyleBorderTest(absltest.TestCase):

  def setUp(self):

    super().setUp()

    index_sizes = [3, 5, 7]
    column_sizes = [2, 4, 8, 9]
    test_df = pd.DataFrame(
        index=pd.MultiIndex.from_product([range(c) for c in index_sizes]),
        columns=pd.MultiIndex.from_product(
            [range(i) for i in column_sizes]),
        data=0)
    test_df.index.names = [f'row{i}' for i in range(len(index_sizes))]
    test_df.columns.names = [f'col{c}' for c in range(len(column_sizes))]

    pipe = test_df.pipe(df_style.border_data)

    # Stack and extract border headings into named columns
    pipe_stack = pipe.stack(test_df.columns.names).str.extract(
        r'(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+);$')
    pipe_stack.columns = ['header', 'top', 'right', 'bottom', 'left']

    self.index_sizes = index_sizes
    self.column_sizes = column_sizes
    self.pipe_stack = pipe_stack

  def testBorderStyle(self):
    # Headers should all be border_style.
    npt.assert_array_equal(self.pipe_stack.header, df_style._BORDER_STYLE)

  def testFixedCells(self):
    # Borders may only be at the right and bottom of cells.  Top and left should
    # have no border

    npt.assert_array_equal(self.pipe_stack.top, df_style._NO_BORDER)
    npt.assert_array_equal(self.pipe_stack.left, df_style._NO_BORDER)

  def testRowColumnInvariance(self):
    # Right borders should not change across rows.
    npt.assert_array_equal(self.pipe_stack.xs(0, level='row0').right,
                           self.pipe_stack.xs(1, level='row0').right)

    # Bottom borders should not change across columns
    npt.assert_array_equal(self.pipe_stack.xs(0, level='col0').bottom,
                           self.pipe_stack.xs(1, level='col0').bottom)

  def testColumnBorders(self):
    first_row_borders = self.pipe_stack.xs((0, 0, 0),
                                           level=['row0', 'row1', 'row2']).right
    product_boundaries = np.array(list(reversed(self.column_sizes))).cumprod()

    # Test from msb (b=border) to lsb.  More significant borders dominate so
    # make sure to remove more significant borders when testing less significant
    # borders.
    #
    # e.g. if a ls_border (dashed) occurred every 2 cols and a ms_border (solid)
    # occurred every 4 cols, we'd expect the border-style to be
    #
    # none, dashed, none, solid, none, dashed, none, none
    #
    # (for style purposes we don't border the last column)
    #
    # So first test solid @ every_4, then test dashed @ every_2 except for where
    # we already tested for solid.

    already_tested = set()
    for border, boundaries in zip(df_style._HTML_BORDERS,
                                  product_boundaries[2::-1]):
      test_indices = set(
          np.arange(boundaries - 1, len(first_row_borders), boundaries)[:-1]
      ) - already_tested

      npt.assert_array_equal(np.where(first_row_borders == border)[0],
                             sorted(test_indices))
      already_tested |= test_indices

    # If it hasn't been tested it should have no border
    not_tested = set(range(len(first_row_borders))) - already_tested
    npt.assert_array_equal(
        np.where(first_row_borders == df_style._NO_BORDER)[0],
        sorted(not_tested))

  def testRowBorders(self):
    first_col_borders = self.pipe_stack.xs(
        (0, 0, 0, 0), level=['col0', 'col1', 'col2', 'col3']).bottom
    product_boundaries = np.array(list(reversed(self.index_sizes))).cumprod()

    # See comment in testColumnBorders.  We use a similar strategy here.
    already_tested = set()
    for border, boundaries in zip(df_style._HTML_BORDERS,
                                  product_boundaries[1::-1]):
      test_indices = set(
          np.arange(boundaries - 1, len(first_col_borders),
                    boundaries)[:-1]) - already_tested

      npt.assert_array_equal(np.where(first_col_borders == border)[0],
                             sorted(test_indices))
      already_tested |= test_indices

    # If it hasn't been tested it should have no border
    not_tested = set(range(len(first_col_borders))) - already_tested
    npt.assert_array_equal(
        np.where(first_col_borders == df_style._NO_BORDER)[0],
        sorted(not_tested))


if __name__ == '__main__':
  absltest.main()

"""Tests for batch_design."""

import random

from absl.testing import absltest
from cell_img.morphology_based_screening_batch_design import batch_design
import numpy as np
import pandas as pd


def make_example_cell_line_df():
  """Creates an example cell line dataframe.

  This example frame contains columns typical of cell line metadata and can be
  used to run the code, in the abscence of real data, for testing or for demo
  purposes.
  Returns:
    The example dataframe
  """
  cell_ids = [str(x) for x in list(np.arange(1, 61))]

  pair_ids = list(np.repeat([str(x) for x in list(np.arange(0, 30))], 2))

  disease_state = ['healthy', 'disease'] * 30

  ages = [random.randrange(50, 75, 1) for i in range(60)]

  doubling_time = [(random.randrange(200, 400, 1) / 100) for dt in range(60)]

  df = pd.DataFrame(
      list(zip(cell_ids, pair_ids, disease_state, ages, doubling_time)),
      columns=['cell_id', 'pair_id', 'disease_state', 'age', 'doubling_time'])

  return df


class BatchDesignTest(absltest.TestCase):

  def test_make_batch(self):
    cell_df = make_example_cell_line_df()
    batch_df, ns_source_plate = batch_design.make_batch(
        cell_df, qc_categories=['doubling_time', 'age'])
    self.assertEqual(batch_df.shape, (2304, 12))
    self.assertEqual(ns_source_plate.shape, (96, 10))

  def test_make_batch_raises(self):
    cell_df = make_example_cell_line_df()
    cell_df['age'].iloc[0] = 'invalid'
    with self.assertRaisesRegex(
        ValueError, 'At least one columns not suitable for optimized seeding'):
      batch_design.make_batch(cell_df, qc_categories=['doubling_time', 'age'])


if __name__ == '__main__':
  absltest.main()

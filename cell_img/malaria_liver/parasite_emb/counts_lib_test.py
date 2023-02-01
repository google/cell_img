"""Tests for counts_lib."""

from absl.testing import absltest
from cell_img.malaria_liver.parasite_emb import counts_lib


class CountsUtilsTest(absltest.TestCase):

  def test_yield_metrics_per_element(self):
    elem = {
        'batch': 'test_batch',
        'plate': 'test_plate',
        'well': 'test_well',
        'stage_result': 'test_stage'
    }
    results = list(counts_lib.yield_metrics_per_element(elem))
    expected_well_key = counts_lib.WellKey(
        batch='test_batch',
        plate='test_plate',
        well='test_well'
        )
    self.assertSequenceEqual(
        results,
        [(expected_well_key, 'num_obj', 1),
         (expected_well_key, 'num_test_stage', 1)])

  def test_missing_info(self):
    elem = {
        'batch': 'test_batch',
        'well': 'test_well',
        'stage_result': 'test_stage'
    }
    results = list(counts_lib.yield_metrics_per_element(elem))
    expected_well_key = counts_lib.WellKey(
        batch='test_batch',
        plate='missing_plate',
        well='test_well'
        )
    self.assertSequenceEqual(
        results,
        [(expected_well_key, 'num_obj', 1),
         (expected_well_key, 'num_test_stage', 1)])

if __name__ == '__main__':
  absltest.main()


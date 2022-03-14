"""Tests for cell_img.image_grid.downsample_lib."""

from absl.testing import absltest
from cell_img.image_grid import downsample_lib


class DownsampleLibTest(absltest.TestCase):

  def test_get_delta_factors(self):
    downsampling_factors = [[1, 1, 1], [2, 3, 1], [4, 12, 1]]
    res = downsample_lib.get_delta_factors(downsampling_factors)
    self.assertSameStructure(res, [[2, 3, 1], [2, 4, 1]])

  def test_get_delta_factors_raises_if_not_int(self):
    downsampling_factors = [[1], [2], [5]]
    with self.assertRaisesRegex(ValueError,
                                r'Only integers are allowed but got \[2.5\]'):
      downsample_lib.get_delta_factors(downsampling_factors)

  def test_split_downsample_level_from_path(self):
    path = '/my/path/root/s54'
    root_path, level = downsample_lib.split_downsample_level_from_path(path)
    self.assertEqual(root_path, '/my/path/root')
    self.assertEqual(level, 54)

  def test_split_downsample_level_from_path_raises(self):
    with self.assertRaisesRegex(ValueError,
                                'does not appear to be a downsample path'):
      downsample_lib.split_downsample_level_from_path('/invalid/path')

  def test_join_downsample_level_to_path(self):
    self.assertEqual(
        downsample_lib.join_downsample_level_to_path('/my/path/root', 54),
        '/my/path/root/s54')

  def test_join_downsample_level_to_path_raises(self):
    with self.assertRaisesRegex(ValueError,
                                'appears to already be a downsample path'):
      downsample_lib.join_downsample_level_to_path('/my/path/root/s0', 54)

  def test_split_join_downsample_level_and_path(self):
    orig_path = '/my/path/root/s54'
    root_path, level = downsample_lib.split_downsample_level_from_path(
        orig_path)
    rejoined = downsample_lib.join_downsample_level_to_path(root_path, level)
    self.assertEqual(orig_path, rejoined)


if __name__ == '__main__':
  absltest.main()

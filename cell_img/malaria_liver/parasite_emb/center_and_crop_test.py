"""Tests for center_and_crop_patch."""

from absl.testing import absltest
from cell_img.malaria_liver.parasite_emb import center_and_crop
import numpy as np


class CenterAndCropPatchTest(absltest.TestCase):

  def test_get_pixel_coords(self):
    coords = center_and_crop._get_pixel_coords(size=4)
    np.testing.assert_almost_equal(coords, [-1.5, -0.5, 0.5, 1.5])

  def test_get_radius(self):
    radius = center_and_crop.get_radius(2)
    expected = np.ones((2, 2)) * np.sqrt(0.5)
    np.testing.assert_almost_equal(radius, expected)

  def test_center_and_crop_finds_block(self):
    values = np.zeros((8, 8))
    values[5:7, 1:3] = 1.
    row, col = center_and_crop.get_corner_for_crop(
        values=values, crop_size=4, radial_power=1.)
    self.assertEqual((row, col), (4, 0))

  def test_center_and_crop_handles_edges(self):
    values = np.zeros((8, 8))
    values[0:2, 0:2] = 1.
    row, col = center_and_crop.get_corner_for_crop(
        values=values, crop_size=4, radial_power=1.)
    self.assertEqual((row, col), (0, 0))

    values = np.zeros((8, 8))
    values[6:, 6:] = 1.
    row, col = center_and_crop.get_corner_for_crop(
        values=values, crop_size=4, radial_power=1.)
    self.assertEqual((row, col), (4, 4))

  def test_get_max_min_angles(self):
    values = np.zeros((8, 8))
    for i in range(4):
      values[i, i] = -1
      values[3-i, 4+i] = 1

    expected_min = 145
    expected_max = 34
    for i in range(8):
      if i <= 4:
        test_values = values
      else:
        test_values = np.flipud(values)
      test_values = np.rot90(test_values, k=i)
      theta_min, theta_max = center_and_crop.get_min_max_angles(test_values)
      standardized = center_and_crop.to_canonical_rotation(
          test_values, theta_min, theta_max)
      tmin, tmax = center_and_crop.get_min_max_angles(standardized)
      self.assertEqual(expected_min, tmin)
      self.assertEqual(expected_max, tmax)

  def test_process_one_element_fails_on_bad_element(self):
    with self.assertRaises(ValueError):
      center_and_crop.process_one_element(
          element={}, stain_indices=[2], crop_size=20,
          do_rotate=True, do_center=True)

  def test_crop_does_not_move_center(self):
    # set up dictionary Element
    old_center_row = 100
    old_center_col = 500
    element = {}
    element['image'] = np.ones((12, 12, 2), dtype=np.uint8)
    element['center_row'] = old_center_row
    element['center_col'] = old_center_col

    # crop but do not center or rotate.
    updated_elem = center_and_crop.process_one_element(
        element, stain_indices=[0], crop_size=4, do_rotate=False,
        do_center=False)

    # get it back. No centering so center should stay the same.
    self.assertEqual(old_center_row, element['center_row'])
    self.assertEqual(old_center_col, element['center_col'])
    self.assertEqual(updated_elem['image'].shape, (4, 4, 2))

  def test_odd_crop_size(self):
    # set up dictionary Element
    old_center_row = 100
    old_center_col = 500
    element = {}
    element['image'] = np.ones((12, 12, 2), dtype=np.uint8)
    element['center_row'] = old_center_row
    element['center_col'] = old_center_col

    # crop but do not center or rotate.
    updated_elem = center_and_crop.process_one_element(
        element, stain_indices=[0], crop_size=3, do_rotate=False,
        do_center=False)

    # get it back. No centering so center should stay the same.
    self.assertEqual(old_center_row, element['center_row'])
    self.assertEqual(old_center_col, element['center_col'])
    self.assertEqual(updated_elem['image'].shape, (3, 3, 2))


if __name__ == '__main__':
  absltest.main()

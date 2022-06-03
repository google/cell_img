"""Tests for image_lib."""

from absl.testing import absltest
from cell_img.common import image_lib

import numpy as np


class ImageLibTest(absltest.TestCase):

  def test_normalize_image(self):
    # test the defaults -- will normalize to be between min and 0.75
    img = np.array([0, 4, 8])
    expected_norm_img = np.array([0, 0.375, 0.75])
    norm_img = image_lib.normalize_image(img)
    np.testing.assert_allclose(norm_img, expected_norm_img)
    img = np.array([1, 5, 9])
    expected_norm_img = np.array([0, 0.375, 0.75])
    norm_img = image_lib.normalize_image(img)
    np.testing.assert_allclose(norm_img, expected_norm_img)

    # test the lowest max (do not overcorrect a very dim image)
    img = np.array([0, 1, 1])
    expected_norm_img = np.array([0, 0.075, 0.075])
    norm_img = image_lib.normalize_image(img, lowest_max=10)
    np.testing.assert_allclose(norm_img, expected_norm_img)

    # test the norm_max
    img = np.array([0, 4, 8])
    expected_norm_img = np.array([0, 0.25, 0.5])
    norm_img = image_lib.normalize_image(img, norm_max=0.5)
    np.testing.assert_allclose(norm_img, expected_norm_img)
    img = np.array([1, 5, 9])
    expected_norm_img = np.array([0, 0.5, 1.0])
    norm_img = image_lib.normalize_image(img, norm_max=1.0)
    np.testing.assert_allclose(norm_img, expected_norm_img)

  def test_normalize_per_color_image(self):
    # create slices that we can normalize individually or together
    img = np.array([[[0, 0], [0, 0]], [[1, 2], [1, 2]]])

    # normalize slices together
    expected_norm_img = np.array([[[0, 0], [0, 0]],
                                  [[0.5, 1], [0.5, 1]]])
    norm_img = image_lib.normalize_image(img, norm_max=1.0)
    np.testing.assert_allclose(norm_img, expected_norm_img)

    # norm separately
    expected_norm_per_c = np.array([[[0, 0], [0, 0]], [[1, 1], [1, 1]]])
    img_norm_per_c = image_lib.normalize_per_color_image(img, norm_max=1.0)
    np.testing.assert_allclose(img_norm_per_c, expected_norm_per_c)


if __name__ == '__main__':
  absltest.main()

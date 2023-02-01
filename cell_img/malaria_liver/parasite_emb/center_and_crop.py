"""Methods for centering and cropping patches."""

from typing import Any, Dict, List, Tuple

from cell_img.malaria_liver.parasite_emb import config
import numpy as np
import scipy.ndimage
import scipy.signal
import skimage.transform


def _get_pixel_coords(size: int) -> np.ndarray:
  """Get coordinates for a row of pixels with the origin in the center.

  Coordinates should be ..., -3/2, -1/2, 1/2, 3/2, ... (even)
  or:                   ..., -2, -1, 0, 1, 2, ... (odd)

  Args:
    size: image size in pixels.

  Returns:
    Coordinates for a single row or column.
  """
  if size % 2 == 0:
    coords = np.linspace(1, size // 2, size // 2) - 0.5
    return np.concatenate([-coords[::-1], coords], axis=-1)
  else:
    cut = size // 2
    return np.arange(-cut, cut + 1)


def get_radius(size: int) -> np.ndarray:
  """Get the radial distance of points in a square from the center.

  Args:
    size: image size in pixels.

  Returns:
    Matrix of radii**2 with shape (size, size)
  """
  coords_sq = _get_pixel_coords(size) ** 2.
  return np.sqrt(coords_sq[:, None] + coords_sq[None, :])


def get_corner_for_crop(
    values: np.ndarray,
    crop_size: int,
    radial_power=0.5) -> Tuple[int, int]:
  """Find the coordinates at which to crop an image.

  The idea is that we want to crop an image centered around the bulk of some
  stain or stains. Here value is a 2-d array that summarizes the stains we want
  to use for centering. With malaria images, for example, we want to maximize
  the non-DAPI stains, so we might use something like
  values = np.sum(image[:, :, non_dapi_stains]**2, axis=-1)
  We convolve values against a square mask of size (crop_size, crop_size)
  that has a maximum value at the center and falls off radially as
  1/radius^radial_power. We crop so that the convolution is maximized. For
  malaria data, using radial_power=0.5 works reasonably well.

  The cropped image will always be fully contained within the source image.

  Args:
    values: 2-D array of values to use for centering
    crop_size: size of the cropped image in pixels
    radial_power: we generate weights to convolve with values as
      1/radius^radial_power

  Returns:
    Coordinates for the corner of the cropped image, row and col. The values
    are chosen to maximize the sum
    np.sum(image[row:row+crop_size, col:col+crop_size] * weights)
    where weights are given by 1/get_radius(crop_size)^radial_power

  Raises:
    ValueError: if radial_power <= 0
  """
  if crop_size > np.min(values.shape[:2]):
    raise ValueError('crop_size must be <= image dimensions, %d >= min(%s)' % (
        crop_size, str(values.shape[:2])))
  if radial_power <= 0.:
    raise ValueError('radial_power must be positive, got %f' % radial_power)
  weights = get_radius(crop_size) ** (-radial_power)
  conv = scipy.signal.convolve2d(values, weights, mode='valid')
  max_by_row = np.max(conv, axis=-1)
  max_row = np.argmax(max_by_row)
  max_col = np.argmax(conv[max_row, :], axis=-1)
  return max_row, max_col


def get_min_max_angles(values: np.ndarray) -> Tuple[float, float]:
  """Find the angles at which values has axial extremes.

  Find the angles for which the average of values over a 90 degree cone
  centered at the origin are minimized and maximized. Note: ignores values
  more than size/2 units from the center, i.e. we look only at the maximal
  circle inscribed within the square.

  Args:
    values: square array of values

  Returns:
    The angles (in degrees) at which the average of values a 90 degree cone
    from the center of the square and centered at theta achieves its minimum
    and maximum.
  """
  if len(values.shape) != 2 or values.shape[0] != values.shape[1]:
    raise ValueError('values should be 2D square, got %s' % str(values.shape))
  size = values.shape[0]
  radius = get_radius(size)
  values_polar = skimage.transform.warp_polar(values)
  values_polar = values_polar[:, :size // 2]
  flat = np.mean(values_polar * radius[size // 2, size // 2:], axis=-1)
  s = scipy.ndimage.filters.convolve1d(
      flat, weights=np.ones((90,))/90., mode='wrap')
  return np.argmin(s), np.argmax(s)


def to_canonical_rotation(
    values: np.ndarray,
    theta_min: float,
    theta_max: float,
) -> np.ndarray:
  """Rotate an image so max angle is in 1st quadrant & min is in 1st or 2nd."""
  # rotate to put theta_max in the first quadrant
  rotated = np.rot90(values, k=theta_max // 90)

  delta = 90 * (theta_max // 90)
  # update theta_min and theta_max
  theta_max -= delta
  theta_min -= delta
  # standardize theta_min so it lies in [0, 360)
  theta_min -= 360 * (theta_min // 360)

  if theta_min > 180:
    rotated = np.rot90(np.flipud(rotated), k=-1)
  return rotated


def _get_stain_sum(image: np.ndarray, stain_indices: np.ndarray) -> np.ndarray:
  return np.sum(image[:, :, stain_indices]**2., axis=-1)


def _corner_for_just_crop(values, crop_size: int):
  """Returns the upper left corner values to crop without centering."""
  return (values.shape[0] - crop_size) // 2, (values.shape[1] - crop_size) // 2


def crop_and_rotate_image(
    image: np.ndarray,
    stain_indices: np.ndarray,
    crop_size: int,
    rotate: bool,
    center: bool
    ) -> Tuple[np.ndarray, int, int]:
  """Center and crop images based on specified stains.

  Args:
    image: A numpy array with the image values. These images are assumed to be
      a stack of 2D images (e.g. multiple microscope images of the same position
      in different stains). The first dimensions are the 2D images and the
      different stains are in the third dimension.
    stain_indices: A numpy array with the indices (within the image array) for
      the stains to use in the centering and rotating algorithms. For example,
      cells are frequently centered using only the DAPI nuclear stain.
    crop_size: Integer size of the number of pixels per side for the square
      of the cropped image.
    rotate: Boolean indicating whether the patches should be rotated to a
      canonical rotation.
    center: Boolean indicating whether the patches should be centered before
      cropping. False indicates that the current center will continue to be
      the center in the cropped image.

  Returns:
    The centered, cropped, rotated image and the row, col value for the number
      of pixels the center moved. This row and column indicate the corner of
      the new patch within the original patch image (i.e they are guaranteed to
      always be positive).
  """

  if crop_size > np.min(image.shape[:2]):
    raise ValueError('crop_size must be <= image dimensions, %d >= min(%s)' % (
        crop_size, str(image.shape[:2])))

  stain_sum = _get_stain_sum(image, stain_indices)

  if center:
    row, col = get_corner_for_crop(
        stain_sum, crop_size=crop_size)
  else:
    row, col = _corner_for_just_crop(image, crop_size)
  cropped_image = image[row:row + crop_size, col:col + crop_size, :]

  if rotate:
    stain_sum = _get_stain_sum(image, stain_indices)
    theta_min, theta_max = get_min_max_angles(stain_sum ** 2.)
    return to_canonical_rotation(
        cropped_image, theta_min=theta_min, theta_max=theta_max), row, col
  else:
    return cropped_image, row, col


def process_one_element(
    element: Dict[Any, Any], stain_indices: List[int], crop_size: int,
    do_rotate: bool, do_center: bool) -> Dict[Any, Any]:
  """Wrapper for beam crops/centers/rotates and returns the adjusted patch.

  This function can be used in a beam pipeline where the elements are
  dictionaries of the expected format. It also provides a usage example.

  Args:
    element: Dictionary with the data. Expected to have keys: 'image' should
      have the patch image as a np array with 3 dimensions, where the first
      two dimensions are the X/Y of the image and the third dimension is the
      stain. 'center_row' and 'center_col' indicate the original position
      of the patch within the bigger site image. Center row is along the
      first dimension in the input image array and center col is along the
      second.
    stain_indices: Integer list indicating which stains in the patch stack
      should be used in the centering and rotating algorithms. For example,
      cell centering frequently uses only the DAPI nuclear stain, in which
      as the stain indices list would be [2] if the DAPI stain was the
      third stain in the stack.
    crop_size: Integer number of pixels per side for the cropped patch. Must
      be equal to or smaller than the original patch size.
    do_rotate: Bool indicating whether the patch should be rotated to a
      canonical rotation.
    do_center: Bool indicating whether the patch should be centered before
      cropping.

  Returns:
    The element with the original 'image', 'center_row', and 'center_col'
    values changed. Adds new values 'crop_row' and 'crop_col' which are the
    row and column of the upper left corner of the new patch within the original
    patch.

  Raises:
    ValueError if the element does not have the required keys.
  """
  required_keys = [config.IMAGE, config.CENTER_ROW,
                   config.CENTER_COL]
  missing_keys = []
  for k in required_keys:
    if k not in element: missing_keys.append(k)
  if missing_keys:
    raise ValueError('Element is missing required keys: %s' % missing_keys)

  # get the original image from the element and process it
  image = element[config.IMAGE]
  new_img, row_adjust, col_adjust = crop_and_rotate_image(
      image, np.array(stain_indices), crop_size, do_rotate, do_center)

  # the center value will be adjusted based on the new center and the old and
  # new patch sizes.
  center_row_adjust = row_adjust - image.shape[0] // 2 + (crop_size + 1) // 2
  center_col_adjust = col_adjust - image.shape[1] // 2 + (crop_size + 1) // 2

  # save the results back into the element
  element[config.IMAGE] = new_img
  element[config.CROP_ROW] = row_adjust
  element[config.CROP_COL] = col_adjust
  element[config.CENTER_ROW] = (
      element[config.CENTER_ROW] + center_row_adjust)
  element[config.CENTER_COL] = (
      element[config.CENTER_COL] + center_col_adjust)

  return element





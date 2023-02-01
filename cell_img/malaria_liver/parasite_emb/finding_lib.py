"""Lightweight version of cell_center_patch_picker.
"""
import collections
import copy
import logging
from typing import Any, Dict, List, Tuple, Union, Generator

import apache_beam as beam
from cell_img.malaria_liver.parasite_emb import config
import numpy as np
import tensorflow as tf


_METRICS_NAMESPACE = 'Finding'

# Constants for the tf.image.non_max_suppression that prunes overlapping boxes.
OVERLAP_MAX_OUTPUT_SIZE = 100
OVERLAP_DEFAULT_MIN_SCORE = 0.01
# IOU threshold was originally 0.05 to keep more options, 0.5 is a more
# typical value.
OVERLAP_IOU_THRESHOLD = 0.5

# NOTE: critique will complain about collections and tell you to use attrs,
# but that causes problems with pickle and the beam pipeline fails
ModelInfo = collections.namedtuple('ModelInfo',
                                   ['path', 'use_pipeline_norm', 'notes'])
Point = collections.namedtuple('Point',
                               ['row', 'column', 'confidence', 'overlap'])
Patch = collections.namedtuple(
    'Patch', ['upper_left', 'lower_right', 'confidence', 'overlap'])


class BatchCellCenterFinder(beam.DoFn):
  """Identifies cell centers in a batch site images."""

  def __init__(self, model_path: str):
    super().__init__()
    self.model_path = model_path
    self._model = None
    self._infer_fn = None

  def setup(self):
    self._model = tf.saved_model.load(self.model_path)

    @tf.function
    def _infer_fn(batch_input):
      return self._model.signatures['serving_default'](batch_input)
    self._infer_fn = _infer_fn

    @tf.function
    def _convert_fn(img):
      # Convert to 255 range and add batch dimension.
      img = 255.0 * img
      rev_img = img[:, :, ::-1]
      rev_img = tf.expand_dims(rev_img, axis=0)
      return tf.cast(rev_img, tf.uint8)
    self._convert_fn = _convert_fn

  def process_batched_elements(self, elements: List[Dict[str, Any]]):
    # Make a copy to avoid mutating input to beam transform.
    example_dicts = copy.deepcopy(elements)
    # Image dims
    image_width = example_dicts[0][config.IMAGE].shape[0]
    image_height = example_dicts[0][config.IMAGE].shape[1]
    img_list = []
    for e in example_dicts:
      img_list.append(self._convert_fn(e[config.IMAGE]))
    imgs = tf.concat(img_list, axis=0)
    result_dict = self._infer_fn(imgs)
    for i, _ in enumerate(example_dicts):
      example_cell_centers = extract_best_centers(result_dict,
                                                  image_width,
                                                  image_height,
                                                  i)
      example_cell_centers_list = []
      for cell_center in example_cell_centers:
        row = cell_center[0]
        col = cell_center[1]
        if len(cell_center) == 2:
          # Set confidence to 1.0 if not provided.
          confidence = 1.0
          overlap = False
        elif len(cell_center) == 4:
          confidence = cell_center[2]
          overlap = cell_center[3]
        else:
          raise ValueError('Cell centers are malformed. len %d ' %
                           len(cell_center))
        example_cell_centers_list.append(Point(row, col, confidence, overlap))
      beam.metrics.Metrics.counter(_METRICS_NAMESPACE, 'num_cell_centers').inc(
          len(example_cell_centers_list))
      yield example_dicts[i], example_cell_centers_list

  def process(self, batched_elements: List[Dict[str, Any]]):
    """Processes one batched element."""
    for output in self.process_batched_elements(batched_elements):
      yield output


def extract_best_centers(results_dict: Dict[str, tf.Tensor],
                         width: int,
                         height: int,
                         example_index: int,
                         min_thresh: float = OVERLAP_DEFAULT_MIN_SCORE
                         ) -> List[Point]:
  """Remove cell centers that pass a score threshold."""
  points = []
  example_boxes = results_dict['detection_boxes'][example_index]
  example_scores = results_dict['detection_scores'][example_index]
  valid_indices = tf.image.non_max_suppression(
      example_boxes,
      example_scores,
      max_output_size=OVERLAP_MAX_OUTPUT_SIZE,
      score_threshold=min_thresh,
      iou_threshold=OVERLAP_IOU_THRESHOLD)
  valid_indices = set(valid_indices.numpy())
  for i, (box, score) in enumerate(zip(example_boxes, example_scores)):
    if score < min_thresh:
      continue
    x, y = (box[0] + box[2]) / 2.0 * width, (box[1] + box[3]) / 2.0 * height
    if i in valid_indices:
      overlap = False
    else:
      overlap = True
    points.append(Point(int(x), int(y), score, overlap))
  return points


class MakePatches(beam.PTransform):
  """Picks patches at cell centers."""

  def __init__(self, min_confidence: float, pixels_per_side: int):
    """Constructor."""
    super().__init__()
    self.min_confidence = min_confidence
    self.pixels_per_side = pixels_per_side

  def expand(
      self, p_elements_and_centers: beam.pvalue.PCollection
  ) -> Tuple[beam.pvalue.PCollection, beam.pvalue.PCollection]:
    """Picks patches at cell centers.

    Args:
      p_elements_and_centers: A PCollection of 2-tuples where the first element
        is the whole image element with metadata from the input, and the second
        element is the corresponding list of cell center points.

    Returns:
      p_elements_and_patches: A PCollection of 2-tuples where the first element
        is an unchanged whole image element from the input, and the second
        element is the corresponding list of patches.
    """

    p_elements_and_patches = p_elements_and_centers | beam.Map(
        self._make_patches)
    return p_elements_and_patches

  def _filter_low_confidence(self,
                             cell_centers_list: List[Point]) -> List[Point]:
    """Drops the low confidence cell centers."""
    total_centers = len(cell_centers_list)
    cell_centers_list = [
        x for x in cell_centers_list if x.confidence > self.min_confidence
    ]
    logging.info('Dropped from %d to %d cell centers', total_centers,
                 len(cell_centers_list))
    beam.metrics.Metrics.counter(
        _METRICS_NAMESPACE,
        'num_low_confidence').inc(total_centers - len(cell_centers_list))
    return cell_centers_list

  @beam.typehints.no_annotations
  def _make_patches(
      self, image_elements_and_centers: Tuple[Dict[Any, Any], List[Point]]
  ) -> Tuple[Dict[Any, Any], List[Patch]]:
    """Converts the cell centers to patches for an image group bundle.

    Cell centers too close to the image edge are dropped.

    Args:
      image_elements_and_centers: A PCollection of 2-tuples where the first
        element is the whole image element with metadata from the input, and the
        second element is the corresponding list of cell center points.

    Returns:
      A 2-tuple of where the first element is the input image group
      bundle, and the second element is the corresponding list of
      Patch objects indicating patch coordinates. The cell and
      patch counts per site are set in the image group other_features.
    """
    # Make a copy to avoid mutating input to beam transform.
    image_elements_and_centers = copy.deepcopy(
        image_elements_and_centers)
    image_element, cell_centers_list = image_elements_and_centers

    cell_centers_list = self._filter_low_confidence(cell_centers_list)

    patches = []
    for cell_center in cell_centers_list:
      upper_left, lower_right = drop_edges(
          image_element[config.IMAGE].shape[:2], cell_center,
          self.pixels_per_side)
      if upper_left is not None:
        patches.append(
            Patch(
                upper_left=upper_left,
                lower_right=lower_right,
                confidence=cell_center.confidence,
                overlap=cell_center.overlap))
    num_cells_detected_in_site = len(cell_centers_list)
    num_patches_in_site = len(patches)
    image_element['num_patches_in_site'] = num_patches_in_site
    image_element['num_cells_detected_in_site'] = num_cells_detected_in_site

    beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                 'num_patches').inc(num_patches_in_site)

    return image_element, patches


def drop_edges(image_shape: List[int], cell_center: Point,
               side_length: int) -> Tuple[Any, Any]:
  """Returns a pair of Point objects representing the image.

  This function attempts to return Points that represent a square subset of the
  input image centered at the input cell center. In cases where the cell center
  is near the boundary of the input image, (None, None) is returned to indicate
  that the cell cannot be properly extracted.

  Args:
    image_shape: The (num_rows, num_cols) shape of the image from which to
        extract a square subset.
    cell_center: The Point object of the center of the region to
        extract.
    side_length: The length in pixels of the desired output image.

  Returns:
    A pair of (upper_left, lower_right) Points to extract, or (None, None) if
    the requested extraction is not possible.

  Raises:
    ValueError: The requested side length is not a positive integer.
  """
  if side_length < 1:
    raise ValueError('Side length must be a positive integer: %s' % side_length)

  num_rows, num_cols = image_shape
  if min(num_rows, num_cols) < side_length:
    raise ValueError('Image of shape %s cannot create images with length %d' %
                     (image_shape, side_length))
  fraction_kept = (num_rows - side_length) * (num_cols - side_length) / (
      num_rows * num_cols)
  if fraction_kept < 0.7:
    logging.warning(
        'Images of size %sx%s only capture %s of the possible cell centers',
        side_length, side_length, fraction_kept)

  (min_row, max_row, min_col, max_col) = _patch_coords_without_boundary_check(
      cell_center.row, cell_center.column, side_length)
  if min_row < 0 or max_row > num_rows or min_col < 0 or max_col > num_cols:
    return (None, None)

  return (Point(row=min_row, column=min_col, confidence=1, overlap=False),
          Point(row=max_row, column=max_col, confidence=1, overlap=False))


def _patch_coords_without_boundary_check(center_row, center_col, side_length):
  desired_scale, pad = divmod(side_length, 2)
  min_row = center_row - desired_scale
  max_row = center_row + desired_scale + pad
  min_col = center_col - desired_scale
  max_col = center_col + desired_scale + pad
  return min_row, max_row, min_col, max_col


class PatchExtractor(beam.PTransform):
  """Extracts each patch images."""

  def expand(self, p_bundles_and_patches: beam.pvalue.PCollection
            ) -> beam.pvalue.PCollection:
    """Extracts each patch images.

    Args:
      p_bundles_and_patches: A PCollection of 2-tuples where the first element
        is a whole image element with metadata, and the second element is the
        corresponding list of patches to be extracted.

    Returns:
      A PCollection of dictionaries with patch image and metadata.
    """
    return p_bundles_and_patches | 'extract_patches' >> beam.ParDo(
        self._extract_patches)

  def _extract_patches(self, image_elements_and_patches: Tuple[
      Dict[Any, Any], List[Patch]]
                      ) -> Generator[Dict[Any, Any], None, None]:
    """Extracts the given patches from a whole image group bundle.

    Args:
      image_elements_and_patches: A 2-tuple where the first element
        is a whole image element with metadata, and the second element is the
        corresponding list of patches to be extracted.

    Yields:
      A patch image and metadata dictionary for each patch in the input with the
      image data extracted from the input whole image. The image metadata are
      updated to correspond to the patch.
    """
    # Make a copy to avoid mutating input to beam transform.
    image_elements_and_patches = copy.deepcopy(
        image_elements_and_patches)
    image_element, patches = image_elements_and_patches
    beam.metrics.Metrics.counter(_METRICS_NAMESPACE, 'num_whole_img').inc()
    beam.metrics.Metrics.counter(_METRICS_NAMESPACE, 'num_input_patches').inc(
        len(patches))

    sorted_patches = sorted(patches, key=_sort_patch_key_fn)

    for patch in sorted_patches:
      patch_element = _extract_one_patch(
          patch, image_element)
      if patch_element:
        beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                     'num_output_patches').inc()
        yield patch_element


def _extract_one_patch(
    patch: Patch,
    image_element: Dict[Any, Any]) -> Union[Dict[Any, Any], None]:
  """Returns a patch dict or None if the patch is invalid.

  Args:
    patch: The original Patch object
    image_element: The whole image and metadata dictionary to extract the patch
      from.

  Returns:
    A new patch dictionary extracted from the whole image dictionary.
  """
  patch_element = copy.deepcopy(image_element)
  # Remove features which correspond to the whole image but not the patch.
  element_keys = list(patch_element.keys())
  for key in element_keys:
    if key.startswith('full_image'):
      del patch_element[key]

  whole_image = image_element[config.IMAGE]
  patch_image_data = sub_image(whole_image, patch.upper_left, patch.lower_right)
  # Add patch data
  patch_element[config.IMAGE] = patch_image_data
  patch_element[config.CENTER_ROW] = (
      patch.upper_left.row + patch.lower_right.row) // 2
  patch_element[
      config.CENTER_COL] = (
          patch.upper_left.column + patch.lower_right.column) // 2
  patch_element[config.FINDING_CONFIDENCE] = float(patch.confidence)
  patch_element[config.FINDING_OVERLAP] = patch.overlap

  beam.metrics.Metrics.counter(_METRICS_NAMESPACE, 'num_patch_groups').inc()
  return patch_element


def _sort_patch_key_fn(patch):
  return (patch.upper_left.row, patch.upper_left.column, patch.lower_right.row,
          patch.lower_right.column)


def sub_image(image: np.ndarray, upper_left: Point,
              lower_right: Point) -> np.ndarray:
  """Returns a sub-image containing the data within the specified coordinates.

  Args:
    image: The whole image from which to extract a sub-region.
    upper_left: An inclusive Point representing the upper
        left pixel.
    lower_right: An exclusive Point representing the lower
        right pixel.

  Returns:
    A numpy array representing the sub-image.

  Raises:
    ValueError: The specified region is invalid.
  """
  if not is_valid_sub_image(image.shape, upper_left.row, upper_left.column,
                            lower_right.row, lower_right.column):
    raise ValueError('Invalid region for shape %s: %s %s' %
                     (image.shape, upper_left, lower_right))
  return image[upper_left.row:lower_right.row,
               upper_left.column:lower_right.column, :]


def is_valid_sub_image(image_shape: Tuple[int, int, int], upper_left_row: int,
                       upper_left_col: int, lower_right_row: int,
                       lower_right_col: int) -> bool:
  num_rows, num_cols, _ = image_shape
  return (0 <= upper_left_row < lower_right_row <= num_rows and
          0 <= upper_left_col < lower_right_col <= num_cols)

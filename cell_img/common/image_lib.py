"""Library functions to process img stacks into typical visualizations used.

The functions in this module turn a raw stack of all the channel images and
turn it into a matplotlib figure that the users are used to visualizing. This
figure generally has multiple combinations of the different stains to highlight
different attributes of the biological objects.
"""

import collections
import enum
import math
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_CONTACT_SHEET_NCOLS = 4


# A generic naming scheme for stains.
class Stain(enum.Enum):
  STAIN_1 = 1
  STAIN_2 = 2
  STAIN_3 = 3


# This list defines the overall set of thumbnails:
# Stain1 (B&W)       Stain2 (B&W)        Stain3 (B&W)     Composite(RGB)
# Where Composite = Stain1(blue) + Stain2(red) + Stain3(green)
EXAMPLE_FIGURE = [
    # ROW 1
    [  # Each tuple gives name and then stain list in that position
        ('Stain1', [Stain.STAIN_1]),
        ('Stain2', [Stain.STAIN_2]),
        ('Stain3', [Stain.STAIN_3]),
        (
            'composite',
            [  # Notice this is in RGB order
                Stain.STAIN_2, Stain.STAIN_3, Stain.STAIN_1
            ]),
    ]
]

# This dictionary matches the stain enum with the layer of the stack in
# the input image. The input images are an RGB stack in this order.
EXAMPLE_STAIN_TO_INDEX_DICT = {
    Stain.STAIN_2: 0,
    Stain.STAIN_3: 1,
    Stain.STAIN_1: 2,
}


def create_multi_figure(stain_stack_img: np.ndarray, figure_config: List[Any],
                        stain_to_index_map: Dict[Any, int]) -> plt.Figure:
  """Given an array of stain patch images, create a matplotlib figure.

  See EXAMPLE_FIGURE and EXAMPLE_STAIN_TO_INDEX_DICT as an example that takes
  an input image with 3 stains and shows each stain individually and then the
  composite image. This is nice because each stain individually can be shown
  in grey and normalized to the full dynamic range for just that stain, whereas
  the composite is stacked and then normalized so relative brightness can be
  visualized. So, for example, if STAIN_1 is DAPI, and your DAPI is very bright,
  you'd be able to see the details of the DAPI in the first grey image, but
  when stacked the blue would be very light relative to the other stains.

  Args:
    stain_stack_img: Numpy array representing the images a stack of stain imgs.
    figure_config: List representing the images to show in the figure.
    stain_to_index_map: Dictionary mapping the stain value used in the config to
      an index in the stain_stack_img.

  Returns:
    The matplotlib figure with each image as a subplot.
  """

  row_length = 1
  for fig_row in figure_config:
    row_length = max(row_length, len(fig_row))

  fig, ax_s = plt.subplots(
      len(figure_config), row_length, figsize=(12 * len(figure_config), 12))
  for row_index in range(len(figure_config)):
    if len(figure_config) == 1:
      cur_row_ax = collections.deque(ax_s)
    else:
      cur_row_ax = collections.deque(ax_s[row_index])
    for (fig_title, fig_stains) in figure_config[row_index]:
      ax = cur_row_ax.popleft()
      if len(fig_stains) == 1:
        stain = fig_stains[0]
        stain_index = stain_to_index_map[stain]
        img = stain_stack_img[:, :, stain_index]
      else:
        all_images = []
        img_shape = stain_stack_img[:, :, 0].shape
        zero_arr = np.zeros(img_shape)
        for stain in fig_stains:
          if stain is not None:
            stain_index = stain_to_index_map[stain]
            all_images.append(stain_stack_img[:, :, stain_index])
          else:
            all_images.append(zero_arr)
        # Normalize after compositing together to retain relative brightness.
        img = normalize_image(np.dstack(all_images))
      ax.imshow(img, cmap='gray')
      ax.grid('off')
      ax.set_xticks([])
      ax.set_yticks([])
      ax.set_title(fig_title, fontdict={'fontweight': 'bold'})
  fig.tight_layout(pad=0.8)
  return fig


def create_contact_sheet(image_list: List[np.ndarray],
                         ncols: Optional[int] = None,
                         nrows: Optional[int] = None) -> plt.Figure:
  """Saves out a contact sheet of RGB images in a plt figure.

  The images in this input list should already be normalized to use the
  desired dynamic range (see normalize_image below as one example.)

  Args:
    image_list: List of image arrays, should already be normalized. Each array
      is expected to have one dimension for each stain.
    ncols: Integer number of coluns in the contact sheet.
    nrows: Integer number of rows in the contact sheet.

  Returns:
    A matplotlib figure with thumbnails.
  """
  if not ncols:
    ncols = DEFAULT_CONTACT_SHEET_NCOLS
    nrows = math.ceil(len(image_list) / ncols)

  if len(image_list) > (nrows * ncols):
    raise ValueError('There were %d cols and %d rows but we have %d images.' %
                     (ncols, nrows, len(image_list)))

  fig, axarr = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows))

  for i, img in enumerate(image_list):
    col = i % ncols
    row = i // ncols
    if nrows == 1 and ncols == 1:
      ax = axarr
    elif nrows == 1:
      ax = axarr[col]
    elif ncols == 1:
      ax = axarr[row]
    else:
      ax = axarr[row, col]
    ax.imshow(img)

  for col in range(ncols):
    for row in range(nrows):
      if nrows == 1:
        ax = axarr[col]
      else:
        ax = axarr[row, col]
      ax.grid('off')
      ax.set_xticks([])
      ax.set_yticks([])

  fig.tight_layout(pad=0)
  return fig


def normalize_image(img, lowest_max=0, norm_max=0.75):
  """Rescale min and max image values to 0.0 and norm_max, respectively.

  Args:
    img: Numpy array of the image pixel intensities
    lowest_max: If the brightest pixel is below this value, use this value
      instead. This prevents too much noise in very dim images.
    norm_max: The value for the max. This would typically be 1.0 to use the full
      dynamic range, but some tools provide a way to brighten images further,
      and they look blown out if the max goes all the way up already.

  Returns:
    The normalized image.
  """
  min_value = img.min()
  max_value = max(lowest_max, img.max())
  return ((img - min_value) / (max_value - min_value)) * norm_max

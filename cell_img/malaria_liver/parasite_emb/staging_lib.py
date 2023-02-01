"""Run stage classification on a batch of patch images.
"""
import copy

import apache_beam as beam
from cell_img.malaria_liver.parasite_emb import config
import tensorflow as tf


class BatchDoFn(beam.DoFn):
  """Do fn to run stage classification on a batch of patch images."""

  def __init__(self, model_path, input_patch_size, input_num_channels,
               channel_order):
    self._staging_model = None
    self._infer_fn = None
    self.model_path = model_path
    self.channel_order = channel_order

    if len(input_patch_size) != 2:
      raise ValueError(f'Expected 2d patch size, received {input_patch_size}')
    self.input_patch_height = input_patch_size[0]
    self.input_patch_width = input_patch_size[1]
    self.input_num_channels = input_num_channels

  def setup(self):
    # Load the model and convert patches to the required input size/shape
    non_tf_hub_model = tf.keras.models.load_model(self.model_path,
                                                  compile=False)

    self._staging_model = tf.keras.Sequential([
        tf.keras.Input(
            shape=(self.input_patch_height, self.input_patch_width, 3)),
        non_tf_hub_model,
        tf.keras.layers.Softmax(),
    ])

    # To speed up inference, use non-eager execution
    @tf.function
    def _infer_fn(inputs):
      return self._staging_model(inputs)

    self._infer_fn = _infer_fn

    @tf.function
    def _convert_fn(img):
      img = tf.cast(img, tf.float32)
      # Reorder channels.
      img = tf.gather(img, self.channel_order, axis=2)
      return tf.expand_dims(img, axis=0)

    self._convert_fn = _convert_fn

  def process_batched_elements(self, batched_elements):
    # Make a copy to avoid mutating input to beam transform
    elements = copy.deepcopy(batched_elements)

    img_list = []
    for element in elements:
      img_list.append(self._convert_fn(element[config.IMAGE]))
    imgs = tf.concat(img_list, axis=0)
    softmax_scores = self._infer_fn(imgs)

    # Add stage names and class
    for element, softmax_score in zip(elements, softmax_scores):
      element[config.STAGE_INFER] = softmax_score.numpy()
      element[config.STAGE_NAMES
              ] = config.STAGING_CLASSES
      yield element

  def process(self, batched_elements):
    """Processes one batched element."""
    for output in self.process_batched_elements(batched_elements):
      yield output

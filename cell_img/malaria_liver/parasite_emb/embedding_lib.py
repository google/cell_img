"""Compute embeddings for a batch of examples."""
import copy

import apache_beam as beam
from cell_img.malaria_liver.parasite_emb import config
import tensorflow as tf
import tensorflow_hub as hub

# Constants specific to this model's training parameters.
_MODEL_IMAGE_DIM = 384


def _create_model(input_image_height,
                  input_image_width,
                  model_path,
                  model_output_size,
                  random_matrix_seed=2342343):
  return tf.keras.Sequential([
      tf.keras.Input(shape=(input_image_height, input_image_width, 3)),
      tf.keras.layers.Resizing(_MODEL_IMAGE_DIM, _MODEL_IMAGE_DIM),
      hub.KerasLayer(model_path, trainable=False),
      tf.keras.layers.Dense(
          model_output_size,
          kernel_initializer=tf.keras.initializers.GlorotNormal(
              seed=random_matrix_seed)),
  ])


class BatchDoFn(beam.DoFn):
  """Do fn to process a batch of inputs."""

  def __init__(self, model_path, input_image_size,
               input_num_channels, emb_dim_size, emb_seed):
    self._emb_model = None
    self._infer_fn = None
    self.model_path = model_path
    self.input_image_height = input_image_size[0]
    self.input_image_width = input_image_size[1]
    self.input_num_channels = input_num_channels
    self.emb_dim_size = emb_dim_size
    self.final_emb_dim_size = self.input_num_channels * emb_dim_size
    self.emb_seed = emb_seed

  def setup(self):
    # Load model and ensure that input images are resized to the required
    # input.
    self._emb_model = _create_model(self.input_image_height,
                                    self.input_image_width,
                                    self.model_path,
                                    self.emb_dim_size,
                                    self.emb_seed)

    # To speed up inference, use non-eager execution.
    @tf.function
    def _infer_fn(inputs):
      return self._emb_model(inputs)

    self._infer_fn = _infer_fn

    @tf.function
    def _convert_multichannel_to_batch(img):
      # Put into range 0 to 1.0 range.
      img = tf.cast(img, tf.float32)
      # Expect (row, col, channel) image.
      imgs = tf.transpose(img, perm=[2, 0, 1])
      imgs = tf.expand_dims(imgs, axis=-1)
      imgs = tf.image.grayscale_to_rgb(imgs)
      return imgs

    self._convert_fn = _convert_multichannel_to_batch

  def process_batched_elements(self, elements):
    # Make a copy to avoid mutating input to beam transform.
    example_dicts = copy.deepcopy(elements)
    img_list = []
    for e in example_dicts:
      img_list.append(self._convert_fn(e[config.IMAGE]))
    imgs = tf.concat(img_list, axis=0)
    embs = self._infer_fn(imgs)
    # Reshape so that each embedding is a concatenation of all channels.
    num_elements, emb_dim = tf.shape(embs)
    num_elements = num_elements // self.input_num_channels
    emb_dim = emb_dim * self.input_num_channels
    embs = tf.reshape(embs, [num_elements, emb_dim])
    embs_list = tf.split(embs, num_elements.numpy())
    for example_dict, emb in zip(example_dicts, embs_list):
      # Remove batch dimension.
      emb = tf.squeeze(emb)
      if len(emb.shape) > 1 or emb.shape[0] != self.final_emb_dim_size:
        raise ValueError(
            f'Shape of emb: {emb.shape} '
            f'does not match expected: ({self.final_emb_dim_size})')
      example_dict[config.EMBEDDING] = emb.numpy()
      yield example_dict

  def process(self, batched_elements):
    """Processes one element batched."""
    for output in self.process_batched_elements(batched_elements):
      yield output

"""Tests for embedding_lib."""
from unittest import mock

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import util as beam_test_util
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.transforms import util as transforms_util

from cell_img.malaria_liver.parasite_emb import config
from cell_img.malaria_liver.parasite_emb import embedding_lib
import numpy as np
import tensorflow as tf


_TEST_MODEL_PATH = 'MOCK'
_TEST_MODEL_IMAGE_SIZE = tuple([299, 299])
_TEST_INPUT_IMAGE_SIZE = tuple([128, 128])
_TEST_INPUT_NUM_CHANNELS = 2
_TEST_EMB_DIM_SIZE = 64
_TEST_EMB_SEED = 4234234


def _create_mock_model(input_image_height, input_image_width,
                       model_path, model_output_size, emb_seed):
  del model_path
  return tf.keras.Sequential([
      tf.keras.Input(shape=(input_image_height, input_image_width, 3)),
      tf.keras.layers.GlobalAveragePooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(
          model_output_size,
          kernel_initializer=tf.keras.initializers.GlorotNormal(seed=emb_seed)),
  ])


class EmbeddingTest(absltest.TestCase):

  @mock.patch.object(embedding_lib, '_create_model', new=_create_mock_model)
  def test_can_run(self):
    num_examples = 7
    batch_size = 3
    random_seed = 123123123
    img_size = (_TEST_INPUT_IMAGE_SIZE[0], _TEST_INPUT_IMAGE_SIZE[1],
                _TEST_INPUT_NUM_CHANNELS)

    inputs = []
    for i in range(num_examples):
      seed = random_seed + i
      random_img = np.random.RandomState(seed=seed).uniform(size=img_size)
      inputs.append({'key': i, config.IMAGE: random_img})

    def run_pipeline(inputs):
      with TestPipeline(beam.runners.direct.DirectRunner()) as p:
        patches = p | beam.Create(inputs)
        batched_patches = patches | transforms_util.BatchElements(
            min_batch_size=1, max_batch_size=batch_size)
        results = batched_patches | beam.ParDo(
            embedding_lib.BatchDoFn(
                model_path=_TEST_MODEL_PATH,
                input_image_size=_TEST_INPUT_IMAGE_SIZE,
                input_num_channels=_TEST_INPUT_NUM_CHANNELS,
                emb_dim_size=_TEST_EMB_DIM_SIZE,
                emb_seed=_TEST_EMB_SEED))
        # Assert that the output PCollection matches the number of input data.
        num_results = results | beam.Map(lambda x: x['key'])
        beam_test_util.assert_that(
            num_results,
            beam_test_util.equal_to([e['key'] for e in inputs]),
            label='CheckSameNumberOutput')

    run_pipeline(inputs)


if __name__ == '__main__':
  absltest.main()

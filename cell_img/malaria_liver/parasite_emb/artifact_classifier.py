"""Contains the beam DoFn to run the artifact classifier lightgbm."""

import copy
from typing import Any, Dict, List, Optional

import apache_beam as beam
from apache_beam.transforms import window
from apache_beam.utils import windowed_value


from cell_img.malaria_liver.parasite_emb import config
import jax
import jax.numpy as jnp

import lightgbm
import numpy as np


# The name of the feature in the TF Example that holds the embedding
# vector to use.
DEFAULT_EMB_FEATURE_NAME = 'embedding/InceptionRand64'

_METRICS_NAMESPACE = 'ArtifactClassifier'

# Number of embedding entries to be droped to remove DAPI.
DAPI_EMB_SIZE = 64


def _load_lgbm_model(model_txt_path: str, scaling_txt_path: str):
  """Load the lightgbm model from disk."""
  # load the model and scaling params
  # lightgbm does not play well with gfile so copy the model to a local
  # tmp file to load.
  artifact_model = lightgbm.Booster(model_file=model_txt_path)
  scaling_params = np.load(scaling_txt_path)

  return artifact_model, scaling_params


def _add_artifact_preds(elements: List[Dict[Any, Any]],
                        model: lightgbm.Booster, scaling_params: np.ndarray,
                        embedding_feature_name: str,):
  """Adds the artifact prediction and stage prediction to the examples.

  Args:
    elements: List of dictionaries. Needs to have fields for the
      embedding vector, and for previous stage prediction inference. The
      examples in the list will be edited in place.
    model: The lightgbm model for the artifact prediction.
    scaling_params: The numpy array with the scaling params to correct the
      prediction values.
    embedding_feature_name: String name of the embedding vector to use.
      It is expected to have 192 dimensions, the first 64 of which are DAPI
      and will be dropped.
  """
  emb_arr = _extract_embedding_array(elements, embedding_feature_name)

  # do prediction
  p_artifact = model.predict(emb_arr)
  p_artifact_scaled = np.array(jax.scipy.special.expit(
      scaling_params[0] + jnp.exp(scaling_params[1]) * p_artifact))

  for artifact_pred, tfe in zip(p_artifact_scaled, elements):
    save_one_prediction(artifact_pred, tfe)


def _extract_embedding_array(
    elements: List[Dict[Any, Any]],
    embedding_feature_name: str) -> np.ndarray:
  """Create a np array with the embeddings from the proto list."""
  emb_list = []
  for example in elements:
    full_emb = example[embedding_feature_name]
    if len(full_emb) != 192:
      raise ValueError(
          'Len of full is %i, but expected 192 dimension embedding vector' %
          len(full_emb))
    # DROP DAPI by appending after DAPI_EMB_SIZE
    emb_list.append(full_emb[DAPI_EMB_SIZE:])

  return np.array(emb_list)


def save_one_prediction(artifact_pred: float,
                        example_element: Dict[Any, Any]):
  """Updates the example with the artifact prediction and a class prediction."""
  example_element[config.ARTIFACT_PRED] = artifact_pred

  # put the pieces together to make a general prediction
  # rules: if artifact_pred > 0.5, it is an artifact
  # otherwise it is the max of the hypnozoite and schizont prediction
  if artifact_pred > 0.5:
    example_element[config.STAGE_RESULT] = config.ARTIFACT
  else:
    prediction_values = example_element[config.STAGE_INFER]
    prediction_class_names = example_element[
        config.STAGE_NAMES]
    h_value = prediction_values[prediction_class_names.index(
        config.HYPNOZOITE)]
    s_value = prediction_values[prediction_class_names.index(
        config.SCHIZONT)]
    if h_value >= s_value:
      example_element[config.STAGE_RESULT] = config.HYPNOZOITE
    else:
      example_element[
          config.STAGE_RESULT] = config.SCHIZONT


class PredictArtifacts(beam.PTransform):
  """Predict staining artifacts for a PCollection of TensorFlow Examples.

  Returns PCollection with copies of each Example with prediction added.
  """

  def __init__(
      self, model_txt_path: str, scaling_txt_path: str,
      embedding_feature_name: Optional[str] = DEFAULT_EMB_FEATURE_NAME):
    """Constructor.

    Args:
      model_txt_path: String path to the lgbm artifact model txt file.
      scaling_txt_path: String path to the numpy array scaling txt file.
      embedding_feature_name: Name of the feature in the example protos that
       contains the embedding vector.
    """
    self._model_txt_path = model_txt_path
    self._scaling_txt_path = scaling_txt_path
    self._embedding_feature_name = embedding_feature_name

  def expand(self, p_elements: beam.PCollection) -> beam.PCollection:
    return p_elements | 'ClassifyingArtifacts' >> beam.ParDo(
        _RunInference(self._model_txt_path, self._scaling_txt_path,
                      self._embedding_feature_name))


class _RunInference(beam.DoFn):
  """A DoFn that runs TF model in batches."""

  def __init__(self, model_txt_path: str, scaling_txt_path: str,
               embedding_feature_name: str):
    """Initialize.

    Args:
      model_txt_path: String path to the lgbm artifact model txt file.
      scaling_txt_path: String path to the numpy array scaling txt file.
      embedding_feature_name: Name of the feature in the example protos that
      contains the embedding vector.
    """
    self.embedding_feature_name = embedding_feature_name
    self.artifact_model, self.scaling_params = _load_lgbm_model(
        model_txt_path, scaling_txt_path)

  def process_batched_elements(self, elements):
    beam.metrics.Metrics.counter(
        _METRICS_NAMESPACE, '_RunTFModelFN_process_batched_elements').inc()

    # Make a copy to avoid mutating input to beam transform.
    elements = copy.deepcopy(elements)
    _add_artifact_preds(elements, self.artifact_model,
                        self.scaling_params, self.embedding_feature_name)

    beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                 'num_classes_computed').inc(len(elements))

    for element_with_class in elements:
      yield windowed_value.WindowedValue(element_with_class, -1,
                                         [window.GlobalWindow()])
      beam.metrics.Metrics.counter(
          _METRICS_NAMESPACE,
          '_RunInference_process_batched_elements_yield').inc()

  def process(self, batched_elements):
    """Processes one element batched."""
    for output in self.process_batched_elements(batched_elements):
      yield output

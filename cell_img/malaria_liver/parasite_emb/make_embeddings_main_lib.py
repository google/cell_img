"""Methods for creating embeddings from images."""

import os
from typing import Any, Dict, Optional, List

import apache_beam as beam
from apache_beam.options import pipeline_options
from apache_beam.transforms import util as transforms_util

from cell_img.malaria_liver.parasite_emb import artifact_classifier
from cell_img.malaria_liver.parasite_emb import center_and_crop
from cell_img.malaria_liver.parasite_emb import config
from cell_img.malaria_liver.parasite_emb import counts_lib
from cell_img.malaria_liver.parasite_emb import embedding_lib
from cell_img.malaria_liver.parasite_emb import finding_lib
from cell_img.malaria_liver.parasite_emb import staging_lib
from cell_img.malaria_liver.parasite_emb import whole_img_lib

import numpy as np

# For the embedding-only pipeline, assume the user wants to keep all the
# objects in the input file, so set the min confidence at zero.
DEFAULT_MIN_CONFIDENCE = 0.0


class MeanEmbeddingFn(beam.transforms.core.CombineFn):
  """CombineFn for computing a mean embedding."""

  def create_accumulator(self):
    return (np.zeros(192), 0)

  def add_input(self, accumulator, element):
    this_sum, this_count = accumulator
    return this_sum + element, this_count + 1

  def merge_accumulators(self, accumulators):
    sums, counts = zip(*accumulators)
    return np.sum(sums, axis=0), sum(counts)

  def extract_output(self, accumulator):
    this_sum, this_count = accumulator
    if this_count == 0:
      return float('NaN')
    return this_sum / float(this_count)


def _convert_namedtuple_for_output(element) -> Dict[Any, Any]:
  """Convert from a NamedTuple back to a dictionary for parquet output."""
  return element._asdict()


def _patch_is_hypnozoite(element: Dict[Any, Any]) -> bool:
  """Determine if the patch is predicted to be a hypnozoite."""
  return element[config.STAGE_RESULT] == config.HYPNOZOITE


def run_embeddings_pipeline(
    image_csv: str,
    metadata_csv: str,
    raw_channel_order: List[str],
    channel_order: List[str],
    whole_image_size: List[int],
    log_brightness_min: List[float],
    log_brightness_max: List[float],
    output_dir: str,
    model_path: str,
    min_confidence: float,
    pixels_per_side: int,
    num_output_shards: Optional[int],
    count_csv_dir: str,
    crop_size: int,
    stain_indices: List[int],
    do_rotate: bool,
    do_center: bool,
    batch_size: int,
    embedding_model_path: str,
    embedding_model_output_size: int,
    embedding_model_output_seed: int,
    staging_model_path: str,
    staging_channel_order: List[int],
    artifact_model_path: str,
    artifact_scaling_path: str,
    options: pipeline_options.PipelineOptions = None) -> beam.Pipeline:
  """Creates the beam pipeline to process the site images and write outputs."""

  input_num_channels = len(channel_order)

  p = beam.Pipeline(options=options)
  p_images = (
      p
      | 'CreateJointMetaData' >> beam.Create(
          whole_img_lib.read_metadata(image_csv, metadata_csv))
      | 'LoadMultiChannelImages' >> beam.Map(
          whole_img_lib.load_img,
          raw_channel_order=raw_channel_order,
          channel_order=channel_order,
          whole_image_size=whole_image_size))

  p_rescaled_log_images = (
      p_images
      | 'RescaleImages' >> beam.Map(
          whole_img_lib.log_and_rescale_img,
          log_brightness_min=log_brightness_min,
          log_brightness_max=log_brightness_max))

  p_patches = (
      p_rescaled_log_images
      | 'BatchImages' >> transforms_util.BatchElements(
          min_batch_size=1, max_batch_size=batch_size)
      | 'FindCellCenters' >> beam.ParDo(
          finding_lib.BatchCellCenterFinder(model_path))
      | 'MakePatches' >> finding_lib.MakePatches(min_confidence,
                                                 pixels_per_side)
      | 'ExtractPatches' >> finding_lib.PatchExtractor())

  p_patches |= 'CenterCropRotate' >> beam.Map(
      center_and_crop.process_one_element,
      stain_indices=stain_indices,
      crop_size=crop_size,
      do_rotate=do_rotate,
      do_center=do_center)

  p_patches |= (
      'StagingBatchPatch' >> transforms_util.BatchElements(
          min_batch_size=1, max_batch_size=batch_size)
      | 'AddStage' >> beam.ParDo(staging_lib.BatchDoFn(
          model_path=staging_model_path,
          input_patch_size=[crop_size, crop_size],
          input_num_channels=input_num_channels,
          channel_order=staging_channel_order))
  )

  p_patches |= (
      'EmbeddingBatchPatch' >> transforms_util.BatchElements(
          min_batch_size=1, max_batch_size=batch_size)
      | 'AddPatchEmbedding' >> beam.ParDo(embedding_lib.BatchDoFn(
          model_path=embedding_model_path,
          input_image_size=[crop_size, crop_size],
          input_num_channels=input_num_channels,
          emb_dim_size=embedding_model_output_size,
          emb_seed=embedding_model_output_seed))
  )

  p_patches |= (
      'ArtifactBatchPatch' >> transforms_util.BatchElements(
          min_batch_size=1, max_batch_size=batch_size)
      | 'PredictArtifacts' >> artifact_classifier.PredictArtifacts(
          model_txt_path=artifact_model_path,
          scaling_txt_path=artifact_scaling_path,
          embedding_feature_name=config.EMBEDDING))

  _ = (
      p_patches
      | 'ConvertPatchImageForOutput' >> beam.Map(
          whole_img_lib.convert_img_for_output)
      | 'WriteAllPatches' >> beam.io.parquetio.WriteToParquet(
          os.path.join(output_dir, 'patches.parquet'),
          schema=config.get_processed_patch_schema(
              patch_image_dim1=crop_size,
              patch_image_dim2=crop_size,
              num_channels=input_num_channels,
              emb_dim_size=embedding_model_output_size * input_num_channels),
          num_shards=num_output_shards))

  # limit to hypnozoites and save
  p_hypnozoite = (
      p_patches | 'FilterHypno' >> beam.Filter(_patch_is_hypnozoite))
  _ = (
      p_hypnozoite
      | 'ConvertHypnozoiteImageForOutput' >> beam.Map(
          whole_img_lib.convert_img_for_output)
      | 'WriteHypnoPatches' >> beam.io.parquetio.WriteToParquet(
          os.path.join(output_dir, 'hypnozoite_patches.parquet'),
          schema=config.get_processed_patch_schema(
              patch_image_dim1=crop_size,
              patch_image_dim2=crop_size,
              num_channels=input_num_channels,
              emb_dim_size=embedding_model_output_size * input_num_channels),
          num_shards=num_output_shards))

  # group hypnozoites by plate/well and calculate the mean embeddings
  # this file will be much smaller, only use 1 output shard
  _ = (
      p_hypnozoite | 'CalcMeanEmb' >> beam.GroupBy(
          batch=lambda e: e[config.BATCH],
          plate=lambda e: e[config.PLATE],
          well=lambda e: e[config.WELL],
          ).aggregate_field(
              lambda e: e[config.EMBEDDING],
              MeanEmbeddingFn(), config.MEAN_EMBEDDING)
      | 'ConvertMeanEmbForOutput' >> beam.Map(_convert_namedtuple_for_output)
      | 'WriteMeanEmbedding' >> beam.io.parquetio.WriteToParquet(
          os.path.join(output_dir, 'well_mean_hypnozoite.parquet'),
          schema=config.get_mean_embedding_schema(
              emb_dim_size=embedding_model_output_size * input_num_channels),
          num_shards=1))

  _ = (
      p_patches | 'object_counts' >> counts_lib.WellCountAggregator(
          output_dir=count_csv_dir,
          count_fn=counts_lib.yield_metrics_per_element))

  return p.run()


def run_emb_creation_only_pipeline(
    image_csv: str,
    object_metadata_csv: str,
    raw_channel_order: List[str],
    channel_order: List[str],
    whole_image_size: List[int],
    log_brightness_min: List[float],
    log_brightness_max: List[float],
    output_dir: str,
    pixels_per_side: int,
    num_output_shards: Optional[int],
    count_csv_dir: str,
    crop_size: int,
    stain_indices: List[int],
    do_rotate: bool,
    do_center: bool,
    batch_size: int,
    embedding_model_path: str,
    embedding_model_output_size: int,
    embedding_model_output_seed: int,
    options: pipeline_options.PipelineOptions = None) -> beam.Pipeline:
  """Creates the beam pipeline to process the site images and write outputs."""

  input_num_channels = len(channel_order)

  p = beam.Pipeline(options=options)
  p_images = (
      p
      | 'CreateJointMetaData' >> beam.Create(
          whole_img_lib.read_object_metadata(image_csv, object_metadata_csv))
      | 'LoadMultiChannelImages' >> beam.Map(
          whole_img_lib.load_img,
          raw_channel_order=raw_channel_order,
          channel_order=channel_order,
          whole_image_size=whole_image_size))

  p_rescaled_log_images = (
      p_images
      | 'RescaleImages' >> beam.Map(
          whole_img_lib.log_and_rescale_img,
          log_brightness_min=log_brightness_min,
          log_brightness_max=log_brightness_max))

  p_patches = (
      p_rescaled_log_images
      | 'BatchImages' >> transforms_util.BatchElements(
          min_batch_size=1, max_batch_size=batch_size)
      | 'FindCellCenters' >> beam.ParDo(
          finding_lib.BatchCsvCenterFinder())
      | 'MakePatches' >> finding_lib.MakePatches(DEFAULT_MIN_CONFIDENCE,
                                                 pixels_per_side)
      | 'ExtractPatches' >> finding_lib.PatchExtractor())

  p_patches |= 'CenterCropRotate' >> beam.Map(
      center_and_crop.process_one_element,
      stain_indices=stain_indices,
      crop_size=crop_size,
      do_rotate=do_rotate,
      do_center=do_center)

  p_patches |= (
      'EmbeddingBatchPatch' >> transforms_util.BatchElements(
          min_batch_size=1, max_batch_size=batch_size)
      | 'AddPatchEmbedding' >> beam.ParDo(embedding_lib.BatchDoFn(
          model_path=embedding_model_path,
          input_image_size=[crop_size, crop_size],
          input_num_channels=input_num_channels,
          emb_dim_size=embedding_model_output_size,
          emb_seed=embedding_model_output_seed))
  )

  _ = (
      p_patches
      | 'ConvertPatchImageForOutput' >> beam.Map(
          whole_img_lib.convert_img_for_output)
      | 'WriteAllPatches' >> beam.io.parquetio.WriteToParquet(
          os.path.join(output_dir, 'patches.parquet'),
          schema=config.get_object_processed_patch_schema(
              patch_image_dim1=crop_size,
              patch_image_dim2=crop_size,
              num_channels=input_num_channels,
              emb_dim_size=embedding_model_output_size * input_num_channels),
          num_shards=num_output_shards))

  # limit to hypnozoites and save
  p_hypnozoite = (
      p_patches | 'FilterHypno' >> beam.Filter(_patch_is_hypnozoite))
  _ = (
      p_hypnozoite
      | 'ConvertHypnozoiteImageForOutput' >> beam.Map(
          whole_img_lib.convert_img_for_output)
      | 'WriteHypnoPatches' >> beam.io.parquetio.WriteToParquet(
          os.path.join(output_dir, 'hypnozoite_patches.parquet'),
          schema=config.get_object_processed_patch_schema(
              patch_image_dim1=crop_size,
              patch_image_dim2=crop_size,
              num_channels=input_num_channels,
              emb_dim_size=embedding_model_output_size * input_num_channels),
          num_shards=num_output_shards))

  # group hypnozoites by plate/well and calculate the mean embeddings
  # this file will be much smaller, only use 1 output shard
  _ = (
      p_hypnozoite | 'CalcMeanEmb' >> beam.GroupBy(
          batch=lambda e: e[config.BATCH],
          plate=lambda e: e[config.PLATE],
          well=lambda e: e[config.WELL],
          ).aggregate_field(
              lambda e: e[config.EMBEDDING],
              MeanEmbeddingFn(), config.MEAN_EMBEDDING)
      | 'ConvertMeanEmbForOutput' >> beam.Map(_convert_namedtuple_for_output)
      | 'WriteMeanEmbedding' >> beam.io.parquetio.WriteToParquet(
          os.path.join(output_dir, 'well_mean_hypnozoite.parquet'),
          schema=config.get_mean_embedding_schema(
              emb_dim_size=embedding_model_output_size * input_num_channels),
          num_shards=1))

  _ = (
      p_patches | 'object_counts' >> counts_lib.WellCountAggregator(
          output_dir=count_csv_dir,
          count_fn=counts_lib.yield_metrics_per_element))

  return p.run()


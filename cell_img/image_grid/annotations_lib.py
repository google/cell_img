"""Library for making annotations to view in neuroglancer.

This assumes the annotations are being added for a layer created by the
image_grid pipeline. The index for this layer is used to convert from metadata
labels like batch plate etc. to YX coordinates.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

from cell_img.image_grid import annotations_utils
from cell_img.image_grid import ts_index_lib
import neuroglancer
from neuroglancer import write_annotations
import pandas as pd


def new_annotation_layer(
    df: pd.DataFrame,
    image_grid_path: str,
    shape: str = 'rect',
    include_properties: bool = True
) -> Tuple[Any, Dict[str, List[Union[float, str]]], Optional[Any],
           Optional[dict]]:
  """Create objects to export a layer of annotations.

  Args:
    df: A DataFrame with one row per annotation. There must be columns
      corresponding to the labels in the image_grid tensorstore index.
    image_grid_path: The path to the image_grid tensorstore on cloud
      corresponding to the annotations.
    shape: Annotation shape. 'point' or 'rect'
    include_properties: True to include properties for each annotation.

  Returns:
    coords: A list of lists. Each entry corresponds to the lower and upper
      bounds for the annotation.
    dim_dict: A dictionary containing the dimension names, units and scales for
      the annotations.
    properties_df: A pd.DataFrame containing the properties for each
      annotation. Note that the annotations in this df cannot be strings,
      instead they are numbers corresponding to an entry in enum_properties.
    enum_properties: A dict matching string annotations to numbers
  """
  s0_metadata = annotations_utils.read_s0_metadata(image_grid_path)
  ts_index = ts_index_lib.index_from_tensorstore_metadata(s0_metadata)
  coords_df = annotations_utils.get_point_coords(df, ts_index)
  if shape == 'point':
    coords = coords_df.values
  elif shape == 'rect':
    coords = annotations_utils.to_rect_coords(df, coords_df, ts_index)
  else:
    raise ValueError('Unknown shape %s' % shape)
  dim_info = annotations_utils.get_dimension_info(
      s0_metadata, coords_df.columns)
  dim_dict = {'names': dim_info[0], 'units': dim_info[1], 'scales': dim_info[2]}

  if include_properties:
    properties_df, enum_properties = annotations_utils.make_properties_df(df)
  else:
    properties_df = None
    enum_properties = None
  return coords, dim_dict, properties_df, enum_properties


def export_annotations(
    df: pd.DataFrame,
    image_grid_path: str,
    local_output_dir: str,
    shape: str = 'rect',
    include_properties: bool = True):
  """Write an annotations layer to a local directory.

  Args:
    df: A DataFrame with one row per annotation. There must be columns
      corresponding to the labels in the image_grid tensorstore index.
    image_grid_path: The path to the image_grid tensorstore on cloud
      corresponding to the annotations.
    local_output_dir: A local path to write the annotation layer.
    shape: Annotation shape. 'point' or 'rect'
    include_properties: True to include properties for each annotation.
  """
  coords, dim_dict, properties_df, enum_properties = new_annotation_layer(
      df, image_grid_path, shape, include_properties)

  coordinate_space = neuroglancer.CoordinateSpace(**dim_dict)

  properties_list = []
  if properties_df is not None:
    for prop_name in properties_df.columns:
      prop_type = str(properties_df[prop_name].dtype)
      if prop_name in enum_properties:
        enum_property = enum_properties[prop_name]
        labels = []
        values = []
        for label, value in enum_property.items():
          labels.append(label)
          values.append(value)
        prop = neuroglancer.AnnotationPropertySpec(
            id=prop_name,
            type=prop_type,
            enum_values=values,
            enum_labels=labels)
      else:
        prop = neuroglancer.AnnotationPropertySpec(id=prop_name, type=prop_type)
      properties_list.append(prop)

    if shape == 'point':
      annotation_type = 'point'
    elif shape == 'rect':
      annotation_type = 'axis_aligned_bounding_box'
    else:
      raise ValueError(
          f'Shape {shape} not supported, must be "point" or "rect"')
    writer = write_annotations.AnnotationWriter(
        coordinate_space=coordinate_space,
        annotation_type=annotation_type,
        properties=properties_list,
    )
    for i, coord in enumerate(coords):
      num_coords = len(coord)
      if shape == 'point':
        writer.add_point(coord, **properties_df.iloc[i].to_dict())
      elif shape == 'rect':
        writer.add_axis_aligned_bounding_box(coord[:num_coords // 2],
                                             coord[num_coords // 2:],
                                             **properties_df.iloc[i].to_dict())
      else:
        raise ValueError(
            f'Shape {shape} not supported, must be "point" or "rect"')
    writer.write(local_output_dir)

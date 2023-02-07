"""Tests for cell_img.image_grid.annotations_lib."""

import os

from absl.testing import absltest
from cell_img.image_grid import annotations_lib
from cell_img.image_grid import downsample_lib
from cell_img.image_grid import test_data
from cell_img.image_grid import ts_write_lib
import tensorstore as ts


class AnnotationsLibTest(absltest.TestCase):

  def test_export_annotations(self):
    tensorstore_path = self.create_tempdir().full_path
    tensorstore_path_s0 = downsample_lib.join_downsample_level_to_path(
        tensorstore_path, 0)
    spec_without_metadata = ts_write_lib.create_spec_from_path(
        tensorstore_path_s0)
    image_metadata_df = test_data.generate_image_metadata_df()

    dataset = ts_write_lib.create_tensorstore(spec_without_metadata,
                                              image_metadata_df, test_data.AXES,
                                              test_data.AXES_WRAP,
                                              test_data.IMAGE_DTYPE,
                                              test_data.IMAGE_SHAPE)
    spec = dataset.spec()
    dataset = ts.open(spec).result()
    _ = downsample_lib.create_downsample_levels(spec)

    site_df = image_metadata_df.drop(columns='stain').drop_duplicates()

    # Export rectangluar annotations.
    annot_dir = self.create_tempdir().full_path
    annotations_lib.export_annotations(site_df, tensorstore_path, annot_dir)
    self.assertTrue(os.path.exists(os.path.join(annot_dir, 'info')))

    # Export point annotations.
    point_annot_dir = self.create_tempdir().full_path
    annotations_lib.export_annotations(
        site_df, tensorstore_path, point_annot_dir, shape='point'
    )
    self.assertTrue(os.path.exists(os.path.join(point_annot_dir, 'info')))


if __name__ == '__main__':
  absltest.main()
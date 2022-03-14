"""Tests for cell_img.image_grid.ts_write_lib."""

from absl.testing import absltest
from cell_img.image_grid import test_data
from cell_img.image_grid import ts_index_lib
from cell_img.image_grid import ts_write_lib
import numpy as np
import tensorstore as ts


class TsWriteLibTest(absltest.TestCase):

  def test_create_spec_from_path_local(self):
    spec = ts_write_lib.create_spec_from_path('/path/to/local/dir')
    self.assertSameStructure(spec['kvstore'], {
        'driver': 'file',
        'path': '/path/to/local/dir',
    })

  def test_create_spec_from_path_gcs(self):
    spec = ts_write_lib.create_spec_from_path('gs://my-bucket/path/to/dir')
    self.assertEqual(spec['path'], 'path/to/dir')
    self.assertSameStructure(spec['kvstore'], {
        'driver': 'gcs',
        'bucket': 'my-bucket',
    })

  def test_create_tensorstore(self):
    spec_without_metadata = ts_write_lib.create_spec_from_path(
        self.create_tempdir().full_path)
    image_metadata_df = test_data.generate_image_metadata_df()

    dataset = ts_write_lib.create_tensorstore(spec_without_metadata,
                                              image_metadata_df, test_data.AXES,
                                              test_data.AXES_WRAP,
                                              test_data.IMAGE_DTYPE,
                                              test_data.IMAGE_SHAPE)
    spec = dataset.spec()
    ts.open(spec).result()

  def test_create_tensorstore_raises_with_bad_shape(self):
    bad_image_shape = (1, 2, 3)
    spec_without_metadata = ts_write_lib.create_spec_from_path(
        self.create_tempdir().full_path)
    image_metadata_df = test_data.generate_image_metadata_df()
    with self.assertRaisesRegex(ValueError,
                                'Image shape should be 2 dimensional'):
      ts_write_lib.create_tensorstore(spec_without_metadata, image_metadata_df,
                                      test_data.AXES, test_data.AXES_WRAP,
                                      test_data.IMAGE_DTYPE, bad_image_shape)

  def test_create_tensorstore_fails_when_it_already_exists(self):
    spec_without_metadata = ts_write_lib.create_spec_from_path(
        self.create_tempdir().full_path)
    image_metadata_df = test_data.generate_image_metadata_df()

    ts_write_lib.create_tensorstore(spec_without_metadata, image_metadata_df,
                                    test_data.AXES, test_data.AXES_WRAP,
                                    test_data.IMAGE_DTYPE,
                                    test_data.IMAGE_SHAPE)
    with self.assertRaisesRegex(ValueError, '.*Error writing local file*'):
      ts_write_lib.create_tensorstore(spec_without_metadata, image_metadata_df,
                                      test_data.AXES, test_data.AXES_WRAP,
                                      test_data.IMAGE_DTYPE,
                                      test_data.IMAGE_SHAPE)

  def test_create_tensorstore_write_image_and_read_it_back(self):
    spec_without_metadata = ts_write_lib.create_spec_from_path(
        self.create_tempdir().full_path)
    image_metadata_df = test_data.generate_image_metadata_df()
    _ = test_data.write_image_metadata_csv_and_images(
        self.create_tempdir().full_path, image_metadata_df)

    image_dtype, image_shape = ts_write_lib.read_image_properties(
        image_metadata_df[test_data.IMAGE_PATH_COL].iloc[0])
    ts_dataset = ts_write_lib.create_tensorstore(spec_without_metadata,
                                                 image_metadata_df,
                                                 test_data.AXES,
                                                 test_data.AXES_WRAP,
                                                 image_dtype, image_shape)
    spec = ts_dataset.spec()
    spec_with_metadata = ts.open(spec).result()

    ts_index = ts_index_lib.index_from_spec(spec_with_metadata.spec().to_json())

    single_image_metadata = image_metadata_df.iloc[-1, :]

    view_slice, image_array = ts_write_lib.get_view_slice_and_read_image(
        single_image_metadata, ts_index, test_data.IMAGE_PATH_COL)

    ts_dataset[view_slice] = image_array
    retrieved_image_array = ts_dataset[view_slice]
    np.testing.assert_equal(image_array, retrieved_image_array)

  def test_get_and_set_path_gcs(self):
    spec = {
        'kvstore': {
            'driver': 'gcs',
            'bucket': 'my_bucket',
        },
        'path': 'path/to/tensorstore',
    }
    self.assertEqual('gs://my_bucket/path/to/tensorstore',
                     ts_write_lib.get_path_from_spec(spec))
    new_path = 'gs://my_bucket/NEW_path/to/tensorstore'
    ts_write_lib.set_path_in_spec(spec, new_path)
    self.assertEqual(new_path, ts_write_lib.get_path_from_spec(spec))

  def test_get_and_set_path_file(self):
    spec = {
        'kvstore': {
            'driver': 'file',
            'path': 'path/to/tensorstore',
        },
    }
    self.assertEqual('path/to/tensorstore',
                     ts_write_lib.get_path_from_spec(spec))
    new_path = 'NEW_path/to/tensorstore'
    ts_write_lib.set_path_in_spec(spec, new_path)
    self.assertEqual(new_path, ts_write_lib.get_path_from_spec(spec))


if __name__ == '__main__':
  absltest.main()

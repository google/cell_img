"""Tests for bald_lib."""

from unittest import mock

from absl.testing import absltest
from cell_img.bald_phenotypes import bald_lib
import numpy as np
import pandas as pd


def mock_train_models_on_cv_folds(split_data,
                                  n_dimensions,
                                  n_labels,
                                  use_label_smoothing=True,
                                  n1=64,
                                  n2=32,
                                  p_dropout=0.5,
                                  epochs=4000,
                                  kwargs=None):
  # unused
  del n_dimensions, n_labels, use_label_smoothing, n1, n2, p_dropout, epochs, kwargs  # pylint: disable=line-too-long
  return [None] * len(split_data), None


def mock_compute_predictions(input_features, model_ensemble, n_samples):
  # make a fake model to convert well label to one-hot encoding
  # this will be used to match the fake model predictions to metadata for
  # testing test_compute_bald_with_cv
  del model_ensemble, n_samples
  enc = np.zeros((len(input_features), 45))
  for i, row in enumerate(input_features):
    enc[i][row[0] * 10 + row[1] - 1] = 1
  return np.stack([enc] * 5, axis=-1)


def mock_make_label_dict(input_df, label_category=None):
  # used for testing test_compute_bald_with_cv
  # this label_dict converts the one-hot encoding back to well format
  del input_df, label_category
  label_dict = {
      i: 'ABCDE'[i // 10] + '{:02d}'.format(i % 10) for i in range(45)
  }
  return label_dict


class BaldPhenotypesSampleWellsTest(absltest.TestCase):

  def test_sample_wells_from_same_plate(self):
    # input data with same batch and plate, different wells, 2 labels
    input_data = pd.DataFrame(np.arange(3), columns=['embedding'])
    input_data['label'] = [1, 2, 2]
    input_data['batch'] = [
        'batch' + str(item) for item in [1] * len(input_data)
    ]
    input_data['plate'] = [
        'plate' + str(item) for item in [1] * len(input_data)
    ]
    input_data['well'] = np.arange(3)
    input_data.set_index(['batch', 'plate', 'well', 'label'], inplace=True)

    # try sampling 1 well/label
    sample = bald_lib.sample_wells(input_data, 'label', 1)
    self.assertLen(sample, 2)
    self.assertLen(sample.index.get_level_values('label').unique(), 2)

    # try sampling 2 wells/label; label 1 should only be sampled once
    sample = bald_lib.sample_wells(input_data, 'label', 2)
    self.assertLen(sample, 3)
    self.assertLen(sample.index.get_level_values('label').unique(), 2)
    self.assertLen(sample.query('label==1'), 1)

  def test_sample_wells_from_different_plate(self):
    # if a well comes from a different plate or batch,
    # it should be recognized as a different well
    input_data = pd.DataFrame(np.arange(3), columns=['embedding'])
    input_data['label'] = [1, 2, 2]
    input_data['batch'] = ['batch' + str(item) for item in [1, 1, 1]]
    input_data['plate'] = ['plate' + str(item) for item in [1, 1, 2]]
    input_data['well'] = [2, 1, 1]
    input_data.set_index(['batch', 'plate', 'well', 'label'], inplace=True)

    sample = bald_lib.sample_wells(input_data, 'label', 1)
    self.assertLen(sample, 2)
    self.assertLen(sample.index.get_level_values('label').unique(), 2)

  def test_sample_wells_from_same_well(self):
    # All the embeddings in the same well should be sampled together
    input_data = pd.DataFrame(np.arange(8), columns=['embedding'])
    input_data['label'] = [1, 1, 1, 1, 2, 2, 2, 2]
    input_data['batch'] = [
        'batch' + str(item) for item in [1] * len(input_data)
    ]
    input_data['plate'] = [
        'plate' + str(item) for item in [1] * len(input_data)
    ]
    input_data['well'] = [1, 1, 3, 3, 2, 2, 4, 4]
    input_data.set_index(['batch', 'plate', 'well', 'label'], inplace=True)

    sample = bald_lib.sample_wells(input_data, 'label', 1)
    self.assertLen(sample, 4)
    self.assertLen(sample.index.get_level_values('label').unique(), 2)


class BaldPhenotypesBalancedInputTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.input_data = pd.DataFrame(np.arange(24), columns=['embedding'])
    self.input_data['compound'] = np.repeat([1, 2, 2, 3, 3, 3], 4)
    self.input_data['well'] = np.arange(24)
    self.input_data.set_index(['well', 'compound'], inplace=True)

    self.cv_folds = 4
    self.cv_data, self.split_idx = bald_lib._make_balanced_model_input_data(
        self.input_data, self.cv_folds)

  def test_make_balanced_model_input_data_output_shapes(self):
    # check shapes of outputs
    self.assertEqual(len(self.cv_data), self.cv_folds)  # pylint: disable= g-generic-assert
    self.assertEqual(len(self.split_idx), self.cv_folds)  # pylint: disable= g-generic-assert

  def test_make_balanced_model_input_data_cv_feature_shapes(self):
    for cv_fold in range(self.cv_folds):
      # check shapes of features
      self.assertEqual(
          self.cv_data[cv_fold][0].shape,
          (self.input_data.shape[0] *
           (self.cv_folds - 2) / self.cv_folds, self.input_data.shape[1]))
      self.assertEqual(
          self.cv_data[cv_fold][2].shape,
          (self.input_data.shape[0] / self.cv_folds, self.input_data.shape[1]))
      self.assertEqual(
          self.cv_data[cv_fold][4].shape,
          (self.input_data.shape[0] / self.cv_folds, self.input_data.shape[1]))

  def test_make_balanced_model_input_data_cv_label_shapes(self):
    for cv_fold in range(self.cv_folds):
      # check shapes of labels
      self.assertEqual(
          self.cv_data[cv_fold][1].shape,
          (self.input_data.shape[0] * (self.cv_folds - 2) / self.cv_folds,
           len(self.input_data.index.get_level_values('compound').unique())))
      self.assertEqual(
          self.cv_data[cv_fold][3].shape,
          (self.input_data.shape[0] / self.cv_folds,
           len(self.input_data.index.get_level_values('compound').unique())))
      self.assertEqual(
          self.cv_data[cv_fold][5].shape,
          (self.input_data.shape[0] / self.cv_folds,
           len(self.input_data.index.get_level_values('compound').unique())))

  def test_make_balanced_model_input_data_cv_index_shapes(self):
    for cv_fold in range(self.cv_folds):
      # check shapes of indices
      self.assertEqual(  # pylint: disable= g-generic-assert
          len(self.split_idx[cv_fold][0]),
          len(self.input_data) * (self.cv_folds - 2) / self.cv_folds)
      self.assertEqual(  # pylint: disable= g-generic-assert
          len(self.split_idx[cv_fold][1]),
          len(self.input_data) / self.cv_folds)
      self.assertEqual(  # pylint: disable= g-generic-assert
          len(self.split_idx[cv_fold][2]),
          len(self.input_data) / self.cv_folds)

  def test_make_balanced_model_input_data_stratified(self):
    # check stratified sampling
    for cv_fold in range(self.cv_folds):
      self.assertCountEqual(self.cv_data[cv_fold][1].sum(axis=0),
                            np.array([1, 2, 3]) * 2)
      self.assertCountEqual(self.cv_data[cv_fold][3].sum(axis=0),
                            np.array([1, 2, 3]))
      self.assertCountEqual(self.cv_data[cv_fold][5].sum(axis=0),
                            np.array([1, 2, 3]))

  def test_make_balanced_model_input_data_split_index_matches_data(self):
    # check cross validation data reconstructed from
    # self.split_idx matches self.cv_data
    for cv_fold in range(self.cv_folds):
      self.assertCountEqual(self.cv_data[cv_fold][0],
                            self.input_data.values[self.split_idx[cv_fold][0]])
      self.assertCountEqual(self.cv_data[cv_fold][2],
                            self.input_data.values[self.split_idx[cv_fold][1]])
      self.assertCountEqual(self.cv_data[cv_fold][4],
                            self.input_data.values[self.split_idx[cv_fold][2]])


class BaldPhenotypesTrainModelTest(absltest.TestCase):

  def test_make_train_model(self):
    model, hist = bald_lib._make_train_model(
        train_features=np.array([[1], [2]]),
        train_labels=np.array([[0, 1, 0], [1, 0, 0]]),
        val_features=np.array([[3]]),
        val_labels=np.array([[1, 0, 0]]),
        n_dimensions=1,
        n_labels=3,
        n1=2,
        n2=2,
        p_dropout=0.5,
        epochs=2,
        kwargs={'patience': 1})
    # check output shape
    prediction = model.predict(np.array([[2], [3]]))
    self.assertEqual(prediction.shape, (2, 3))

    # check model trained for the given amount of epochs
    self.assertEqual(hist.params['epochs'], 2)
    self.assertLen(hist.history['val_loss'], 2)


class BaldPhenotypesComputeTest(absltest.TestCase):

  @mock.patch.object(
      bald_lib, '_train_models_on_cv_folds', new=mock_train_models_on_cv_folds)
  @mock.patch.object(bald_lib, '_make_label_dict', new=mock_make_label_dict)
  @mock.patch.object(
      bald_lib, '_compute_predictions', new=mock_compute_predictions)
  def test_compute_bald_with_cv(self):
    # input dataframe is shuffled and split into cross validation folds;
    # models are trained on the training set of each fold and tested on the
    # corresponding test set, and merged to one dataframe stacked_predictions.
    # I want to test if this function matches each data point in
    # stacked_predictions to the right metadata index of the input dataframe.
    input_data = pd.DataFrame(
        np.vstack((np.repeat([0, 1, 2, 3, 4], 5), np.tile([1, 2, 3, 4, 5],
                                                          5))).T)
    # well value of each row is of the format 'ABCDE'[row[0]]+0+row[1]
    # example [2,3]->C03, [4,1]->E01
    input_data['well'] = [
        '{}{:02d}'.format(alpha, num)  # pylint: disable= g-complex-comprehension
        for alpha in ['A', 'B', 'C', 'D', 'E']
        for num in range(5)
    ]
    input_data['batch'] = 'Pc00-000'
    input_data['plate'] = '00000'
    input_data['compound'] = np.repeat([0, 1, 2, 3, 4], 5)
    input_data.set_index(['batch', 'plate', 'well', 'compound'], inplace=True)

    stats_df, _ = bald_lib.compute_bald_with_cv(
        input_data,
        cv_folds=5,
        n_ensemble_models=25,
        normalize_input_data=False)
    self.assertCountEqual(
        stats_df.index.get_level_values('well').values, stats_df.label.values)


if __name__ == '__main__':
  absltest.main()

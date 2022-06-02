"""Functions for identifying cell phenotypes using BALD.

References: https://arxiv.org/abs/1703.02910, https://arxiv.org/abs/1112.5745
"""

import os
from typing import Optional, Dict, List, Tuple, Text, Any

from absl import logging
import fsspec
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import components
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

sns.set_style('ticks')
sns.set_style('whitegrid')

VERY_SMALL_NUMBER = 0.00001


def _make_label_dict(emb_df: pd.DataFrame,
                     label_category: Optional[Text] = 'compound') -> Dict:
  label_dict = {}
  unique_labels = sorted(emb_df.index.get_level_values(label_category).unique())
  for i, label in enumerate(unique_labels):
    label_dict[i] = label
  return label_dict


def _label_smoothing(input_vectors: np.ndarray, alpha: float,
                     num_categories: int) -> np.ndarray:
  return (1.0 - alpha) * input_vectors + (alpha / num_categories)


def _normalize_wrt_label(
    emb_df: pd.DataFrame,
    label_category: Optional[Text] = 'compound',
    control_label: Optional[Text] = 'DMSO') -> pd.DataFrame:
  """Normalizes embeddings with respect to the control value for the category.

  Args:
    emb_df: Input dataframe with embeddings
    label_category: Quantity to use as labels, e.g. compound, dose
    control_label: Control value for label_category, e.g. DMSO for compound

  Returns:
    pd.DataFrame with normalized embeddings
  """
  label_values = emb_df.query("%s == '%s'" % (label_category, control_label))
  label_mean_emb = np.mean(label_values, axis=0)
  label_std_emb = np.std(label_values, axis=0)
  output_features = emb_df.values
  output_features -= label_mean_emb
  output_features /= label_std_emb
  return pd.DataFrame(
      data=output_features, index=emb_df.index, columns=emb_df.columns)


def sample_wells(emb_df: pd.DataFrame, label_category: str,
                 n_sampled: int) -> pd.DataFrame:
  """Samples at most n_sampled wells from each category.

  This function is used directly from the example colab.

  Args:
    emb_df: Input dataframe with embeddings
    label_category: Quantity to use as labels, e.g. compound, dose
    n_sampled: number of wells to sample from each category. If number of wells
      in the category < n_sampled, max(number of wells in the category,
      n_sampled) is sampled

  Returns:
    pd.DataFrame with sampled wells
  """
  well_counts = emb_df.reset_index().groupby(
      by=['batch', 'plate', 'well', label_category]).count().reset_index()[[
          'batch', 'plate', 'well', label_category
      ]]
  well_counts = well_counts.sample(frac=1)
  labels = well_counts[label_category].unique()
  df_sampled = []
  for label in labels:
    subset = well_counts.query('%s=="%s"' %
                               (label_category, label)).values[:, :3]
    for i in range(min(len(subset), n_sampled)):
      df_sampled.append(
          emb_df.query('batch=="%s" and plate=="%s" and well=="%s"' %
                       (subset[i][0], subset[i][1], subset[i][2])))
  return pd.concat(df_sampled)


def _make_balanced_model_input_data(
    emb_df_normalized: pd.DataFrame,
    cv_folds: int,
    label_category: Optional[Text] = 'compound',
    random_state: Optional[int] = 123123123) -> Tuple[List, List]:
  """Makes stratified k-fold cross validation folds for training.

  This code splits the input embeddings into cross validation folds of sizes
  N_EMBEDDINGS * (cv_folds-2, 1, 1) / cv_folds, each for training, validation
  and test. Indices for the splits are also returned.

  Args:
    emb_df_normalized: Input dataframe with normalized embeddings. Values are of
      size (N_EMBEDDINGS, EMBEDDING_DIMENSIONS)
    cv_folds: number of cross validation folds
    label_category: Quantity to use as labels, e.g. compound, dose
    random_state: Random state used to generate cross validation folds

  Returns:
    2 lists,

    First list (cv_data): Length cv_folds, elements are features and labels
    for each cross validation fold.
    (train_features: np.array of shape (N_EMBEDDINGS * (cv_folds-2)/cv_folds,
                                                      EMBEDDING_DIMENSIONS),
    train_labels: np.array of shape (N_EMBEDDINGS * (cv_folds-2)/cv_folds,
                                                    number of unique labels),

    val_features: np.array of shape (N_EMBEDDINGS/cv_folds,
                                                      EMBEDDING_DIMENSIONS),
    val_labels: np.array of shape (N_EMBEDDINGS/cv_folds,
                                                    number of unique labels),
    test_features: nnp.array of shape (N_EMBEDDINGS/cv_folds,
                                                      EMBEDDING_DIMENSIONS),
    test_labels: np.array of shape (N_EMBEDDINGS/cv_folds,
                                                    number of unique labels)
    )

    Second list (split_idx): Length cv_folds, elements are indices of the input
    dataframe corresponding to the training, validation, and test sets.
    (train_indices: np.array of shape (N_EMBEDDINGS * (cv_folds-2)/cv_folds,),
    val_indices: np.array of shape (N_EMBEDDINGS * 1/cv_folds,),
    test_indices: np.array of shape (N_EMBEDDINGS * 1/cv_folds,)
    )
  """
  input_label_strings = emb_df_normalized.index.get_level_values(
      label_category).values
  enc = OneHotEncoder(sparse=False)
  input_label_string_array = np.expand_dims(input_label_strings, axis=-1)
  input_labels = enc.fit_transform(input_label_string_array)
  input_argmax = np.argmax(input_labels, axis=-1)
  input_features = emb_df_normalized.values

  kfold = model_selection.StratifiedKFold(
      n_splits=cv_folds, shuffle=True, random_state=random_state)
  cv_data = []
  split_idx = []
  for (train_val_ix,
       test_ix) in kfold.split(np.zeros(len(input_argmax)), input_argmax):
    test_features = input_features[test_ix]
    test_labels = input_labels[test_ix]
    train_val_labels = input_labels[train_val_ix]
    train_ix, val_ix, train_labels, val_labels = model_selection.train_test_split(
        np.arange(len(train_val_ix)),
        train_val_labels,
        test_size=1 / (cv_folds - 1),
        stratify=train_val_labels)
    train_features = input_features[train_val_ix[train_ix]]
    train_labels = input_labels[train_val_ix[train_ix]]
    val_features = input_features[train_val_ix[val_ix]]
    val_labels = input_labels[train_val_ix[val_ix]]

    cv_data.append((train_features, train_labels, val_features, val_labels,
                    test_features, test_labels))
    split_idx.append((train_val_ix[train_ix], train_val_ix[val_ix], test_ix))
  return cv_data, split_idx


def _make_train_model(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    n_dimensions: int,
    n_labels: int,
    n1: Optional[int] = 64,
    n2: Optional[int] = 32,
    p_dropout: Optional[float] = 0.5,
    epochs: Optional[int] = 4000,
    kwargs: Optional[Dict[Text, Any]] = None
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
  """Makes and trains a classifier for predicting labels.

  Args:
    train_features: Features used for training
    train_labels: Labels used for training
    val_features: Features used for validation
    val_labels: Labels used for validation
    n_dimensions: number of embedding dimensions
    n_labels: number of unique labels
    n1: number of nodes for first dense layer
    n2: number of nodes for second dense layer
    p_dropout: dropout probability for last layer
    epochs: number of training epochs
    kwargs: parameters used for early stopping. Early stopping is only used if
      extra arguments are provided

  Returns:
    trained model, model training history
  """
  input_layer = tf.keras.Input(shape=(n_dimensions,))
  x = tf.keras.layers.Dense(
      n1, activation=tf.keras.activations.elu)(
          input_layer)
  embedding = tf.keras.layers.Dense(n2, activation=tf.keras.activations.elu)(x)
  x = tf.keras.layers.Dropout(p_dropout)(embedding, training=True)
  logits = tf.keras.layers.Dense(n_labels)(x)

  model_logits = tf.keras.Model(inputs=input_layer, outputs=logits)
  model_logits.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
      loss=tf.nn.softmax_cross_entropy_with_logits,
      metrics=[tf.keras.metrics.CategoricalAccuracy()])
  if kwargs:
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', **kwargs)
    history = model_logits.fit(
        train_features,
        train_labels,
        epochs=epochs,
        callbacks=[es],
        verbose=False,
        batch_size=train_features.shape[0],
        validation_data=(val_features, val_labels))
  else:
    history = model_logits.fit(
        train_features,
        train_labels,
        epochs=epochs,
        verbose=False,
        batch_size=train_features.shape[0],
        validation_data=(val_features, val_labels))
  return model_logits, history


def _train_models_on_cv_folds(
    cv_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                        np.ndarray, np.ndarray]],
    n_dimensions: int,
    n_labels: int,
    use_label_smoothing: Optional[bool] = True,
    n1: Optional[int] = 64,
    n2: Optional[int] = 32,
    p_dropout: Optional[float] = 0.5,
    epochs: Optional[int] = 4000,
    kwargs: Optional[Dict[Text, Any]] = None
) -> Tuple[List[tf.keras.Model], List[tf.keras.callbacks.History]]:
  """Trains models on different cross validation folds.

  Args:
    cv_data: List of length (number of cross validation folds), containing
      (training features, training labels, validation features, validation
      labels, test features, test labels)
    n_dimensions: number of dimensions for embeddings used as features
    n_labels: number of unique labels
    use_label_smoothing: if True, use label smoothing
    n1: number of nodes for first dense layer
    n2: number of nodes for second dense layer
    p_dropout: dropout probability for last layer
    epochs: number of training epochs
    kwargs: parameters used for early stopping. Early stopping is only used if
      extra arguments are provided

  Returns:
    2 lists,
    trained_models: list of trained models for each fold
    model_history: list of model training history for each fold
  """
  trained_models = []
  model_history = []
  for data_split in cv_data:
    train_features, train_labels, val_features, val_labels, _, _ = data_split
    if use_label_smoothing:
      train_labels = _label_smoothing(train_labels, 0.1, n_labels)
    model_logits, history = _make_train_model(
        train_features,
        train_labels,
        val_features,
        val_labels,
        n_dimensions,
        n_labels,
        n1=n1,
        n2=n2,
        p_dropout=p_dropout,
        epochs=epochs,
        kwargs=kwargs)
    trained_models.append(model_logits)
    model_history.append(history)
  return trained_models, model_history


def _compute_predictions_on_test_set(test_cv_folds: List[np.array],
                                     model_ensemble: List[tf.keras.Model],
                                     n_samples: int) -> np.ndarray:
  """Computes prediction on cross validation folds held out for testing.

  For model training on cross validation folds, we held out a test set for each
  fold. Here we compute predictions on the test sets with models trained on the
  training set from the same cross validation fold.

  Args:
    test_cv_folds: cross validation folds held out for testing
    model_ensemble: list of models trained on different cross validation folds.
    n_samples: number of sampled predictions per test set example

  Returns:
    Predictions for each data point in the test sets, merged into one array.
    np.array of shape
    (total number of embeddings, number of unique labels, n_samples)
  """
  new_predictions = []
  for test_features, model in zip(test_cv_folds, model_ensemble):
    new_predictions_split = _compute_predictions(test_features, [model],
                                                 n_samples)
    new_predictions.append(new_predictions_split)
  return np.concatenate(new_predictions)


def _compute_predictions(input_features: np.ndarray,
                         model_ensemble: List[tf.keras.Model],
                         n_samples: int) -> np.ndarray:
  """Computes prediction on a given dataframe of embeddings.

  Args:
    input_features: normalized input feature embeddings (if using a embedding
      dataframe, use df.values)
    model_ensemble: list of models
    n_samples: number of sampled predictions per example

  Returns:
    Predictions for each data point in the input dataframe. np.array of shape
    (total number of embeddings, number of unique labels, n_samples)
  """
  new_predictions = []
  for model_logits in model_ensemble:
    new_predictions.extend([
        tf.nn.softmax(model_logits(input_features, training=True))
        for _ in range(n_samples)
    ])
  new_predictions_stacked = np.stack(new_predictions, axis=-1)

  return new_predictions_stacked


def _entropy_of_average(preds: np.ndarray) -> np.ndarray:
  avg_pred = np.mean(preds, axis=-1)
  return -1 * np.sum(avg_pred * np.log(avg_pred + VERY_SMALL_NUMBER), axis=1)


def _average_entropy(preds: np.ndarray) -> np.ndarray:
  all_preds = -1 * np.sum(preds * np.log(preds + VERY_SMALL_NUMBER), axis=1)
  return np.mean(all_preds, axis=-1)


def _compute_argmax_entropy_bald(
    stacked_predictions: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  # This calculates the BALD objective from https://arxiv.org/pdf/1112.5745.pdf
  argmax = np.argmax(np.mean(stacked_predictions, axis=-1), axis=1)
  entropy = _entropy_of_average(stacked_predictions)
  bald_vals = entropy - _average_entropy(stacked_predictions)
  return argmax, entropy, bald_vals


def make_stats_df_on_data(emb_df: pd.DataFrame,
                          stacked_predictions: np.ndarray,
                          label_dict: Dict[Text, int],
                          split_idx: Optional[Any] = None) -> pd.DataFrame:
  """Makes a dataframe of prediction statistics on cross validation test sets.

  Args:
    emb_df: Input dataframe with embeddings and metadata
    stacked_predictions: predictions on cross validation test sets. np.array of
      shape (total number of embeddings, number of unique labels, n_samples)
    label_dict: dictionary mapping labels to one hot encoding indices
    split_idx: indices of cross validation fold splits. Set to None if there are
      no cross validation folds

  Returns:
    pd.DataFrame. Same multiindex with emb_df and columns argmax, label,
    label_prob, entropy, bald, avg_predictions
  """
  if split_idx:
    # metadata_idx pulls out dataframe indices corresponding to the test set
    metadata_idx = emb_df.index.to_frame(index=False)
    # split_idx=[(train indices, validation indices, test indices)
    #  for each fold],
    # so this line pulls out the test set indices for each k-fold
    metadata_idx = pd.concat([metadata_idx.loc[item[-1]] for item in split_idx])
    metadata_idx.set_index(list(metadata_idx.columns), inplace=True)
    df_index = metadata_idx.index
  else:
    df_index = emb_df.index

  argmax_np, entropy_np, bald_np = _compute_argmax_entropy_bald(
      stacked_predictions)
  avg_predictions = np.split(
      np.mean(stacked_predictions, axis=-1), stacked_predictions.shape[0])
  label_list = [label_dict[x] for x in argmax_np]
  label_prob_list = [
      probs[0][i] for probs, i in zip(avg_predictions, argmax_np)
  ]
  df_dict = {
      'argmax': argmax_np,
      'label': label_list,
      'label_prob': label_prob_list,
      'entropy': entropy_np,
      'bald': bald_np,
      'avg_predictions': avg_predictions,
  }
  return pd.DataFrame(df_dict, index=df_index)


def compute_bald_with_cv(
    input_data_df: pd.DataFrame,
    output_name: Optional[Text] = '',
    label_category: Optional[Text] = 'compound',
    control_label: Optional[Text] = 'DMSO',
    cv_folds: Optional[int] = 6,
    use_label_smoothing: Optional[bool] = True,
    n1: Optional[int] = 64,
    n2: Optional[int] = 32,
    p_dropout: Optional[float] = 0.5,
    epochs: Optional[int] = 4000,
    n_ensemble_models: Optional[int] = 100,
    result_copy_path: Optional[Text] = '',
    return_models: Optional[bool] = False,
    normalize_input_data: Optional[bool] = True,
    kwargs: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, List]:
  """Trains an ensemble of models on embeddings and compute BALD objective.

  This function takes in a dataframe of embeddings, trains an ensemble of models
  on k-fold cross validation folds, and makes predictions on test sets of each
  cross validation fold. Model predictions and BALD objectives are saved to a
  csv file.

  Args:
    input_data_df: Input dataframe with embeddings and metadata
    output_name: string, name for saving output data. If empty, output data is
      not saved
    label_category: str, Quantity to use as labels, e.g. compound, dose
    control_label: str, Control value for label_category, e.g. DMSO for compound
    cv_folds: int, number of k-fold cross validation folds
    use_label_smoothing: bool, use label smoothing if True
    n1: int, number of nodes for first hidden layer of ensemble models
    n2: int, number of nodes for second hidden layer of ensemble models
    p_dropout: dropout probability for last layer of ensemble models
    epochs: number of training epochs for ensemble models
    n_ensemble_models: number of models used in the ensemble
    result_copy_path: if not empty, copy results (output csv, if plot_bald_label
      the plot is also copied) to this path
    return_models: return the trained model ensemble as the second argument
    normalize_input_data: normalize embeddings with respect to embeddings for
      control values
    kwargs: parameters used for early stopping. Early stopping is only used if
      extra arguments are provided

  Returns:
    pd.DataFrame of embeddings with model predictions, list;
    list is empty if return_models is False, otherwise size cv_folds list of
    keras.engine.functional.Functional
  """

  if normalize_input_data:
    input_data_df = _normalize_wrt_label(
        input_data_df,
        label_category=label_category,
        control_label=control_label)
  n_dimensions = input_data_df.shape[1]

  # make a dictionary mapping labels to one hot encoding indices
  label_dict_comp = _make_label_dict(
      input_data_df, label_category=label_category)
  n_labels = len(label_dict_comp)

  # make stratified k-fold cross validation splits
  split_data, split_idx = _make_balanced_model_input_data(
      input_data_df, cv_folds, label_category=label_category)

  # train models on each cross validation folds
  trained_models, _ = _train_models_on_cv_folds(
      split_data,
      n_dimensions,
      n_labels,
      use_label_smoothing=use_label_smoothing,
      n1=n1,
      n2=n2,
      p_dropout=p_dropout,
      epochs=epochs,
      kwargs=kwargs)

  # make predictions on test sets for each cv fold
  n_samples_per_fold = n_ensemble_models // len(trained_models)
  test_cv_folds = [item[-2] for item in split_data]
  stacked_predictions = _compute_predictions_on_test_set(
      test_cv_folds, trained_models, n_samples_per_fold)

  stats_df = make_stats_df_on_data(input_data_df, stacked_predictions,
                                   label_dict_comp, split_idx)
  # save results
  if output_name:
    with fsspec.open(output_name + '.csv', 'w') as f:
      stats_df.to_csv(f)
  else:
    logging.info(
        'output_name is not provided, results are not saved to a file.')

  # Copy results to cloud
  if output_name and result_copy_path:
    with fsspec.open(os.path.join(result_copy_path, output_name + '.csv'),
                     'w') as f:
      stats_df.to_csv(f)

  if return_models:
    return stats_df, trained_models
  else:
    return stats_df, []


def plot_bald_probability(
    stats_df: pd.DataFrame,
    alpha: Optional[float] = 1,
    label_category: Optional[Text] = 'compound') -> plt.figure:
  """Plots BALD objective against label probabilities.

  Args:
    stats_df: Input dataframe with model predictions, expected to have columns:
      "label" which has the assigned label, which is expected to be one of the
      strings found in the label_category values. "label_prob" which has the
      model probabilities for the labels. "bald" which is the value of the BALD
      objective from the model.
    alpha: alpha for the scatterplot output
    label_category: str, Quantity to use as labels, e.g. compound, dose

  Returns:
    matplotlib figure plotting label probabilities and BALD objective
    for each category. Pink dots indicate the data points where the
    BALD classifier matches the true category.
  """
  unique_labels = sorted(stats_df.reset_index()[label_category].unique())

  nrows, ncols = np.ceil(len(unique_labels) / 4).astype(int), 4
  fig, ax_s = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows))
  flattened_ax = [item for sublist in ax_s for item in sublist]

  i = 0
  for label in unique_labels:
    ax = flattened_ax[i]
    i += 1

    examples_in_category = stats_df.query('%s=="%s"' % (label_category, label))
    true_positives = examples_in_category.query('label=="%s"' % label)
    ax.scatter(
        examples_in_category.label_prob,
        examples_in_category.bald,
        s=15,
        alpha=alpha,
        color='gray')
    ax.scatter(
        true_positives.label_prob,
        true_positives.bald,
        s=15,
        alpha=alpha,
        color='magenta')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title(label)
  fig.tight_layout(pad=0)
  return fig


def plot_confusion_matrix(
    stats_df: pd.DataFrame,
    label_category: Optional[Text] = 'compound',
    control_category: Optional[Text] = 'DMSO',
    figsize: Optional[Tuple] = (7, 6)
) -> Tuple[np.ndarray, List[Text]]:
  """Plots confusion matrices for label categories.

  Args:
    stats_df: Input dataframe with model predictions, expected to have columns:
      "label" which has the assigned label, which is expected to be one of the
      strings found in the label_category values, and f"label_category" which is
      the quantity used as labels, e.g. compound, dose.
    label_category: str, Quantity to use as labels, e.g. compound, dose. This
      should be one of the columns of stats_df
    control_category: value for the control category, e.g. "DMSO" for compound.
      If not empty, this category is dropped from comuting the control matrix.
    figsize: tuple, output figure size

  Returns:
    np.array of the confusion matrix and corresponding labels
  """
  if control_category:
    stats_df = stats_df.query('%s!="%s"' % (label_category, control_category))
  labels = sorted(stats_df[label_category].unique())
  confusion_mat = sklearn.metrics.confusion_matrix(
      stats_df[label_category].values, stats_df.label.values, labels=labels)
  plt.figure(figsize=figsize)
  sns.heatmap(confusion_mat, xticklabels=labels, yticklabels=labels, cmap='hot')
  plt.xlabel('Predicted label')
  plt.ylabel('True label')
  plt.tight_layout()
  plt.show()

  return confusion_mat, labels


def get_categories_with_similar_phenotypes(
    confusion_mat: np.ndarray,
    labels: List[Text],
    similarity_cutoff: Optional[float] = 0.15,
    min_samples: Optional[int] = 3) -> List[List[Text]]:
  """Finds groups of categories the BALD classifier misclassifies as other members of the group.

  Args:
    confusion_mat: np.array, confusion matrix from BALD classifier
    labels: list of labels corresponding to the entries of the confusion matrix
    similarity_cutoff: float, threshold used for grouping labels together. If
      the (number of entries in label A predicted as label B)/(total number of
      entries in label A) < similarity_cutoff, labels A and B are not grouped
      together.
    min_samples: int, confusion matrix entries less than min_samples are dropped

  Returns:
    list of lists with each sublist containing similar labels
  """
  cmat_frac = confusion_mat / np.sum(confusion_mat, axis=1)
  cmat_frac = np.multiply(cmat_frac, confusion_mat > min_samples)
  cmat_frac = (cmat_frac > similarity_cutoff).astype(int)
  # treat the confusion matrix as a graph with labels as nodes,
  # and find connected components
  connected_labels = np.vstack(np.where(cmat_frac)).T.tolist()
  connected_labels = [
      item for item in connected_labels if item[::-1] in connected_labels
  ]
  label_graph = nx.Graph()
  for item in connected_labels:
    label_graph.add_edge(*item)
  return [[labels[i]
           for i in s]
          for s in components.connected_components(label_graph)]

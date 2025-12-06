from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, f1_score, average_precision_score
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_selection import SelectFromModel
from tabpfn.model_loading import load_model_criterion_config
import torch
from torch import nn
import pickle
import os
import numpy as np
import pandas as pd


class PredictionResults:
    def __init__(self, ground_truth, prediction_probas, target_names=None):
        self.prediction_probas = prediction_probas
        self.ground_truth = ground_truth
        self.target_names = target_names

    def get_classification_report(self, print_report=True, **kwargs):
        if print_report:
            print(
                classification_report(
                    self.ground_truth,
                    self.prediction_probas.argmax(axis=1),
                    target_names=self.target_names,
                    **kwargs
                )
            )
        return classification_report(
            self.ground_truth,
            self.prediction_probas.argmax(axis=1),
            target_names=self.target_names,
            output_dict=True,
            **kwargs
        )

    def get_roc_auc_score(self, **kwargs):
        return roc_auc_score(self.ground_truth, self.prediction_probas, **kwargs)

    def get_f1_score(self, average="weighted", **kwargs):
        return f1_score(self.ground_truth, self.prediction_probas.argmax(axis=1), average=average)

    def save_prediction_results(self, filename="prediction_results.pkl", directory=None):
        if directory:
            os.makedirs(directory, exist_ok=True)
            filepath = os.path.join(directory, filename)
        else:
            filepath = filename

        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_prediction_results(cls, filename="prediction_results.pkl", directory=None):
        filepath = os.path.join(directory, filename) if directory else filename
        with open(filepath, "rb") as f:
            return pickle.load(f)


def feature_reduction_agglomeration(X: pd.DataFrame, n_features):
    if X.shape[1] <= n_features:
        return X
    feature_agglomeration = FeatureAgglomeration(n_clusters=n_features)
    reduced = feature_agglomeration.fit_transform(X)

    # Build a DataFrame with the new feature names
    columns = X.columns
    labels = feature_agglomeration.labels_
    joined_entries = [";".join(columns[labels == i]) for i in range(n_features)]
    reduced_df = pd.DataFrame(reduced, columns=joined_entries)
    reduced_df.index = X.index

    return reduced_df


def feature_reduction_correlation(X: pd.DataFrame, n_features):
    corr = np.abs(np.corrcoef(X.T))
    np.fill_diagonal(corr, 0)
    np.nan_to_num(corr, copy=False)

    clustering = FeatureAgglomeration(
        n_clusters=n_features, metric="precomputed", linkage="complete"
    )
    clustering.fit(1 - corr)
    reduced = clustering.transform(X)
    # Build a DataFrame with the new feature names
    columns = X.columns
    labels = clustering.labels_
    joined_entries = [";".join(columns[labels == i]) for i in range(n_features)]
    reduced_df = pd.DataFrame(reduced, columns=joined_entries)
    reduced_df.index = X.index
    return reduced_df


def feature_reduction_from_model(
    X_train, y_train, X_test, n_features, model=LogisticRegression(max_iter=1000)
):
    if X_train.shape[1] <= n_features:
        return X_train, X_test
    fs = SelectFromModel(model, max_features=n_features)
    fs.fit(X_train, y_train)
    X_train_reduced = fs.transform(X_train)
    X_test_reduced = fs.transform(X_test)
    # Build a DataFrame with the new feature names
    reduced_df_train = pd.DataFrame(X_train_reduced, columns=fs.get_feature_names_out())
    reduced_df_train.index = X_train.index
    reduced_df_test = pd.DataFrame(X_test_reduced, columns=fs.get_feature_names_out())
    reduced_df_test.index = X_test.index
    return reduced_df_train, reduced_df_test


def load_pytorch_tabpfn(device="cuda:0"):
    models, _, _, _ = load_model_criterion_config(
        model_path=None,
        check_bar_distribution_criterion=False,
        cache_trainset_representation=False,
        which="classifier",
        version="v2.5",
        download_if_not_exists=True,
    )
    model = models[0]
    # Disable feature grouping
    model.features_per_group = 1
    model = model.to(device)
    model = model.eval()
    return model


def get_feature_dependent_noise(x_tensor, std):
    # The noise std is proportional to the standard deviation of each feature
    stds = x_tensor.std(dim=0, keepdim=True)
    stds[stds == 0] = 1  # Avoid division by zero
    noise = torch.randn_like(x_tensor) * (std * stds)
    return noise


def get_linear_added_features(x, features_to_be_added, sparsity, noise_std):
    """
    Adds new linear features to the input tensor with controlled sparsity and feature-dependent noise.
    """
    W_sparse = nn.Linear(x.shape[-1], features_to_be_added, bias=False)
    W_sparse.weight.data *= (torch.rand_like(W_sparse.weight) < sparsity).float()
    x = W_sparse(x)

    dependent_noise = get_feature_dependent_noise(x, noise_std)
    x += dependent_noise
    return x.detach()


def get_new_features(
    x_tensor,
    features_to_be_added,
    sparsity=0.01,
    noise_std=3,
    include_original=True,
    include_original_prob=0.5,
):

    x_new = get_linear_added_features(
        x_tensor, features_to_be_added, sparsity=sparsity, noise_std=noise_std
    )
    if np.random.rand() < include_original_prob and include_original:
        x_new = torch.cat([x_tensor, x_new], dim=-1)
        x_new = x_new[..., torch.randperm(x_new.shape[-1])]
    return x_new.detach()

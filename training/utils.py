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
                    **kwargs,
                )
            )
        return classification_report(
            self.ground_truth,
            self.prediction_probas.argmax(axis=1),
            target_names=self.target_names,
            output_dict=True,
            **kwargs,
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


@torch.no_grad()
def detect_categorical_features(x_tensor, max_unique_ratio=0.05, max_unique_abs=20):
    """Detect categorical columns using unique counts."""
    B, T, D = x_tensor.shape
    if D == 0:
        return torch.zeros(D, dtype=torch.bool, device=x_tensor.device)

    total_samples = B * T
    if total_samples == 0:
        return torch.zeros(D, dtype=torch.bool, device=x_tensor.device)
    elif total_samples == 1:
        n_unique = torch.ones(D, device=x_tensor.device)
    else:
        # flatten to get category counts over all batches
        x_flat = x_tensor.reshape(-1, D)  # is now shape [B*T, D]
        n_unique = torch.empty(D, device=x_tensor.device, dtype=torch.int32)
        for d in range(D):
            n_unique[d] = torch.unique(x_flat[:, d]).numel()

    ratio_mask = (n_unique.float() / total_samples) < max_unique_ratio
    abs_mask = n_unique <= max_unique_abs
    return ratio_mask | abs_mask


@torch.no_grad()
def merge_cats_frequency(base: torch.Tensor, max_cats: int) -> torch.Tensor:
    """Merge rare categories into frequent ones to reduce category count."""
    vals, inv = torch.unique(base, return_inverse=True)
    K = vals.numel()
    if K <= max_cats:
        return base

    counts = torch.bincount(inv, minlength=K)
    order = torch.argsort(counts)
    head = order[-max_cats:]
    tail = order[:-max_cats]

    tgt_indices = torch.randint(0, head.numel(), (tail.numel(),), device=base.device)
    mapping = torch.arange(K, device=base.device)
    mapping[tail] = head[tgt_indices]

    codes = mapping[inv]
    return vals[codes]


@torch.no_grad()
def strat_partial_patch_sparse(X_cat, sparsity=0.1, max_cats=100):
    """Create a new categorical feature by mixing random parts of existing ones."""
    T, D = X_cat.shape
    n_dep = max(1, int(sparsity * D))

    # sample donor columns
    dep_idx = torch.randperm(D, device=X_cat.device)[:n_dep]

    # for each time step, pick a random donor column index
    donor_idx = torch.randint(0, n_dep, (T,), device=X_cat.device)
    new_col = X_cat[torch.arange(T, device=X_cat.device), dep_idx[donor_idx]]

    # merge rare categories
    new_col = merge_cats_frequency(new_col, max_cats=max_cats)
    return new_col


@torch.no_grad
def get_categorical_added_features(X_cat, n_cat, sparsity=0.05, max_cats=20):
    B, T, D = X_cat.shape
    X_new_cat = torch.empty(size=(B, T, n_cat), device=X_cat.device)
    # sample max categories according to a beta distribution
    max_cats_values = sample_max_cats(size=n_cat, high=max_cats, lam=0.08, device=X_cat.device)
    for b in range(B):
        for i in range(n_cat):
            new_feature = strat_partial_patch_sparse(
                X_cat[b], sparsity=sparsity, max_cats=max_cats_values[i]
            )
            X_new_cat[b, :, i] = new_feature
    return X_new_cat


# TODO !IMPORTANT Double check (was dim=0 before)
@torch.no_grad()
def get_feature_dependent_noise(x_tensor, std):
    stds = x_tensor.std(dim=0, keepdim=True, correction=0)
    if torch.isnan(stds).any():
        print(f"NAN detected in stds in get_feature_dependent_noise", flush=True)
        stds = torch.nan_to_num(stds, nan=1.0)
    stds[stds == 0] = 1
    noise = torch.randn_like(x_tensor) * (std * stds)
    return noise


@torch.no_grad()
def get_linear_added_features(x, features_to_be_added, sparsity, noise_std):
    """
    Adds new linear features to the input tensor with controlled sparsity and feature-dependent noise.
    """
    W_sparse = nn.Linear(x.shape[-1], features_to_be_added, bias=False).to(x.device)
    W_sparse.weight.data *= (torch.rand_like(W_sparse.weight) < sparsity).float()
    x = W_sparse(x)

    dependent_noise = get_feature_dependent_noise(x, noise_std)
    x += dependent_noise
    return x.detach()

@torch.no_grad()
def get_new_features_mixed_batched(
    x_tensor,
    features_to_be_added,
    sparsity=0.01,
    noise_std=3,
    max_cats=20,
    include_original=True,
    include_original_prob=0.5,
):
    """
    This version operates on the entire batch and has the downside,
    that the features between the datasets within the batch are not shared
    so it has wrong assumptions. It still works since the batch size for training is so low but the cleaner version
    is >>get_new_features_mixed<<.
    """
    # detect all categorical features
    cat_mask = detect_categorical_features(x_tensor=x_tensor)

    X_cat = x_tensor[..., cat_mask] if cat_mask.any() else None
    X_cont = x_tensor[..., ~cat_mask] if (~cat_mask).any() else None

    # ratio of categorical to total features
    cat_ratio = (X_cat.shape[-1] if X_cat is not None else 0) / max(x_tensor.shape[-1], 1)
    n_cat = int(features_to_be_added * cat_ratio)
    n_cont = features_to_be_added - n_cat

    # add new features
    X_new_cont = (
        get_linear_added_features(X_cont, n_cont, sparsity, noise_std)
        if X_cont is not None and n_cont > 0
        else None
    )
    X_new_cat = (
        get_categorical_added_features(X_cat, n_cat, sparsity=sparsity, max_cats=max_cats)
        if X_cat is not None and n_cat > 0
        else None
    )

    # combine
    if X_new_cont is not None and X_new_cat is not None:
        X_new = torch.cat([X_new_cont, X_new_cat], dim=-1)
    elif X_new_cont is not None:
        X_new = X_new_cont
    elif X_new_cat is not None:
        X_new = X_new_cat
    else:
        X_new = x_tensor  # nothing to add

    # optionally include and shuffle original features
    if include_original and np.random.rand() < include_original_prob:
        X_new = torch.cat([x_tensor, X_new], dim=-1)
        X_new = X_new[..., torch.randperm(X_new.shape[-1])]

    return X_new.detach()


@torch.no_grad()
def get_new_features_mixed(
    x_tensor,
    features_to_be_added,
    sparsity=0.01,
    noise_std=3,
    max_cats=20,
    include_original=True,
    include_original_prob=0.5,
):

    include_orig = np.random.rand() < include_original_prob and include_original
    b, t, d = x_tensor.shape
    x_widened = []
    for i in range(b):
        # for every dataset perform feature widening independently!
        new_x = get_new_features_mixed_helper(
            x_tensor=x_tensor[i].unsqueeze(0),
            features_to_be_added=features_to_be_added,
            sparsity=sparsity,
            noise_std=noise_std,
            max_cats=max_cats,
            include_original=include_orig,
        )
        x_widened.append(new_x.squeeze(0))
        # print(new_x.shape)
    Ds = [xi.shape[-1] for xi in x_widened]
    minD = min(Ds)

    x_widened = torch.stack([xi[..., :minD] for xi in x_widened], dim=0)

    return x_widened


@torch.no_grad()
def get_new_features_mixed_helper(
    x_tensor, features_to_be_added, sparsity=0.01, noise_std=3, max_cats=20, include_original=True
):
    # detect all categorical features
    cat_mask = detect_categorical_features(
        x_tensor=x_tensor, max_unique_ratio=0.001, max_unique_abs=20
    )

    X_cat = x_tensor[..., cat_mask] if cat_mask.any() else None
    X_cont = x_tensor[..., ~cat_mask] if (~cat_mask).any() else None
    # ratio of categorical to total features
    cat_ratio = (X_cat.shape[-1] if X_cat is not None else 0) / max(x_tensor.shape[-1], 1)
    #print(f"Cat ratio: {cat_ratio}")
    n_cat = int(features_to_be_added * cat_ratio)
    n_cont = features_to_be_added - n_cat
    # print(f"Cat ratio is: {cat_ratio}")

    # add new features
    X_new_cont = (
        get_linear_added_features(X_cont, n_cont, sparsity, noise_std)
        if X_cont is not None and n_cont > 0
        else None
    )
    X_new_cat = (
        get_categorical_added_features(X_cat, n_cat, sparsity=sparsity, max_cats=max_cats)
        if X_cat is not None and n_cat > 0
        else None
    )

    # combine
    if X_new_cont is not None and X_new_cat is not None:
        X_new = torch.cat([X_new_cont, X_new_cat], dim=-1)
    elif X_new_cont is not None:
        X_new = X_new_cont
    elif X_new_cat is not None:
        X_new = X_new_cat
    else:
        X_new = x_tensor  # nothing to add

    # optionally include and shuffle original features
    if include_original:
        X_new = torch.cat([x_tensor, X_new], dim=-1)
        X_new = X_new[..., torch.randperm(X_new.shape[-1])]

    return X_new.detach()


def sample_max_cats(low=3, high=20, size=1, lam=0.25, device="cuda"):
    """
    Sample integers between [low, high] (inclusive)
    from a discrete exponential distribution.
    Smaller x are more likely for larger lambda.
    """
    values = torch.arange(low, high + 1, device=device)
    weights = torch.exp(-lam * (values - low).float())
    probs = weights / weights.sum()
    samples = torch.multinomial(probs, num_samples=size, replacement=True)
    return values[samples]

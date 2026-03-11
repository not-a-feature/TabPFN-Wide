from xgboost import XGBClassifier
from pytabkit import RealMLP_TD_Classifier
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class XGBoostClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, device="cuda", n_estimators=1000, n_jobs=-1, random_state=42, **kwargs):
        self.device = device
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        tree_method = "hist"  # Efficient default
        device_arg = "cpu"

        if "cuda" in self.device or "gpu" in self.device:
            device_arg = self.device

        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            tree_method=tree_method,
            device=device_arg,
            **self.kwargs,
        )

        self.model.fit(X, y)
        return self

    def to(self, device):
        pass

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class RealMLPClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, device="cuda", random_state=42, **kwargs):
        self.device = device
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.model = RealMLP_TD_Classifier(
            device=self.device,
            random_state=self.random_state,
            n_cv=1,
            n_refit=0,
            verbosity=0,
            **self.kwargs,
        )
        self.model.fit(X, y)
        # Patch eval method
        if not hasattr(self.model, "eval"):
            self.model.eval = lambda: None
        return self

    def to(self, device):
        pass

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

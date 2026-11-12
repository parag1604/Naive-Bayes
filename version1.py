"""version1: numerically stable Gaussian naive Bayes (log-space + variance smoothing).

What is NEW compared to version0
- Work with log-probabilities. A product of D small densities underflows to 0
  for large D, but the sum of their logs is perfectly representable:
      log p(y=k | x) = log p(y=k) + sum_j log N(x_j; mu_kj, var_kj) + const.
- Variance smoothing. If a feature is constant within a class its variance is
  0 and the Gaussian collapses to a spike. We add a small epsilon to every
  variance (scikit-learn uses 1e-9 times the largest feature variance).
- Vectorised with numpy broadcasting: no Python loops over samples/features.

With these two changes the model also works on the 64-pixel digits dataset,
where version0 fails:  python version1.py --dataset digits
"""
import numpy as np
from sklearn.naive_bayes import GaussianNB

from utils import load_data, get_parser, evaluate


class GaussianNaiveBayes:
    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
        self.labels = None
        self.log_priors = None  # shape (K,)
        self.means = None       # shape (K, D)
        self.vars = None        # shape (K, D)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.labels = np.unique(y)
        # epsilon is relative to the data scale, as in scikit-learn
        epsilon = self.var_smoothing * X.var(axis=0).max()
        self.log_priors = np.log([np.mean(y == k) for k in self.labels])
        self.means = np.stack([X[y == k].mean(axis=0) for k in self.labels])
        self.vars = np.stack([X[y == k].var(axis=0) for k in self.labels]) + epsilon

    def predict_log_joint(self, X: np.ndarray) -> np.ndarray:
        """log p(y=k) + log p(x | y=k) for every sample and class: shape (N, K)."""
        # broadcast (N, 1, D) against (K, D) -> (N, K, D), then sum over features
        diff = X[:, None, :] - self.means
        log_lik = -0.5 * (np.log(2 * np.pi * self.vars) + diff ** 2 / self.vars)
        return self.log_priors + log_lik.sum(axis=2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.labels[np.argmax(self.predict_log_joint(X), axis=1)]


def main(args) -> None:
    X_train, X_test, y_train, y_test = load_data(args.dataset)
    evaluate(GaussianNaiveBayes(), X_train, X_test, y_train, y_test, "Our")
    evaluate(GaussianNB(), X_train, X_test, y_train, y_test, "Sklearn")


if __name__ == "__main__":
    main(get_parser(__doc__.splitlines()[0]).parse_args())

"""version0: Gaussian naive Bayes from scratch.

What this file teaches
- Bayes rule for classification: predict argmax_k p(y=k | x), and
  p(y=k | x) is proportional to p(y=k) * p(x | y=k).
- The "naive" assumption: features are conditionally independent given the
  class, so p(x | y=k) = prod_j p(x_j | y=k). Each factor is a 1-D Gaussian
  whose mean and std are estimated from the class-k training points only.
- Training is just counting and averaging; there is no optimisation loop.
- We check the result against scikit-learn's GaussianNB.

Try `python version0.py --dataset digits` too: it breaks. Some pixels are
constant within a class (std = 0), so the Gaussian density is undefined, and
multiplying 64 small densities underflows. version1 fixes both problems.
"""
import numpy as np
from scipy.stats import norm
from sklearn.naive_bayes import GaussianNB

from utils import load_data, get_parser, evaluate


class GaussianNaiveBayes:
    def __init__(self):
        self.params = []
        self.labels = []
        self.priors = []

    def _get_gaussian_probability(
            self,
            x: float,
            mu: float,
            std: float,
    ) -> float:
        return norm.pdf(x, loc=mu, scale=std)

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray
    ) -> None:
        # Store the class labels and their prior probabilities p(y = k)
        self.labels = np.unique(y)
        self.priors = [np.mean(y == label) for label in self.labels]

        # Calculate the class conditional mean and std for each feature
        # (since we are assuming features are mutually independent)
        self.params = []
        for feature_idx in range(X.shape[1]):
            self.params.append([])
            feature = X[:, feature_idx]
            # Calculate the mean and std for each class for the given feature
            for label in self.labels:
                cols_given_label = feature[y == label]
                mu = np.mean(cols_given_label)
                std = np.std(cols_given_label)
                self.params[feature_idx].append((mu, std))

    def predict(
            self,
            X: np.ndarray
    ) -> np.ndarray:
        preds = []
        # Predict the label for each sample
        for x in X:
            # unnormalised posterior p(y = k) * prod_j p(x_j | y = k)
            prob_y_given_X = np.ones(len(self.labels))
            for k in range(len(self.labels)):
                # Prior probability of the label
                prob_y_given_X[k] *= self.priors[k]
                # Likelihood of each feature under the class-k Gaussian
                for i in range(len(x)):
                    prob_y_given_X[k] *= self._get_gaussian_probability(
                        x[i], self.params[i][k][0], self.params[i][k][1])
            # The evidence p(x) is the same for every class, so the MAP class
            # is the argmax of the unnormalised posterior
            preds.append(self.labels[np.argmax(prob_y_given_X)])
        return np.array(preds)


def main(args) -> None:
    # Load data
    X_train, X_test, y_train, y_test = load_data(args.dataset)

    # Train and evaluate our model, then compare with sklearn
    evaluate(GaussianNaiveBayes(), X_train, X_test, y_train, y_test, "Our")
    evaluate(GaussianNB(), X_train, X_test, y_train, y_test, "Sklearn")


if __name__ == "__main__":
    main(get_parser(__doc__.splitlines()[0]).parse_args())

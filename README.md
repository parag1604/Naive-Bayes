# Gaussian Naive Bayes

Code for a Gaussian naive Bayes classifier written from scratch and checked against scikit-learn.

---

## Theory

Please note that the theory corresponding to the codes is hosted on this [page](https://parag1604.github.io/notes/Machine-Learning/Bayesian-Decision-Theory#naive-bayes) (section "Naive Bayes")

## Python Environment Setup

Use Python >= 3.10 and install the requirements:

```bash
pip install -r requirements.txt
```

- Numpy - required for vectorized operations
- Scipy - for the Gaussian density in version0
- Scikit-learn - for loading the datasets and for the reference GaussianNB

## Data

Both datasets ship with scikit-learn (`load_iris`, `load_digits`), so nothing needs to be downloaded. Select one with `--dataset iris` (default) or `--dataset digits`.

## Running

### 0. Naive Bayes from scratch

```bash
python version0.py
```

Estimates a class prior and a per-class, per-feature Gaussian, and predicts the class with the largest `p(y) * prod_j p(x_j | y)`. On iris it matches scikit-learn exactly:

```bash
$ python version0.py
Our train accuracy: 0.9500 | test accuracy: 1.0000
Sklearn train accuracy: 0.9500 | test accuracy: 1.0000
```

Now try `python version0.py --dataset digits`. Our accuracy drops to about 10% (chance level), while scikit-learn gets about 86%. Some pixels are constant within a class (zero variance) and the product of 64 densities underflows. Can you see why from the code?

### 1. Log-space and variance smoothing

```bash
python version1.py --dataset digits
```

New: the model sums log-densities instead of multiplying densities, adds a small epsilon to every variance (as scikit-learn's `var_smoothing` does), and is vectorized with broadcasting. It now matches scikit-learn on both datasets (test accuracy 1.000 on iris and 0.856 on digits).

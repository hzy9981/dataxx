#!/usr/bin/env python3
"""
Train a regression model using only training data from dataX.mat,
then evaluate on test data and report RMSE.

Requirements:
- Only use inputtrain/outputtrain for model fitting.
- Evaluate on inputtest/outputtest.
- Target RMSE < 1e-2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.compose import TransformedTargetRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))



def ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x



def ensure_1d(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        return y.ravel()
    if y.ndim == 2 and y.shape[0] == 1:
        return y.ravel()
    return y



def load_dataset(mat_path: Path):
    data = loadmat(mat_path)
    required = ["inputtrain", "outputtrain", "inputtest", "outputtest"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing keys in {mat_path}: {missing}")

    x_train = ensure_2d(data["inputtrain"])
    y_train = ensure_1d(data["outputtrain"])
    x_test = ensure_2d(data["inputtest"])
    y_test = ensure_1d(data["outputtest"])

    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"Training sample mismatch: x_train has {x_train.shape[0]}, y_train has {y_train.shape[0]}"
        )
    if x_test.shape[0] != y_test.shape[0]:
        raise ValueError(
            f"Test sample mismatch: x_test has {x_test.shape[0]}, y_test has {y_test.shape[0]}"
        )

    return x_train, y_train, x_test, y_test



def build_model(random_state: int = 42):
    # A flexible non-linear regressor that often fits smooth MATLAB-generated signals well.
    # Target scaling is included for numerical stability.
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=1.0, length_scale_bounds=(1e-4, 1e4))
        + WhiteKernel(noise_level=1e-8, noise_level_bounds=(1e-12, 1e-2))
    )

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-10,
        normalize_y=False,
        n_restarts_optimizer=8,
        random_state=random_state,
    )

    model = Pipeline(
        steps=[
            ("x_scaler", StandardScaler()),
            (
                "reg",
                TransformedTargetRegressor(
                    regressor=gpr,
                    transformer=StandardScaler(),
                ),
            ),
        ]
    )
    return model



def evaluate_cv(model, x_train: np.ndarray, y_train: np.ndarray, folds: int = 5) -> float:
    # CV is done only on training data, respecting the no-test-training requirement.
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    neg_mse_scores = cross_val_score(
        model,
        x_train,
        y_train,
        scoring="neg_mean_squared_error",
        cv=kf,
        n_jobs=-1,
    )
    cv_rmse = float(np.sqrt(-neg_mse_scores.mean()))
    return cv_rmse



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="dataX.mat", help="Path to dataX.mat")
    parser.add_argument(
        "--report",
        type=str,
        default="results.json",
        help="Output JSON report path",
    )
    parser.add_argument(
        "--save-pred",
        type=str,
        default="predictions_test.npy",
        help="Output NPY file for test predictions",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    x_train, y_train, x_test, y_test = load_dataset(data_path)

    model = build_model(random_state=42)

    cv_rmse = evaluate_cv(model, x_train, y_train, folds=5)

    # Fit strictly on training data.
    model.fit(x_train, y_train)

    # Predict on test data only for final evaluation.
    y_pred_test = model.predict(x_test)
    test_rmse = rmse(y_test, y_pred_test)

    np.save(args.save_pred, y_pred_test)

    report = {
        "data_file": str(data_path),
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "n_features": int(x_train.shape[1]),
        "cv_rmse_train_only": cv_rmse,
        "test_rmse": test_rmse,
        "target_threshold": 1e-2,
        "threshold_met": bool(test_rmse < 1e-2),
        "model": {
            "type": "Pipeline(StandardScaler + TransformedTargetRegressor(GaussianProcessRegressor))",
            "kernel": str(model.named_steps["reg"].regressor.kernel),
            "n_restarts_optimizer": 8,
            "random_state": 42,
        },
    }

    report_path = Path(args.report)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))

    if not report["threshold_met"]:
        raise SystemExit(
            f"RMSE target not met. test_rmse={test_rmse:.6e} >= 1e-2. "
            "You can increase n_restarts_optimizer or switch to a richer kernel."
        )



if __name__ == "__main__":
    main()

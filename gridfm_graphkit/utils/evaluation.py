"""Reconstruction-task evaluation on a test set.

Shared core for the eval script (`examples/evaluate_reconstruction.py`) and the
`Reconstruction_evaluation` notebook. Computes reconstruction metrics on the
**masked** targets (the actual learning signal), denormalized to physical units,
for two prediction modes:

- **point**: a single deterministic forward pass (dropout off).
- **probabilistic**: MC-dropout (`num_mc_samples` stochastic passes) -> per-bus
  predictive mean + std, plus calibration metrics (NLL, PICP, interval width).

The MC switch reuses the model's own `mc_dropout` flag (see `models/mc.py`):
`eval()` keeps BatchNorm on running stats while dropout stays active.
"""

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from gridfm_graphkit.datasets.globals import PD, QD, PG, QG, VM, VA, PQ, PV, REF

FEATURE_IDX = [PD, QD, PG, QG, VM, VA]
FEATURE_NAMES = ["PD", "QD", "PG", "QG", "VM", "VA"]
FEATURE_UNITS = ["MW", "MVar", "MW", "MVar", "p.u.", "degrees"]
BUS_NAMES = {0: "PQ", 1: "PV", 2: "REF"}
_EPS = 1e-6  # std floor so NLL/calibration stay finite when MC variance collapses


def _bus_type(x_np):
    """Per-node bus type code from the one-hot columns (0=PQ, 1=PV, 2=REF)."""
    bus = np.full(x_np.shape[0], -1, dtype=np.int64)
    bus[x_np[:, PQ] == 1] = 0
    bus[x_np[:, PV] == 1] = 1
    bus[x_np[:, REF] == 1] = 2
    return bus


@torch.no_grad()
def run_inference(task, loader, normalizer, device="cpu", num_mc_samples=0):
    """Run the task over a test loader, return denormalized per-node arrays.

    Returns a dict of np arrays aligned by node:
        y_true, y_point  : [N, 6] ground truth / deterministic prediction
        mask             : [N, 6] bool, which entries were reconstructed
        bus_type         : [N]    0/1/2 = PQ/PV/REF
        y_mc_mean, y_mc_std : [N, 6] present only if num_mc_samples >= 2

    `num_mc_samples` enables the probabilistic pass. Requires a model exposing
    the `mc_dropout` flag (GPSTransformer / GNN_TransformerConv do).
    """
    task = task.to(device).eval()
    do_mc = num_mc_samples and num_mc_samples >= 2
    if do_mc and not hasattr(task.model, "mc_dropout"):
        raise ValueError("model has no mc_dropout flag; cannot do MC-dropout")

    ys, points, masks, buses, mc_means, mc_stds = [], [], [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        fwd = dict(
            x=batch.x,
            pe=batch.pe,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            batch=batch.batch,
            mask=batch.mask,
        )
        # Deterministic point prediction. forward() fills masked inputs with the
        # model's mask_value in place; later MC passes re-fill identically.
        out = task(**fwd)
        points.append(normalizer.inverse_transform(out.clone()).cpu().numpy())

        if do_mc:
            task.model.mc_dropout = True  # dropout active, BatchNorm stays eval
            try:
                samples = torch.stack(
                    [normalizer.inverse_transform(task(**fwd).clone()) for _ in range(num_mc_samples)]
                )  # [S, N, 6]
            finally:
                task.model.mc_dropout = False
            mc_means.append(samples.mean(0).cpu().numpy())
            mc_stds.append(samples.std(0).cpu().numpy())

        ys.append(normalizer.inverse_transform(batch.y.clone()).cpu().numpy())
        masks.append(batch.mask.cpu().numpy().astype(bool))
        buses.append(_bus_type(batch.x.cpu().numpy()))

    arrays = {
        "y_true": np.concatenate(ys),
        "y_point": np.concatenate(points),
        "mask": np.concatenate(masks),
        "bus_type": np.concatenate(buses),
    }
    if do_mc:
        arrays["y_mc_mean"] = np.concatenate(mc_means)
        arrays["y_mc_std"] = np.concatenate(mc_stds)
    return arrays


def _select(arrays, f, masked_only, bus=None):
    m = arrays["mask"][:, f] if masked_only else np.ones(len(arrays["mask"]), bool)
    if bus is not None:
        m = m & (arrays["bus_type"] == bus)
    return m


def _rmse(err):
    return float(np.sqrt(np.mean(err**2))) if len(err) else np.nan


def point_metrics(arrays, masked_only=True):
    """Per-feature RMSE / MAE of the deterministic prediction, split by bus type.

    `masked_only=True` scores only reconstructed entries (the task); False scores
    all nodes (matches the model's `test_step`, which also keeps known inputs).
    """
    rows = []
    for f, name, unit in zip(FEATURE_IDX, FEATURE_NAMES, FEATURE_UNITS):
        m = _select(arrays, f, masked_only)
        err = arrays["y_point"][m, f] - arrays["y_true"][m, f]
        row = {
            "feature": name,
            "unit": unit,
            "n": int(m.sum()),
            "RMSE": _rmse(err),
            "MAE": float(np.mean(np.abs(err))) if m.any() else np.nan,
        }
        for b, bname in BUS_NAMES.items():
            mb = _select(arrays, f, masked_only, b)
            row[f"RMSE_{bname}"] = _rmse(arrays["y_point"][mb, f] - arrays["y_true"][mb, f])
        rows.append(row)
    return pd.DataFrame(rows).set_index("feature")


def prob_metrics(arrays, masked_only=True, alpha=0.05):
    """Per-feature probabilistic metrics from the MC-dropout mean + std.

    RMSE/MAE on the MC mean; Gaussian NLL; PICP (empirical coverage of the
    (1-alpha) interval, target = 1-alpha); MPIW (mean interval width = sharpness
    of the interval); `sharpness` (mean predictive std).
    """
    if "y_mc_std" not in arrays:
        raise ValueError("no MC outputs in arrays; run_inference with num_mc_samples >= 2")
    z = norm.ppf(1 - alpha / 2)
    picp_col = f"PICP@{round((1 - alpha) * 100)}%"
    rows = []
    for f, name, unit in zip(FEATURE_IDX, FEATURE_NAMES, FEATURE_UNITS):
        m = _select(arrays, f, masked_only)
        yt = arrays["y_true"][m, f]
        mu = arrays["y_mc_mean"][m, f]
        sd = np.maximum(arrays["y_mc_std"][m, f], _EPS)
        err = mu - yt
        if m.any():
            nll = float(np.mean(0.5 * np.log(2 * np.pi * sd**2) + 0.5 * (err**2) / sd**2))
            picp = float(np.mean(np.abs(err) <= z * sd))
            mpiw = float(np.mean(2 * z * sd))
            sharp = float(np.mean(sd))
            mae = float(np.mean(np.abs(err)))
        else:
            nll = picp = mpiw = sharp = mae = np.nan
        rows.append(
            {
                "feature": name,
                "unit": unit,
                "n": int(m.sum()),
                "RMSE": _rmse(err),
                "MAE": mae,
                "NLL": nll,
                picp_col: picp,
                "MPIW": mpiw,
                "sharpness": sharp,
            }
        )
    return pd.DataFrame(rows).set_index("feature")


def calibration_curve(arrays, masked_only=True, levels=None):
    """Reliability data: for each feature, observed coverage vs nominal level.

    A well-calibrated model lies on the diagonal. Returns {feature: (nominal, observed)}.
    """
    if "y_mc_std" not in arrays:
        raise ValueError("no MC outputs in arrays; run_inference with num_mc_samples >= 2")
    if levels is None:
        levels = np.linspace(0.05, 0.95, 19)
    out = {}
    for f, name in zip(FEATURE_IDX, FEATURE_NAMES):
        m = _select(arrays, f, masked_only)
        absz = np.abs(arrays["y_true"][m, f] - arrays["y_mc_mean"][m, f]) / np.maximum(
            arrays["y_mc_std"][m, f], _EPS
        )
        observed = np.array([np.mean(absz <= norm.ppf(0.5 + lv / 2)) for lv in levels])
        out[name] = (np.asarray(levels), observed)
    return out


# --- plotting (matplotlib only; imported lazily so metrics work headless) -----


def plot_pred_vs_true(arrays, which="point", masked_only=True):
    """2x3 grid of predicted-vs-true scatter, one panel per feature."""
    import matplotlib.pyplot as plt

    key = {"point": "y_point", "mc": "y_mc_mean"}[which]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, f, name, unit in zip(axes.flat, FEATURE_IDX, FEATURE_NAMES, FEATURE_UNITS):
        m = _select(arrays, f, masked_only)
        yt, yp = arrays["y_true"][m, f], arrays[key][m, f]
        ax.scatter(yt, yp, s=8, alpha=0.4)
        lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(f"{name} ({unit})  RMSE={_rmse(yp - yt):.3g}")
        ax.set_xlabel("true")
        ax.set_ylabel(f"pred ({which})")
    fig.suptitle(f"Predicted vs true ({which}, masked entries)", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_calibration(arrays, masked_only=True):
    """Reliability diagram: observed vs nominal coverage, one line per feature."""
    import matplotlib.pyplot as plt

    cal = calibration_curve(arrays, masked_only=masked_only)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    for name, (nominal, observed) in cal.items():
        ax.plot(nominal, observed, marker="o", ms=3, label=name)
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("observed coverage")
    ax.set_title("Calibration (MC-dropout)", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_error_vs_uncertainty(arrays, masked_only=True):
    """2x3 grid of |error| vs predictive std; diagonal = perfect 1:1 scaling."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, f, name, unit in zip(axes.flat, FEATURE_IDX, FEATURE_NAMES, FEATURE_UNITS):
        m = _select(arrays, f, masked_only)
        sd = arrays["y_mc_std"][m, f]
        abserr = np.abs(arrays["y_mc_mean"][m, f] - arrays["y_true"][m, f])
        ax.scatter(sd, abserr, s=8, alpha=0.4)
        hi = float(max(sd.max(), abserr.max()))
        ax.plot([0, hi], [0, hi], "k--", lw=1)
        ax.set_title(f"{name} ({unit})")
        ax.set_xlabel("predictive std")
        ax.set_ylabel("|error|")
    fig.suptitle("Error vs uncertainty (masked entries)", fontweight="bold")
    fig.tight_layout()
    return fig


def _demo():
    """Self-check on synthetic data: needs no model/dataset.

    Falsifiable: perfect predictions -> RMSE 0; a correctly-specified Gaussian
    -> PICP@95 ~ 0.95 and RMSE ~ injected sigma.
    """
    rng = np.random.default_rng(0)
    n = 40000
    y = rng.normal(size=(n, 6))
    arrays = {
        "y_true": y,
        "y_point": y.copy(),
        "mask": np.ones((n, 6), bool),
        "bus_type": rng.integers(0, 3, n),
    }
    assert point_metrics(arrays)["RMSE"].max() < 1e-9, "perfect point pred must give RMSE 0"

    sigma = 0.3
    arrays["y_mc_mean"] = y + rng.normal(scale=sigma, size=(n, 6))
    arrays["y_mc_std"] = np.full((n, 6), sigma)
    pm = prob_metrics(arrays)
    assert abs(pm["PICP@95%"].mean() - 0.95) < 0.02, "calibrated Gaussian must cover ~95%"
    assert abs(pm["RMSE"].mean() - sigma) < 0.02, "RMSE must track injected sigma"

    cal = calibration_curve(arrays)
    nominal, observed = cal["PD"]
    assert np.max(np.abs(nominal - observed)) < 0.03, "calibration curve must sit on diagonal"
    print("evaluation.py self-check passed")


if __name__ == "__main__":
    _demo()

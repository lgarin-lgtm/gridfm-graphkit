#!/usr/bin/env python
"""Evaluate a trained GridFM model on the reconstruction task over a test set.

Point metrics always; probabilistic (MC-dropout) metrics when --num_mc_samples >= 2.
Writes per-feature metric CSVs (and optional plots) to --output_dir.

Example:
    python examples/evaluate_reconstruction.py \
        --config examples/config/case30_ieee_base.yaml \
        --data_path examples/data \
        --model_path examples/models/GridFM_v0_2.pth \
        --output_dir eval_out --num_mc_samples 20 --plots

Note: defaults to --device cpu. On Apple Silicon, MPS lacks the scatter op
torch_geometric needs, so cpu is the working path here.
"""

import argparse
import os
import random

import numpy as np
import torch
import yaml

from gridfm_graphkit.datasets.powergrid_datamodule import LitGridDataModule
from gridfm_graphkit.io.param_handler import NestedNamespace
from gridfm_graphkit.tasks.feature_reconstruction_task import FeatureReconstructionTask
from gridfm_graphkit.utils import evaluation as ev


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_dir", default="eval_out")
    p.add_argument("--network_idx", type=int, default=0, help="which test network (multi-net configs)")
    p.add_argument("--num_mc_samples", type=int, default=20, help="MC-dropout passes; <2 disables probabilistic")
    p.add_argument("--all_entries", action="store_true", help="score all nodes, not only masked targets")
    p.add_argument("--device", default="cpu")
    p.add_argument("--plots", action="store_true", help="also save figures (needs MC for calibration plots)")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = NestedNamespace(**yaml.safe_load(f))
    cfg.data.workers = 0  # single-process loading: avoids spawn issues, fine for a test set
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    dm = LitGridDataModule(cfg, args.data_path)
    dm.setup("test")
    loader = dm.test_dataloader()[args.network_idx]
    normalizer = dm.node_normalizers[args.network_idx]
    network = cfg.data.networks[args.network_idx]

    task = FeatureReconstructionTask(cfg, dm.node_normalizers, dm.edge_normalizers)
    task.load_state_dict(torch.load(args.model_path, map_location=args.device))

    masked_only = not args.all_entries
    print(f"-> Inference on '{network}' test set (device={args.device}, mc={args.num_mc_samples})...")
    arrays = ev.run_inference(
        task, loader, normalizer, device=args.device, num_mc_samples=args.num_mc_samples
    )

    os.makedirs(args.output_dir, exist_ok=True)
    pm = ev.point_metrics(arrays, masked_only=masked_only)
    pm.to_csv(os.path.join(args.output_dir, f"{network}_point_metrics.csv"))
    print("\n=== Point predictions ===")
    print(pm.to_string(float_format=lambda v: f"{v:.4g}"))

    have_mc = "y_mc_std" in arrays
    if have_mc:
        prm = ev.prob_metrics(arrays, masked_only=masked_only)
        prm.to_csv(os.path.join(args.output_dir, f"{network}_prob_metrics.csv"))
        print("\n=== Probabilistic predictions (MC-dropout) ===")
        print(prm.to_string(float_format=lambda v: f"{v:.4g}"))

    if args.plots:
        ev.plot_pred_vs_true(arrays, "point", masked_only).savefig(
            os.path.join(args.output_dir, f"{network}_pred_vs_true_point.png"), bbox_inches="tight"
        )
        if have_mc:
            ev.plot_calibration(arrays, masked_only).savefig(
                os.path.join(args.output_dir, f"{network}_calibration.png"), bbox_inches="tight"
            )
            ev.plot_error_vs_uncertainty(arrays, masked_only).savefig(
                os.path.join(args.output_dir, f"{network}_error_vs_uncertainty.png"), bbox_inches="tight"
            )

    print(f"\nSaved metrics{' + plots' if args.plots else ''} to {args.output_dir}/")


if __name__ == "__main__":
    main()

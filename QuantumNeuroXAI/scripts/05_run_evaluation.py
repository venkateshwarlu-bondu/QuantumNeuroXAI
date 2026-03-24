from __future__ import annotations
import argparse
import os
import torch
from src.utils.seed import set_global_seed
from src.utils.device import get_device
from src.utils.io import load_yaml, save_json
from src.models.baseline_model import BaselineEEGNet
from src.models.quantum_neuro_xai import QuantumNeuroXAI
from src.training.evaluator import evaluate_model, export_confusion_matrix, export_roc_curve
from scripts.common import build_loaders, infer_num_classes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True, choices=["tuh", "chbmit", "bci2a"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--model-type", default="quantum", choices=["quantum", "baseline"])
    args = parser.parse_args()

    gcfg = load_yaml(args.config)
    dcfg = load_yaml(args.dataset_config)
    mcfg = load_yaml(args.model_config)

    set_global_seed(gcfg["seed"])
    device = get_device(gcfg["device"])
    _, _, test_loader = build_loaders(dcfg["processed_manifest_csv"], gcfg["train"]["batch_size"], gcfg["num_workers"])

    task_mode = dcfg["task"]["mode"]
    num_classes = infer_num_classes(dcfg)

    if args.model_type == "baseline":
        model = BaselineEEGNet(mcfg, task_mode=task_mode, num_classes=num_classes)
    else:
        model = QuantumNeuroXAI(mcfg, task_mode=task_mode, num_classes=num_classes)

    # lazy-build by a single forward pass
    first_batch = next(iter(test_loader))
    _ = model(first_batch["x"].to(device))
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device)

    result = evaluate_model(model, test_loader, task_mode, device)
    save_json(result["metrics"], f"{gcfg['metrics_dir']}/test_metrics.json")
    export_confusion_matrix(result["y_true"], result["y_pred"], f"{gcfg['plots_dir']}/confusion_matrix.png")
    if task_mode == "binary":
        export_roc_curve(result["y_true"], result["y_prob"], f"{gcfg['plots_dir']}/roc_curve.png")
    print(result["metrics"])

if __name__ == "__main__":
    main()

from __future__ import annotations
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from src.utils.seed import set_global_seed
from src.utils.device import get_device
from src.utils.io import load_yaml
from src.models.quantum_neuro_xai import QuantumNeuroXAI
from src.explainability.signal_xai import input_saliency, integrated_gradients_signal, channel_band_summary
from src.explainability.model_xai import extract_attention_weights, fused_feature_attribution
from src.explainability.quantum_xai import quantum_sensitivity_analysis, rank_quantum_dimensions
from src.explainability.report_builder import build_unified_report, save_explanation_report
from scripts.common import build_loaders, infer_num_classes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True, choices=["tuh", "chbmit", "bci2a"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--xai-config", required=True)
    args = parser.parse_args()

    gcfg = load_yaml(args.config)
    dcfg = load_yaml(args.dataset_config)
    mcfg = load_yaml(args.model_config)
    xcfg = load_yaml(args.xai_config)

    set_global_seed(gcfg["seed"])
    device = get_device(gcfg["device"])
    _, _, test_loader = build_loaders(dcfg["processed_manifest_csv"], batch_size=1, num_workers=0)

    model = QuantumNeuroXAI(mcfg, task_mode=dcfg["task"]["mode"], num_classes=infer_num_classes(dcfg))
    first_batch = next(iter(test_loader))
    _ = model(first_batch["x"].to(device))
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device)
    model.eval()

    os.makedirs(gcfg["xai_dir"], exist_ok=True)

    for idx, batch in enumerate(test_loader):
        if idx >= xcfg["num_samples"]:
            break
        x = batch["x"].to(device)
        y = batch["y"].cpu().numpy().tolist()
        out = model(x)

        sal = input_saliency(model, x)
        sig_summary = channel_band_summary(sal)

        attn = extract_attention_weights(out)
        fused_attr = fused_feature_attribution(out)
        q_scores = quantum_sensitivity_analysis(model, x, eps=xcfg["quantum_eps"])
        q_rank = rank_quantum_dimensions(q_scores)

        report = build_unified_report(
            signal_summary=sig_summary,
            attention=attn,
            fused_attr=fused_attr,
            quantum_rank=q_rank,
            prediction={"target": y, "probs": out["probs"].detach().cpu().numpy().tolist()},
        )
        save_explanation_report(report, os.path.join(gcfg["xai_dir"], f"sample_{idx:03d}.json"))

        sal_mean = sal.mean(axis=(0, 1))
        plt.figure(figsize=(6, 4))
        plt.imshow(sal_mean.mean(axis=0), aspect="auto")
        plt.title(f"Signal Saliency #{idx}")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(gcfg["xai_dir"], f"sample_{idx:03d}_saliency.png"), dpi=300)
        plt.close()

        if attn is not None:
            plt.figure(figsize=(6, 3))
            plt.plot(attn[0])
            plt.title(f"Attention Weights #{idx}")
            plt.tight_layout()
            plt.savefig(os.path.join(gcfg["xai_dir"], f"sample_{idx:03d}_attention.png"), dpi=300)
            plt.close()

        if q_scores.size > 0:
            plt.figure(figsize=(6, 3))
            plt.bar(np.arange(len(q_scores)), q_scores)
            plt.title(f"Quantum Sensitivity #{idx}")
            plt.tight_layout()
            plt.savefig(os.path.join(gcfg["xai_dir"], f"sample_{idx:03d}_quantum.png"), dpi=300)
            plt.close()

    print(f"Saved explanations to {gcfg['xai_dir']}")

if __name__ == "__main__":
    main()

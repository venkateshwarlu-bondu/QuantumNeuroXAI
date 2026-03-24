from __future__ import annotations
import argparse
from src.utils.seed import set_global_seed
from src.utils.device import get_device
from src.utils.io import load_yaml
from src.models.baseline_model import BaselineEEGNet
from src.training.trainer import Trainer, TrainerConfig
from scripts.common import build_loaders, infer_num_classes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["tuh", "chbmit", "bci2a"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--model-config", required=True)
    args = parser.parse_args()

    gcfg = load_yaml(args.config)
    dcfg = load_yaml(args.dataset_config)
    mcfg = load_yaml(args.model_config)

    set_global_seed(gcfg["seed"])
    device = get_device(gcfg["device"])
    train_loader, val_loader, _ = build_loaders(dcfg["processed_manifest_csv"], gcfg["train"]["batch_size"], gcfg["num_workers"])

    task_mode = dcfg["task"]["mode"]
    num_classes = infer_num_classes(dcfg)
    model = BaselineEEGNet(mcfg, task_mode=task_mode, num_classes=num_classes)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        cfg=TrainerConfig(
            epochs=gcfg["train"]["epochs"],
            learning_rate=gcfg["train"]["learning_rate"],
            weight_decay=gcfg["train"]["weight_decay"],
            patience=gcfg["train"]["patience"],
            scheduler_factor=gcfg["train"]["scheduler_factor"],
            scheduler_patience=gcfg["train"]["scheduler_patience"],
            checkpoint_path=f"{gcfg['checkpoint_dir']}/baseline_best.pt",
            metrics_path=f"{gcfg['metrics_dir']}/baseline_train_metrics.json",
            task_mode=task_mode,
        ),
    )
    result = trainer.fit()
    print(result)

if __name__ == "__main__":
    main()

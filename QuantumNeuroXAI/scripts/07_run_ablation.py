from __future__ import annotations
import json
from src.training.ablations import list_ablation_variants

if __name__ == "__main__":
    print(json.dumps(list_ablation_variants(), indent=2))

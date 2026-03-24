from __future__ import annotations
import argparse
import os
import numpy as np

def make_signal(label: int, channels: int = 19, samples: int = 4096):
    t = np.linspace(0, 8, samples, endpoint=False)
    x = []
    for c in range(channels):
        base = np.sin(2 * np.pi * (4 + c % 4) * t) + 0.2 * np.random.randn(samples)
        if label == 1:
            base += 0.8 * np.sin(2 * np.pi * 12 * t)
        x.append(base)
    return np.asarray(x, dtype=np.float32)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="chbmit", choices=["chbmit", "bci2a", "tuh"])
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    root = os.path.join("data", "raw", args.dataset if args.dataset != "tuh" else "tuh_eeg")
    os.makedirs(root, exist_ok=True)
    for i in range(args.n):
        label = i % 2
        name = f"subject_{i:03d}_{'seiz' if label else 'normal'}.npy"
        np.save(os.path.join(root, name), make_signal(label))
    print(f"Created {args.n} synthetic files in {root}")

import argparse
from pathlib import Path

import torch

from protocol import calibrate

ROOT = Path(__file__).parent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("moshi", "personaplex"))
    args = parser.parse_args()
    if args.model == "moshi":
        from moshi_probe import Probe

        repo = "kyutai/moshika-pytorch-bf16"
    else:
        from personaplex_probe import Probe

        repo = "nvidia/personaplex-7b-v1"
    with torch.inference_mode():
        calibrate(
            Probe(repo, "cuda"),
            ROOT / "speech.wav",
            ROOT / "runs" / f"{args.model}-calibration",
            40,
            250,
        )

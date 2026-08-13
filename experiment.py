import argparse
import json
from pathlib import Path

import torch

from protocol import run

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
    calibration = json.loads(
        (ROOT / "runs" / f"{args.model}-calibration" / "results.json").read_text()
    )
    with torch.inference_mode():
        run(
            Probe(repo, "cuda"),
            Probe(repo, "cuda"),
            ROOT / "speech.wav",
            ROOT / "microphone.wav",
            ROOT / "runs" / args.model,
            40,
            10000,
            3750,
            1000,
            calibration["gap_frames"],
        )

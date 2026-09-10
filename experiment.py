import argparse
import json
from pathlib import Path

import torch

from protocol import run

ROOT = Path(__file__).parent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("moshi", "personaplex"))
    parser.add_argument(
        "--extend",
        type=int,
        metavar="TRIALS",
        help="extend mitigation trials, reusing released calibration and original 40",
    )
    args = parser.parse_args()
    output = ROOT / "runs" / args.model
    if args.extend is not None:
        from extend import GROUPS, extend, load_results

        result = load_results(args.model, output, args.extend)
        if all(len(result[name]) == args.extend for name in GROUPS):
            print(f"Already have {args.extend} held-out mitigation trials")
            raise SystemExit(0)
    if args.model == "moshi":
        from moshi_probe import Probe

        repo = "kyutai/moshika-pytorch-bf16"
    else:
        from personaplex_probe import Probe

        repo = "nvidia/personaplex-7b-v1"
    with torch.inference_mode():
        if args.extend is not None:
            extend(
                Probe(repo, "cuda"),
                Probe(repo, "cuda"),
                ROOT / "speech.wav",
                ROOT / "microphone.wav",
                output,
                result,
                args.extend,
            )
            raise SystemExit(0)
        calibration = json.loads(
            (ROOT / "runs" / f"{args.model}-calibration" / "results.json").read_text()
        )
        run(
            Probe(repo, "cuda"),
            Probe(repo, "cuda"),
            ROOT / "speech.wav",
            ROOT / "microphone.wav",
            output,
            40,
            10000,
            3750,
            1000,
            calibration["gap_frames"],
        )

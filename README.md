# Causal Analysis and Mitigation of Spurious Onsets in Full-Duplex Speech LLMs

Code for reproducing our Moshi and PersonaPlex experiments.

## Setup

Linux, CUDA, two NVIDIA GPUs, and Hugging Face access to `nvidia/personaplex-7b-v1` are required.

```bash
./build
./fetch-noise
```

## Experiments

```bash
./calibrate
./run
```

`calibrate` generates 40 responses per model and derives each onset qualification boundary from the longest nonlexical gap. `run` performs the silence, counterfactual, threshold, and runtime experiments, with 40 trials per model by default. Raw results are written to `runs/`; aggregate results are written to `results.json`.

For full paper replication, follow with `./run --extend 500` to reach 500 held-out mitigation trials per model, keeping calibration and thresholds fixed. Alternatively, run it directly after setup to reuse the [released calibration and 40-trial results](https://github.com/KentoNishi/icassp27-spurious-onsets/releases/tag/results-2026-08-13). Updated results are written to `runs/moshi/results.json` and `runs/personaplex/results.json`.

```bash
./.venv/moshi/bin/python plot_results.py runs/moshi/results.json runs/personaplex/results.json
```

The figures are written to `figures/`.

Paths, GPUs, and the PyTorch wheel index are defined in `config`.

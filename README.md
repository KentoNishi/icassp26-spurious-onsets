# Causal Analysis and Mitigation of Spurious Speech Onsets in Full-Duplex Speech LLMs

Code for reproducing the paper's Moshi and PersonaPlex experiments.

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

`calibrate` generates 40 responses per model and derives each onset boundary from the longest nonlexical gap. `run` performs the silence, counterfactual, threshold, and runtime experiments. Raw results are written to `runs/`; aggregate results are written to `results.json`.

Paths, GPUs, and the PyTorch wheel index are defined in `config`.

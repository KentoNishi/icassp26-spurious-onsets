import hashlib
import io
import json
import tarfile
import urllib.request

from protocol import (
    FRAME_RATE,
    NOISE_DBFS,
    SESSION_FRAMES,
    load_audio,
    load_noise,
    mix,
    paired_gate,
    response,
    save,
    threshold_results,
)

RELEASE_URL = (
    "https://github.com/KentoNishi/icassp27-spurious-onsets/releases/download/"
    "results-2026-08-13/paper-results-20260813.tar.gz"
)
RELEASE_SHA256 = "2107e4a93e34973dadb0ba214b65b509c3f8dececfeefd3f51975fdb2c23519b"
GROUPS = ("microphone_rollouts", "microphone_responses")


def load_results(model, output, trials):
    """Resume existing measurements, or seed them from the paper's release."""
    if trials < 40:
        raise ValueError("extension requires at least the original 40 trials")
    path = output / "results.json"
    if path.exists():
        result = json.loads(path.read_text())
    else:
        print("Loading released calibration and original 40 trials", flush=True)
        with urllib.request.urlopen(RELEASE_URL, timeout=60) as source:
            data = source.read()
        if hashlib.sha256(data).hexdigest() != RELEASE_SHA256:
            raise ValueError("released results checksum mismatch")
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            result = json.load(archive.extractfile(f"runs/{model}/results.json"))
    for name in ("calibration_idle", "calibration_responses"):
        rows = result[name]
        if len(rows) != 40 or {row["seed"] for row in rows} != set(range(40)):
            raise ValueError("extension requires the original 40 calibration trials")
    for name in GROUPS:
        rows = result[name]
        seeds = {row["seed"] for row in rows}
        if (
            len(seeds) != len(rows)
            or not set(range(10000, 10040)) <= seeds
            or not seeds <= set(range(10000, 10000 + trials))
        ):
            raise ValueError(
                f"{name}: duplicate, missing original, or out-of-range seeds"
            )
    if not path.exists():
        output.mkdir(parents=True, exist_ok=True)
        save(output, result)
    return result


def extend(observed, shadow, audio_path, noise_path, output, result, trials):
    """Append held-out trials without recalibration, baseline, or runtime runs."""
    if observed.metadata != result["model"] or shadow.metadata != result["model"]:
        raise ValueError("model settings differ from saved results")
    if (
        result["frame_rate"] != FRAME_RATE
        or result["session_s"] * FRAME_RATE != SESSION_FRAMES
        or result["noise_dbfs"] != NOISE_DBFS
    ):
        raise ValueError("experimental settings differ from saved results")
    prompt = load_audio(audio_path, observed)
    if len(prompt) != result["prompt_frames"]:
        raise ValueError("request length differs from saved results")
    noise = load_noise(noise_path, observed, SESSION_FRAMES)
    noisy_prompt = mix(prompt, noise)
    gap_frames = result["gap_frames"]
    idle_seeds = {row["seed"] for row in result[GROUPS[0]]}
    response_seeds = {row["seed"] for row in result[GROUPS[1]]}
    for run_seed in range(10000, 10000 + trials):
        if run_seed not in idle_seeds:
            result[GROUPS[0]].append(
                paired_gate(
                    observed,
                    shadow,
                    noisy_prompt,
                    gap_frames,
                    run_seed,
                    noise,
                    noise,
                    SESSION_FRAMES,
                )
            )
            result["microphone"] = threshold_results(
                result[GROUPS[0]], result["boundaries"]
            )
            save(output, result)
        if run_seed not in response_seeds:
            result[GROUPS[1]].append(
                response(
                    observed,
                    shadow,
                    noisy_prompt,
                    run_seed,
                    noisy_prompt,
                    [],
                    gap_frames,
                    speech_frames=len(prompt),
                )
            )
            save(output, result)
    print(f"Completed {trials} held-out mitigation trials", flush=True)

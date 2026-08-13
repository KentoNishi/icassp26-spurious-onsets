import json
import math
import statistics
from pathlib import Path


def binomial_cdf(k, n, probability):
    return sum(
        math.comb(n, index) * probability**index * (1 - probability) ** (n - index)
        for index in range(k + 1)
    )


def inverse(function, target):
    low, high = 0.0, 1.0
    for _ in range(80):
        middle = (low + high) / 2
        if function(middle) > target:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def interval(successes, trials, alpha=0.05):
    low = 0.0
    high = 1.0
    if successes:
        low = inverse(
            lambda probability: binomial_cdf(successes - 1, trials, probability),
            1 - alpha / 2,
        )
    if successes < trials:
        high = inverse(
            lambda probability: binomial_cdf(successes, trials, probability),
            alpha / 2,
        )
    return [low, high]


def orders(values):
    values = [value for value in values if value > 0]
    return math.log10(max(values) / min(values)) if len(values) > 1 else 0.0


def validate(result, trials, measurements):
    if len(result["baseline"]) != trials:
        raise ValueError("baseline is incomplete")
    if len(result["calibration_idle"]) != trials:
        raise ValueError("idle calibration is incomplete")
    if len(result["calibration_responses"]) != trials:
        raise ValueError("response calibration is incomplete")
    if len(result["microphone_rollouts"]) != trials:
        raise ValueError("conditional rollouts are incomplete")
    if len(result["microphone_responses"]) != trials:
        raise ValueError("response evaluation is incomplete")
    if set(result["microphone"]) != {"response", "balanced", "suppression"}:
        raise ValueError("threshold evaluations are incomplete")
    if any(len(runs) != trials for runs in result["microphone"].values()):
        raise ValueError("microphone evaluation is incomplete")
    if len(result.get("runtime", {}).get("times_ms", [])) != measurements:
        raise ValueError("runtime benchmark is incomplete")
    seeds = {run["seed"] for run in result["baseline"]}
    groups = [
        result["microphone_rollouts"],
        result["microphone_responses"],
        *result["microphone"].values(),
    ]
    if len(seeds) != trials or any(
        {run["seed"] for run in group} != seeds for group in groups
    ):
        raise ValueError("held-out seeds are incomplete")


def summarize(result, trials=40, measurements=1000):
    validate(result, trials, measurements)
    rate = result["frame_rate"]
    baseline = result["baseline"]
    onset_frames = [
        run["onset_frame"] for run in baseline if run["onset_frame"] is not None
    ]
    onset_times = [frame / rate for frame in onset_frames]
    within = max(orders(run["onset_probabilities"]) for run in baseline)
    across = max(
        orders(
            run["onset_probabilities"][frame]
            for run in baseline
            if frame < len(run["onset_probabilities"])
        )
        for frame in range(
            max(map(len, (run["onset_probabilities"] for run in baseline)))
        )
    )
    thresholds = {}
    names = {"response": "preserve", "balanced": "balanced", "suppression": "suppress"}
    response_scores = [run["score"] for run in result["microphone_responses"]]
    for key, boundary in result["boundaries"].items():
        thresholds[names[key]] = {
            "threshold": boundary,
            "first_onsets": sum(run["onset"] for run in result["microphone"][key]),
            "first_onsets_suppressed": sum(
                run["suppressed"] for run in result["microphone"][key]
            ),
            "response_onsets_preserved": sum(
                score is not None and score > boundary for score in response_scores
            ),
        }
    return {
        "model": result["model"],
        "trials": trials,
        "session_s": result["session_s"],
        "gap_frames": result["gap_frames"],
        "noise_dbfs": result["noise_dbfs"],
        "baseline": {
            "onsets": len(onset_times),
            "incidence": len(onset_times) / trials,
            "confidence_interval_95": interval(len(onset_times), trials),
            "onset_s": {
                "minimum": min(onset_times, default=None),
                "median": statistics.median(onset_times) if onset_times else None,
                "maximum": max(onset_times, default=None),
            },
            "maximum_orders_within_continuation": within,
            "maximum_orders_across_continuations": across,
        },
        "thresholds": thresholds,
        "runtime": {
            "measurements": measurements,
            "median_ms": result["runtime"]["median_ms"],
            "p95_ms": result["runtime"]["p95_ms"],
        },
    }


if __name__ == "__main__":
    root = Path(__file__).parent
    paths = [
        root / "runs" / model / "results.json" for model in ("moshi", "personaplex")
    ]
    summaries = [summarize(json.loads(path.read_text())) for path in paths]
    (root / "results.json").write_text(json.dumps(summaries, indent=2) + "\n")

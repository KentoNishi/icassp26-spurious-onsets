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


def onset_statistics(baseline):
    """Audit each observed onset, excluding its frame from cumulative probability.

    Baseline traces stop at the first qualified onset. Require one probability
    per frame so a missing sample cannot masquerade as a single-frame jump.
    A positive onset after an exactly zero probability has an infinite jump;
    represent that case explicitly, rather than emitting nonstandard JSON Infinity.
    """
    onsets = []
    for run in baseline:
        probabilities = run["onset_probabilities"]
        if not probabilities or any(
            not math.isfinite(value) or not 0 <= value <= 1 + 1e-6
            for value in probabilities
        ):
            raise ValueError("baseline probabilities must be nonempty and in [0, 1]")
        # Summing the float32 sampling distribution can exceed one by roundoff.
        # Bound that numerical error without changing the saved raw trace.
        probabilities = [min(value, 1.0) for value in probabilities]
        frame = run["onset_frame"]
        if frame is None:
            continue
        if not isinstance(frame, int) or frame < 0 or len(probabilities) != frame + 1:
            raise ValueError("onset trace must contain every frame through the onset")
        before = probabilities[:-1]
        probability = probabilities[-1]
        if probability == 0:
            raise ValueError("an observed lexical onset must have positive probability")
        previous = before[-1] if before else None
        cumulative = (
            1.0
            if 1.0 in before
            else -math.expm1(math.fsum(math.log1p(-value) for value in before))
        )
        onsets.append(
            {
                "seed": run["seed"],
                "onset_frame": frame,
                "preceding_probability": previous,
                "onset_probability": probability,
                "log10_jump": (
                    math.log10(probability) - math.log10(previous)
                    if previous is not None and previous > 0
                    else None
                ),
                "zero_preceding_probability": previous == 0,
                "cumulative_probability_before_onset": cumulative,
            }
        )
    previous = [
        row["preceding_probability"]
        for row in onsets
        if row["preceding_probability"] is not None
    ]
    jumps = [row["log10_jump"] for row in onsets if row["log10_jump"] is not None]
    return {
        "onsets": onsets,
        "maximum_preceding_probability": max(previous, default=None),
        "minimum_onset_probability": min(
            (row["onset_probability"] for row in onsets), default=None
        ),
        "minimum_log10_jump": min(jumps, default=None),
        "onsets_with_zero_preceding_probability": sum(
            row["zero_preceding_probability"] for row in onsets
        ),
        "onsets_without_preceding_frame": sum(
            row["preceding_probability"] is None for row in onsets
        ),
        "maximum_cumulative_probability_before_onset": max(
            (row["cumulative_probability_before_onset"] for row in onsets),
            default=None,
        ),
    }


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
    onset_audit = onset_statistics(baseline)
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
            "onset_statistics": onset_audit,
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

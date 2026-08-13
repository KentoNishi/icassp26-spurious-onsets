import json
import random
import time
import wave

import numpy as np
import torch

FRAME_RATE = 12.5
SESSION_FRAMES = int(300 * FRAME_RATE)
NOISE_DBFS = -96.0


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def rng_state():
    return torch.random.get_rng_state(), torch.cuda.get_rng_state_all()


def set_rng(state):
    torch.random.set_rng_state(state[0])
    torch.cuda.set_rng_state_all(state[1])


def load_audio(path, probe):
    with wave.open(str(path), "rb") as source:
        if (
            source.getnchannels() != 1
            or source.getsampwidth() != 2
            or source.getframerate() != probe.mimi.sample_rate
        ):
            raise ValueError("audio must be 16-bit mono at the model sample rate")
        samples = np.frombuffer(source.readframes(source.getnframes()), np.int16)
    active = np.flatnonzero(samples)
    samples = samples[active[0] : active[-1] + 1]
    samples = torch.from_numpy(samples.copy()).float().div_(32768)
    samples = torch.nn.functional.pad(samples, (0, -len(samples) % probe.frame_size))
    return [
        frame.to(probe.device).view(1, 1, -1)
        for frame in samples.split(probe.frame_size)
    ]


def load_noise(path, probe, frames):
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError("noise must be 16-bit mono")
        rate = source.getframerate()
        samples = np.frombuffer(source.readframes(source.getnframes()), np.int16)
    samples = samples.astype(np.float32) / 32768
    length = round(len(samples) * probe.mimi.sample_rate / rate)
    samples = np.interp(
        np.arange(length) * rate / probe.mimi.sample_rate,
        np.arange(len(samples)),
        samples,
    )
    rms = np.sqrt(np.mean(samples**2))
    if not rms:
        raise ValueError("noise must not be silent")
    samples *= 10 ** (NOISE_DBFS / 20) / rms
    samples = np.resize(samples, frames * probe.frame_size)
    samples = torch.from_numpy(samples).float()
    return [
        frame.to(probe.device).view(1, 1, -1)
        for frame in samples.split(probe.frame_size)
    ]


def mix(audio, noise):
    return [
        (frame + input_frame(noise, index)).clamp(-1, 1)
        for index, frame in enumerate(audio)
    ]


def input_frame(audio, frame):
    return audio[frame] if frame < len(audio) else None


def sentence(probe, audio=None):
    tokens = []
    audio = audio or []
    for frame in range(len(audio) + 750):
        token, _ = probe.step(input_frame(audio, frame))
        if token is None:
            continue
        tokens.append(token)
        if (
            frame >= len(audio)
            and token >= 4
            and probe.text(tokens).rstrip().endswith((".", "?", "!"))
        ):
            return probe.text(tokens)
    raise RuntimeError("sentence did not end")


def utterance(probe, audio, gap_frames):
    tokens = []
    gap = 0
    speech = False
    for frame in range(len(audio) + 750):
        token, _ = probe.step(input_frame(audio, frame))
        if token is not None:
            tokens.append(token)
        if token is not None and token >= 4:
            speech = True
            gap = 0
        elif speech:
            gap += 1
        if frame >= len(audio) and gap >= gap_frames:
            return probe.text(tokens)
    raise RuntimeError("utterance did not end")


def prepare(probe, prompt=None, gap_frames=None):
    probe.reset()
    seed(0)
    greeting = sentence(probe)
    answer = None
    if prompt is not None:
        seed(0)
        answer = utterance(probe, prompt, gap_frames)
    return greeting, answer


def trace(probe, audio, run_seed, frames):
    prepare(probe)
    seed(run_seed)
    tokens = []
    for frame in range(len(audio) + frames):
        token, _ = probe.step(input_frame(audio, frame))
        if token is not None:
            tokens.append({"frame": frame, "token": token})
    result = {
        "seed": run_seed,
        "text": probe.text([item["token"] for item in tokens]),
        "tokens": tokens,
    }
    print(f"calibration {run_seed} {result['text']!r}", flush=True)
    return result


def divergence(actual, shadow):
    mean = (actual + shadow) / 2
    actual = torch.where(
        actual > 0,
        actual * (actual.clamp_min(1e-12).log() - mean.clamp_min(1e-12).log()),
        0,
    )
    shadow = torch.where(
        shadow > 0,
        shadow * (shadow.clamp_min(1e-12).log() - mean.clamp_min(1e-12).log()),
        0,
    )
    return float((actual + shadow).sum() / 2)


def decision_boundaries(idle, speech):
    scores = sorted(set(idle + speech))
    boundaries = [np.nextafter(scores[0], -np.inf)]
    boundaries += [(left + right) / 2 for left, right in zip(scores, scores[1:])]
    boundaries.append(scores[-1])

    def accuracy(boundary):
        idle_accuracy = np.mean([score <= boundary for score in idle])
        speech_accuracy = np.mean([score > boundary for score in speech])
        return (idle_accuracy + speech_accuracy) / 2

    speech_min = min(speech)
    response = max((score for score in idle if score < speech_min), default=0.0)
    idle_max = max(idle)
    next_speech = [score for score in speech if score > idle_max]
    suppression = np.nextafter(min(next_speech), -np.inf) if next_speech else idle_max
    return {
        "response": float(response),
        "balanced": float(max(boundaries, key=accuracy)),
        "suppression": float(suppression),
    }


def onset_score(result):
    return result["scores"][0]["score"] if result["scores"] else None


def threshold_results(rollouts, boundaries):
    return {
        name: [
            {
                "seed": rollout["seed"],
                "onset": onset_score(rollout) is not None,
                "suppressed": (
                    onset_score(rollout) is not None
                    and onset_score(rollout) <= boundary
                ),
            }
            for rollout in rollouts
        ]
        for name, boundary in boundaries.items()
    }


def pair_step(observed, shadow, audio, counterfactual, frame):
    state = rng_state()
    token, actual = observed.step(input_frame(audio, frame), paired=True)
    observed_state = rng_state()
    set_rng(state)
    _, shadow_probability = shadow.step(
        input_frame(counterfactual, frame), token, paired=True
    )
    set_rng(observed_state)
    return token, actual.cpu(), shadow_probability.cpu()


def prepare_pair(observed, shadow, prompt, counterfactual_prompt, gap_frames):
    greeting, _ = prepare(observed)
    prepare(shadow)
    seed(0)
    tokens = []
    gap = 0
    speech = False
    for frame in range(len(prompt) + 750):
        token, _, _ = pair_step(observed, shadow, prompt, counterfactual_prompt, frame)
        if token is None:
            if speech:
                gap += 1
        else:
            tokens.append(token)
            if token >= 4:
                speech = True
                gap = 0
            elif speech:
                gap += 1
        if frame >= len(prompt) and gap >= gap_frames:
            return greeting, observed.text(tokens)
    raise RuntimeError("utterance did not end")


def paired_gate(
    observed,
    shadow,
    prompt,
    gap_frames,
    run_seed,
    audio,
    counterfactual,
    frames,
):
    prepare_pair(observed, shadow, prompt, prompt, gap_frames)
    seed(run_seed)
    scores = []
    gap = 0
    for frame in range(frames):
        qualified = gap >= gap_frames
        token, actual, muted = pair_step(observed, shadow, audio, counterfactual, frame)
        if qualified and token is not None and token >= 4:
            score = divergence(actual, muted)
            scores.append({"frame": frame, "score": score})
            print(f"gate {run_seed} {frame} {score:.6f}", flush=True)
            break
        gap = 0 if token is not None and token >= 4 else gap + 1
    return {"seed": run_seed, "scores": scores}


def baseline(probe, prompt, gap_frames, run_seed, audio, frames):
    prepare(probe, prompt, gap_frames)
    seed(run_seed)
    probabilities = []
    onset = None
    gap = 0
    for frame in range(frames):
        token, distribution = probe.step(input_frame(audio, frame))
        if distribution is not None:
            probabilities.append(float(distribution[4:].sum()))
        if token is not None and token >= 4 and gap >= gap_frames:
            onset = frame
            break
        gap = 0 if token is not None and token >= 4 else gap + 1
    print(f"baseline {run_seed} {onset}", flush=True)
    return {
        "seed": run_seed,
        "onset_frame": onset,
        "onset_probabilities": probabilities,
    }


def response(
    observed,
    shadow,
    prompt,
    run_seed,
    audio,
    counterfactual,
    gap_frames,
    speech_frames=None,
):
    prepare_pair(observed, shadow, prompt, prompt, gap_frames)
    seed(run_seed)
    speech_frames = len(audio) if speech_frames is None else speech_frames
    tokens = []
    onset = None
    score = None
    for frame in range(speech_frames + 750):
        token, actual, muted = pair_step(observed, shadow, audio, counterfactual, frame)
        if token is None:
            continue
        tokens.append(token)
        if token >= 4 and onset is None:
            onset = frame
            score = divergence(actual, muted)
        if (
            frame >= speech_frames
            and token >= 4
            and observed.text(tokens).rstrip().endswith((".", "?", "!"))
        ):
            break
    print(f"response {run_seed} {onset} {score}", flush=True)
    return {
        "seed": run_seed,
        "text": observed.text(tokens),
        "onset_frame": onset,
        "score": score,
    }


def benchmark(probe, shadow, measurements):
    prepare(probe)
    prepare(shadow)
    seed(0)
    for _ in range(100):
        pair_step(probe, shadow, [], [], 0)
    times = []
    for _ in range(measurements):
        torch.cuda.synchronize(probe.device)
        start = time.perf_counter()
        pair_step(probe, shadow, [], [], 0)
        torch.cuda.synchronize(probe.device)
        times.append(1000 * (time.perf_counter() - start))
    return {
        "measurements": measurements,
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "times_ms": times,
    }


def save(output, result):
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")


def run(
    probe,
    shadow,
    audio_path,
    noise_path,
    output,
    seeds,
    seed_offset,
    frames=SESSION_FRAMES,
    measurements=1000,
    gap_frames=None,
):
    output.mkdir(parents=True, exist_ok=True)
    prompt = load_audio(audio_path, probe)
    noise = load_noise(noise_path, probe, frames)
    noisy_prompt = mix(prompt, noise)
    greeting, answer = prepare(probe, prompt, gap_frames)
    calibration_path = output / "calibration.json"
    calibration = (
        json.loads(calibration_path.read_text()) if calibration_path.exists() else {}
    )
    calibration_idle = calibration.get("conditional_idle", [])
    calibration_responses = calibration.get("speech", [])

    def save_calibration():
        calibration_path.write_text(
            json.dumps(
                {
                    "noise_dbfs": NOISE_DBFS,
                    "conditional_idle": calibration_idle,
                    "speech": calibration_responses,
                },
                indent=2,
            )
            + "\n"
        )

    idle_seeds = {result["seed"] for result in calibration_idle}
    for run_seed in range(seeds):
        if run_seed not in idle_seeds:
            calibration_idle.append(
                paired_gate(
                    probe,
                    shadow,
                    noisy_prompt,
                    gap_frames,
                    run_seed,
                    noise,
                    noise,
                    frames,
                )
            )
            save_calibration()
    response_seeds = {result["seed"] for result in calibration_responses}
    for run_seed in range(seeds):
        if run_seed not in response_seeds:
            calibration_responses.append(
                response(
                    probe,
                    shadow,
                    noisy_prompt,
                    run_seed,
                    noisy_prompt,
                    [],
                    gap_frames,
                    speech_frames=len(prompt),
                )
            )
            save_calibration()

    idle_scores = [
        onset_score(result)
        for result in calibration_idle
        if onset_score(result) is not None
    ]
    speech_scores = [
        result["score"]
        for result in calibration_responses
        if result["score"] is not None
    ]
    boundaries = decision_boundaries(idle_scores, speech_scores)
    result_path = output / "results.json"
    previous = json.loads(result_path.read_text()) if result_path.exists() else {}
    result = {
        "model": probe.metadata,
        "greeting": greeting,
        "prompt_answer": answer,
        "prompt_frames": len(prompt),
        "context_seed": 0,
        "frame_rate": FRAME_RATE,
        "session_s": frames / FRAME_RATE,
        "gap_frames": gap_frames,
        "noise_dbfs": NOISE_DBFS,
        "boundaries": boundaries,
        "calibration_idle": calibration_idle,
        "calibration_responses": calibration_responses,
        "baseline": previous.get("baseline", []),
        "microphone_rollouts": previous.get("microphone_rollouts", []),
        "microphone": {},
        "microphone_responses": previous.get("microphone_responses", []),
    }
    result["microphone"] = threshold_results(result["microphone_rollouts"], boundaries)
    save(output, result)
    baseline_seeds = {run["seed"] for run in result["baseline"]}
    rollout_seeds = {run["seed"] for run in result["microphone_rollouts"]}
    response_seeds = {run["seed"] for run in result["microphone_responses"]}
    for run_seed in range(seed_offset, seed_offset + seeds):
        if run_seed not in baseline_seeds:
            result["baseline"].append(
                baseline(probe, prompt, gap_frames, run_seed, [], frames)
            )
            save(output, result)
        if run_seed not in rollout_seeds:
            result["microphone_rollouts"].append(
                paired_gate(
                    probe,
                    shadow,
                    noisy_prompt,
                    gap_frames,
                    run_seed,
                    noise,
                    noise,
                    frames,
                )
            )
            result["microphone"] = threshold_results(
                result["microphone_rollouts"], boundaries
            )
            save(output, result)
        if run_seed not in response_seeds:
            result["microphone_responses"].append(
                response(
                    probe,
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
    result["runtime"] = benchmark(probe, shadow, measurements)
    save(output, result)
    print(json.dumps(result), flush=True)


def calibrate(probe, audio_path, output, seeds, frames):
    output.mkdir(parents=True, exist_ok=True)
    audio = load_audio(audio_path, probe)
    result = {
        "model": probe.metadata,
        "frame_rate": FRAME_RATE,
        "prompt_frames": len(audio),
        "traces": [],
    }
    save(output, result)
    for run_seed in range(seeds):
        trace_result = trace(probe, audio, run_seed, frames)
        result["traces"].append(trace_result)
        lexical = [
            item["frame"] for item in trace_result["tokens"] if item["token"] >= 4
        ]
        gap = max(
            (right - left - 1 for left, right in zip(lexical, lexical[1:])),
            default=0,
        )
        result["maximum_gap_frames"] = max(gap, result.get("maximum_gap_frames", 0))
        result["gap_frames"] = result["maximum_gap_frames"] + 1
        save(output, result)

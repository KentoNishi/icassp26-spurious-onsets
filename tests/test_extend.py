import copy
import hashlib
import io
import json
import runpy
import sys
import tarfile
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import extend
from protocol import save


def fixture():
    """Synthetic measurements only; never used as experimental results."""
    return {
        "model": {"repo": "fixture"},
        "frame_rate": 12.5,
        "session_s": 300,
        "prompt_frames": 2,
        "gap_frames": 14,
        "noise_dbfs": -96,
        "boundaries": {"response": 0.1, "balanced": 0.2, "suppression": 0.3},
        "calibration_idle": [{"seed": n} for n in range(40)],
        "calibration_responses": [{"seed": n} for n in range(40)],
        "baseline": [{"seed": n} for n in range(10000, 10040)],
        "runtime": {"p95_ms": 50},
        "microphone_rollouts": [{"seed": n, "scores": []} for n in range(10000, 10040)],
        "microphone_responses": [
            {"seed": n, "score": 0.4} for n in range(10000, 10040)
        ],
        "microphone": {},
    }


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name)
        self.probe = SimpleNamespace(metadata={"repo": "fixture"})
        self.result = fixture()

    def mocked_protocol(self, stack):
        stack.enter_context(patch.object(extend, "load_audio", return_value=[1, 2]))
        stack.enter_context(patch.object(extend, "load_noise", return_value=[3, 4]))
        stack.enter_context(patch.object(extend, "mix", return_value=[5, 6]))
        for name in ("calibrate", "decision_boundaries", "baseline", "benchmark"):
            stack.enter_context(
                patch(f"protocol.{name}", side_effect=AssertionError(name))
            )
        idle = stack.enter_context(
            patch.object(
                extend,
                "paired_gate",
                side_effect=lambda *args: {
                    "seed": args[4],
                    "scores": [{"frame": 42, "score": 0.15}],
                },
            )
        )
        speech = stack.enter_context(
            patch.object(
                extend,
                "response",
                side_effect=lambda *args, **kwargs: {"seed": args[3], "score": 0.25},
            )
        )
        return idle, speech

    def run_extension(self, result, trials):
        extend.extend(
            self.probe,
            self.probe,
            Path("speech.wav"),
            Path("noise.wav"),
            self.output,
            result,
            trials,
        )

    def test_release_bootstrap_checks_digest_and_saves_only_requested_model(self):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            data = json.dumps(self.result).encode()
            member = tarfile.TarInfo("runs/moshi/results.json")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        data = buffer.getvalue()
        with (
            patch.object(
                extend.urllib.request, "urlopen", return_value=io.BytesIO(data)
            ),
            patch.object(extend, "RELEASE_SHA256", hashlib.sha256(data).hexdigest()),
        ):
            result = extend.load_results("moshi", self.output, 500)
        self.assertEqual(result, self.result)
        self.assertEqual(json.loads((self.output / "results.json").read_text()), result)
        self.assertEqual([p.name for p in self.output.iterdir()], ["results.json"])

    def test_cached_results_need_no_download(self):
        save(self.output, self.result)
        with patch.object(extend.urllib.request, "urlopen", side_effect=AssertionError):
            self.assertEqual(
                extend.load_results("moshi", self.output, 500), self.result
            )

    def test_bad_release_digest_does_not_write_results(self):
        with patch.object(
            extend.urllib.request, "urlopen", return_value=io.BytesIO(b"bad")
        ):
            with self.assertRaisesRegex(ValueError, "checksum"):
                extend.load_results("moshi", self.output, 500)
        self.assertFalse((self.output / "results.json").exists())

    def test_invalid_target_does_not_download(self):
        with patch.object(extend.urllib.request, "urlopen", side_effect=AssertionError):
            with self.assertRaisesRegex(ValueError, "at least"):
                extend.load_results("moshi", self.output, 39)

    def test_invalid_seed_sets_and_recalibration_are_rejected_without_writes(self):
        for mutation in ("duplicate", "missing", "outside", "recalibrated"):
            with self.subTest(mutation=mutation):
                result = fixture()
                if mutation == "duplicate":
                    result["microphone_rollouts"].append(
                        result["microphone_rollouts"][0]
                    )
                elif mutation == "missing":
                    result["microphone_responses"].pop()
                elif mutation == "outside":
                    result["microphone_rollouts"].append({"seed": 10500, "scores": []})
                else:
                    result["calibration_idle"].append({"seed": 40})
                save(self.output, result)
                before = (self.output / "results.json").read_bytes()
                with self.assertRaises(ValueError):
                    extend.load_results("moshi", self.output, 500)
                self.assertEqual((self.output / "results.json").read_bytes(), before)

    def test_extension_preserves_original_data_and_exact_protocol_arguments(self):
        before = copy.deepcopy(self.result)
        with ExitStack() as stack:
            idle, speech = self.mocked_protocol(stack)
            self.run_extension(self.result, 41)
        idle.assert_called_once_with(
            self.probe, self.probe, [5, 6], 14, 10040, [3, 4], [3, 4], 3750
        )
        speech.assert_called_once_with(
            self.probe, self.probe, [5, 6], 10040, [5, 6], [], 14, speech_frames=2
        )
        for key, value in before.items():
            if key in extend.GROUPS:
                self.assertEqual(self.result[key][:40], value)
                self.assertEqual(len(self.result[key]), 41)
            elif key != "microphone":
                self.assertEqual(self.result[key], value)
        self.assertFalse(self.result["microphone"]["response"][-1]["suppressed"])
        self.assertTrue(self.result["microphone"]["balanced"][-1]["suppressed"])

    def test_restart_resumes_each_measurement_independently(self):
        with ExitStack() as stack:
            idle, speech = self.mocked_protocol(stack)
            speech.side_effect = RuntimeError("interrupted")
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.run_extension(self.result, 41)
        resumed = extend.load_results("moshi", self.output, 43)
        self.assertEqual(len(resumed["microphone_rollouts"]), 41)
        self.assertEqual(len(resumed["microphone_responses"]), 40)
        with ExitStack() as stack:
            idle, speech = self.mocked_protocol(stack)
            self.run_extension(resumed, 43)
        self.assertEqual([call.args[4] for call in idle.call_args_list], [10041, 10042])
        self.assertEqual(
            [call.args[3] for call in speech.call_args_list], [10040, 10041, 10042]
        )
        with ExitStack() as stack:
            idle, speech = self.mocked_protocol(stack)
            self.run_extension(resumed, 43)
            idle.assert_not_called()
            speech.assert_not_called()

    def test_settings_mismatch_fails_before_running_trials(self):
        for key, value in (("model", {}), ("session_s", 30), ("noise_dbfs", -40)):
            with self.subTest(key=key), ExitStack() as stack:
                result = fixture()
                result[key] = value
                idle, speech = self.mocked_protocol(stack)
                with self.assertRaisesRegex(ValueError, "settings"):
                    self.run_extension(result, 41)
                idle.assert_not_called()
                speech.assert_not_called()

    def test_atomic_save_keeps_previous_checkpoint_when_replace_fails(self):
        save(self.output, self.result)
        before = (self.output / "results.json").read_bytes()
        with patch.object(Path, "replace", side_effect=OSError("interrupted")):
            with self.assertRaises(OSError):
                save(self.output, {"incomplete": True})
        self.assertEqual((self.output / "results.json").read_bytes(), before)

    def test_completed_cli_exits_without_loading_models(self):
        entry = Path(__file__).resolve().parents[1] / "experiment.py"
        with (
            patch.object(sys, "argv", [str(entry), "moshi", "--extend", "40"]),
            patch.object(extend, "load_results", return_value=self.result),
            patch.object(extend, "extend", side_effect=AssertionError),
            patch.dict(
                sys.modules,
                {
                    "moshi_probe": SimpleNamespace(
                        Probe=Mock(side_effect=AssertionError)
                    )
                },
            ),
            self.assertRaises(SystemExit) as stopped,
        ):
            runpy.run_path(str(entry), run_name="__main__")
        self.assertEqual(stopped.exception.code, 0)

    def test_default_cli_keeps_original_experiment_parameters(self):
        entry = Path(__file__).resolve().parents[1] / "experiment.py"
        probes = [object(), object()]
        with (
            patch.object(sys, "argv", [str(entry), "moshi"]),
            patch.object(Path, "read_text", return_value='{"gap_frames": 14}'),
            patch.dict(
                sys.modules,
                {"moshi_probe": SimpleNamespace(Probe=Mock(side_effect=probes))},
            ),
            patch("protocol.run") as run,
        ):
            runpy.run_path(str(entry), run_name="__main__")
        run.assert_called_once_with(
            *probes,
            entry.parent / "speech.wav",
            entry.parent / "microphone.wav",
            entry.parent / "runs" / "moshi",
            40,
            10000,
            3750,
            1000,
            14,
        )

    def test_extension_cli_uses_frozen_results_not_calibration_files(self):
        entry = Path(__file__).resolve().parents[1] / "experiment.py"
        probes = [object(), object()]
        with (
            patch.object(sys, "argv", [str(entry), "moshi", "--extend", "41"]),
            patch.object(extend, "load_results", return_value=self.result),
            patch.object(Path, "read_text", side_effect=AssertionError),
            patch.dict(
                sys.modules,
                {"moshi_probe": SimpleNamespace(Probe=Mock(side_effect=probes))},
            ),
            patch.object(extend, "extend") as run,
            patch("protocol.run", side_effect=AssertionError),
            self.assertRaises(SystemExit) as stopped,
        ):
            runpy.run_path(str(entry), run_name="__main__")
        self.assertEqual(stopped.exception.code, 0)
        run.assert_called_once_with(
            *probes,
            entry.parent / "speech.wav",
            entry.parent / "microphone.wav",
            entry.parent / "runs" / "moshi",
            self.result,
            41,
        )


if __name__ == "__main__":
    unittest.main()

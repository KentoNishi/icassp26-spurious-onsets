import json
import math
import unittest

from summarize import onset_statistics, summarize


def trace(probabilities, onset=True, seed=10000):
    return {
        "seed": seed,
        "onset_frame": len(probabilities) - 1 if onset else None,
        "onset_probabilities": probabilities,
    }


class OnsetStatisticsTests(unittest.TestCase):
    def test_spike_excludes_onset_from_cumulative_probability(self):
        result = onset_statistics([trace([1e-12, 2e-12, 0.96])])
        self.assertEqual(result["maximum_preceding_probability"], 2e-12)
        self.assertEqual(result["minimum_onset_probability"], 0.96)
        self.assertAlmostEqual(result["minimum_log10_jump"], math.log10(0.96 / 2e-12))
        self.assertAlmostEqual(
            result["maximum_cumulative_probability_before_onset"] / 3e-12, 1.0
        )

    def test_minimum_jump_checks_every_onset_not_maximum_range(self):
        result = onset_statistics(
            [
                trace([1e-12, 1e-12, 0.96]),
                trace([1e-12, 0.1, 0.2], seed=10001),
            ]
        )
        self.assertAlmostEqual(result["minimum_log10_jump"], math.log10(2))
        self.assertGreater(result["maximum_cumulative_probability_before_onset"], 0.1)

    def test_zero_previous_probability_has_explicit_infinite_jump(self):
        result = onset_statistics([trace([0.0, 0.96])])
        self.assertIsNone(result["minimum_log10_jump"])
        self.assertEqual(result["onsets_with_zero_preceding_probability"], 1)
        self.assertEqual(result["maximum_cumulative_probability_before_onset"], 0)
        json.dumps(result, allow_nan=False)

    def test_zero_and_finite_previous_probabilities_keep_finite_minimum(self):
        result = onset_statistics([trace([0.0, 0.96]), trace([0.1, 0.2], seed=10001)])
        self.assertAlmostEqual(result["minimum_log10_jump"], math.log10(2))

    def test_first_frame_onset_has_no_adjacent_frame(self):
        result = onset_statistics([trace([0.96])])
        self.assertEqual(result["onsets_without_preceding_frame"], 1)
        self.assertEqual(result["onsets_with_zero_preceding_probability"], 0)
        self.assertIsNone(result["maximum_preceding_probability"])
        self.assertIsNone(result["minimum_log10_jump"])

    def test_censored_runs_do_not_contribute_onset_statistics(self):
        result = onset_statistics(
            [trace([0.1, 0.96]), trace([0.9, 0.9], onset=False, seed=10001)]
        )
        self.assertEqual(len(result["onsets"]), 1)
        self.assertEqual(result["maximum_preceding_probability"], 0.1)

    def test_no_observed_onsets(self):
        result = onset_statistics([trace([0.0, 0.0], onset=False)])
        self.assertEqual(result["onsets"], [])
        self.assertIsNone(result["minimum_onset_probability"])
        self.assertIsNone(result["maximum_cumulative_probability_before_onset"])

    def test_cumulative_probability_handles_one(self):
        result = onset_statistics([trace([1.0, 0.96])])
        self.assertEqual(result["maximum_cumulative_probability_before_onset"], 1)

    def test_float32_sum_roundoff_is_bounded_without_mutating_trace(self):
        run = trace([1.0 + 1e-7, 1.0 + 1e-7])
        result = onset_statistics([run])
        self.assertEqual(result["minimum_onset_probability"], 1.0)
        self.assertEqual(result["maximum_cumulative_probability_before_onset"], 1.0)
        self.assertGreater(run["onset_probabilities"][0], 1.0)

    def test_subnormal_probability_does_not_overflow_jump_or_cancel_cumulative(self):
        result = onset_statistics([trace([5e-324, 0.96])])
        self.assertTrue(math.isfinite(result["minimum_log10_jump"]))
        self.assertGreater(result["minimum_log10_jump"], 323)
        self.assertEqual(result["maximum_cumulative_probability_before_onset"], 5e-324)

    def test_invalid_probabilities_and_missing_frames_are_rejected(self):
        for values in (
            [],
            [-0.1, 0.96],
            [float("nan"), 0.96],
            [float("inf"), 0.96],
            [1.1, 0.96],
            [0.1, 0.0],
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                onset_statistics([trace(values)])
        run = trace([0.1, 0.96])
        run["onset_frame"] = 2
        with self.assertRaisesRegex(ValueError, "every frame"):
            onset_statistics([run])

    def test_summary_preserves_existing_fields_and_adds_audit(self):
        runs = [{"seed": 10000, "onset": True, "suppressed": True}]
        result = {
            "model": "fixture",
            "frame_rate": 12.5,
            "session_s": 300,
            "gap_frames": 2,
            "noise_dbfs": -96,
            "baseline": [trace([1e-12, 0.96])],
            "calibration_idle": [{}],
            "calibration_responses": [{}],
            "microphone_rollouts": runs,
            "microphone_responses": [{"seed": 10000, "score": 0.3}],
            "microphone": {
                name: runs for name in ("response", "balanced", "suppression")
            },
            "boundaries": {
                name: 0.2 for name in ("response", "balanced", "suppression")
            },
            "runtime": {"times_ms": [45.0], "median_ms": 45.0, "p95_ms": 45.0},
        }
        summary = summarize(result, trials=1, measurements=1)
        self.assertEqual(summary["baseline"]["onsets"], 1)
        self.assertIn("maximum_orders_within_continuation", summary["baseline"])
        self.assertEqual(
            summary["thresholds"]["balanced"]["first_onsets_suppressed"], 1
        )
        self.assertEqual(summary["runtime"]["p95_ms"], 45.0)
        self.assertEqual(len(summary["baseline"]["onset_statistics"]["onsets"]), 1)
        json.dumps(summary, allow_nan=False)


if __name__ == "__main__":
    unittest.main()

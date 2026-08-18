import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_results


def fixture(onsets=12, speech_max=0.6):
    """Synthetic data only: exercise layout, not the manuscript's measurements."""
    baseline = []
    for seed in range(40):
        frame = 330 + 250 * seed if seed < onsets else None
        probabilities = np.full(3750 if frame is None else frame + 1, 1e-12)
        if frame is not None:
            probabilities[-1] = 0.96
        baseline.append(
            {
                "seed": seed,
                "onset_frame": frame,
                "onset_probabilities": probabilities.tolist(),
            }
        )
    return {
        "baseline": baseline,
        "frame_rate": 12.5,
        "session_s": 300,
        "calibration_idle": [
            {"scores": [{"score": float(value)}]}
            for value in np.linspace(0.01, 0.06, 12)
        ],
        "calibration_responses": [
            {"score": float(value)} for value in np.linspace(0.2, speech_max, 40)
        ],
        "boundaries": {"response": 0.06, "balanced": 0.13, "suppression": 0.2},
    }


class PaperPlotTests(unittest.TestCase):
    def render(self, function, results):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(plot_results, "OUT", Path(directory)),
                patch.object(plt, "close"),
            ):
                function(results)
                fig = plt.gcf()
                fig.canvas.draw()
                self.assertGreater(
                    (Path(directory) / f"{function.__name__}.pdf").stat().st_size, 1000
                )
            self.addCleanup(plt.close, fig)
        return fig

    def check_titles(self, fig):
        titles = [
            text for text in fig.texts if text.get_text() in ("Moshi", "PersonaPlex")
        ]
        self.assertEqual(len(titles), 2)
        for title in titles:
            self.assertEqual(title.get_fontweight(), "bold")
            self.assertEqual(title.get_fontsize(), 7)

    def test_mechanism_has_paper_geometry_and_shared_axis_key(self):
        fig = self.render(plot_results.mechanism, [fixture(), fixture(11, 0.7)])
        np.testing.assert_allclose(
            fig.get_size_inches() * 72, [plot_results.WIDTH, 100]
        )
        self.check_titles(fig)
        footer = fig.artists[0]
        labels = footer.get_child().get_children()
        self.assertEqual(labels[0].get_text(), "Onset time (s);")
        self.assertEqual(labels[0].get_children()[0].get_fontweight(), "bold")
        self.assertIn("left: incidence; right: probability", labels[1].get_text())
        bounds = footer.get_window_extent(fig.canvas.get_renderer())
        self.assertGreaterEqual(bounds.x0, fig.bbox.x0)
        self.assertLessEqual(bounds.x1, fig.bbox.x1)
        self.assertTrue(all(not ax.get_ylabel() for ax in fig.axes))

    def test_counterfactual_has_paper_geometry_and_external_legend(self):
        fig = self.render(plot_results.counterfactual, [fixture(), fixture(11, 0.7)])
        np.testing.assert_allclose(fig.get_size_inches() * 72, [plot_results.WIDTH, 96])
        self.check_titles(fig)
        divergence = next(
            text for text in fig.texts if text.get_text() == "Divergence (Dₜ)"
        )
        self.assertEqual(divergence.get_fontweight(), "bold")
        self.assertEqual(
            [text.get_text() for text in fig.legends[0].get_texts()],
            ["preserve", "balanced", "suppress"],
        )
        self.assertTrue(all(ax.get_legend() is None for ax in fig.axes))
        legend = fig.legends[0].get_window_extent(fig.canvas.get_renderer())
        self.assertLess(legend.y1, min(ax.bbox.y0 for ax in fig.axes))

    def test_all_zero_censored_traces_and_empty_score_groups_render(self):
        result = fixture(onsets=0)
        for run in result["baseline"]:
            run["onset_probabilities"] = [0.0] * 3750
        result["calibration_idle"] = [{"scores": []}]
        result["calibration_responses"] = [{"score": None}]
        self.render(plot_results.mechanism, [result, result])
        self.render(plot_results.counterfactual, [result, result])


if __name__ == "__main__":
    unittest.main()

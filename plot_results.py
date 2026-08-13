import argparse
import colorsys
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D


MOSHI = "#d62728"
PERSONAPLEX = "#76b900"
GRID = "#dddddd"
TEXT = "#333333"
OUT = Path(__file__).parent / "figures"


def style(ax, *, grid_axis="both"):
    for spine in ax.spines.values():
        spine.set_color(TEXT)
        spine.set_linewidth(1.25)
    ax.tick_params(colors=TEXT, labelsize=10, width=1.15, length=4)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.85)
    ax.set_axisbelow(True)


def shades(color, count):
    hue, lightness, saturation = colorsys.rgb_to_hls(*to_rgb(color))
    return [
        colorsys.hls_to_rgb(
            (hue + hue_offset) % 1,
            np.clip(lightness + lightness_offset, 0.18, 0.78),
            saturation,
        )
        for hue_offset, lightness_offset in zip(
            np.linspace(-0.035, 0.035, count),
            np.linspace(-0.18, 0.18, count),
        )
    ]


def incidence(onsets, horizon=300, runs=40):
    times = np.sort(np.asarray(onsets, dtype=float))
    x = np.concatenate(([0], times, [horizon]))
    y = np.concatenate(([0], np.arange(1, len(times) + 1) / runs, [len(times) / runs]))
    return x, y


def mechanism(results):
    models = (
        ("Moshi", MOSHI, results[0]),
        ("PersonaPlex", PERSONAPLEX, results[1]),
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.15))
    grid = np.linspace(0, 300, 1201)
    prepared = []

    for name, color, result in models:
        runs = result["baseline"]
        rate = result["frame_rate"]
        onsets = [
            run["onset_frame"] / rate for run in runs if run["onset_frame"] is not None
        ]
        observed = np.asarray([len(run["onset_probabilities"]) for run in runs])
        events = np.asarray([run["onset_frame"] is not None for run in runs])
        h0 = events.sum() / observed.sum()
        x, y = incidence(onsets, result["session_s"], len(runs))
        geometric = 1 - (1 - h0) ** (rate * grid)
        trajectories = [np.asarray(run["onset_probabilities"]) for run in runs]
        probability_limits = []
        for trajectory in trajectories:
            positive = trajectory[trajectory > 0]
            probability_limits.extend((positive.min(), positive.max()))
        prepared.append(
            (
                name,
                color,
                rate,
                x,
                y,
                geometric,
                trajectories,
                np.ceil(10 * max(y.max(), geometric.max())) / 10,
                10 ** np.floor(np.log10(min(probability_limits))),
                1.5 * max(probability_limits),
            )
        )

    for ax, data in zip(axes, prepared):
        (
            name,
            color,
            rate,
            x,
            y,
            geometric,
            trajectories,
            incidence_max,
            probability_min,
            probability_max,
        ) = data
        probability_ax = ax.twinx()
        probability_ax.set_zorder(0)
        ax.set_zorder(1)
        ax.patch.set_visible(False)

        ax.step(x, y, where="post", color=color, linewidth=2.7, zorder=4)
        ax.plot(grid, geometric, color=color, linewidth=2.0, linestyle=":", zorder=3)
        for trajectory, trajectory_color in zip(
            trajectories, shades(color, len(trajectories))
        ):
            positive = trajectory[trajectory > 0]
            probability_ax.plot(
                np.arange(len(trajectory)) / rate,
                np.maximum(trajectory, positive.min()),
                color=trajectory_color,
                linewidth=0.3,
                alpha=0.55,
                zorder=1,
            )

        ax.set_xlim(0, 300)
        ax.set_ylim(0, incidence_max)
        ax.set_yticks(np.arange(0, incidence_max + 0.01, 0.1))
        ax.set_xlabel("seconds", fontsize=12, color=TEXT)
        ax.set_ylabel("onset incidence", fontsize=10, color=TEXT, labelpad=1)
        ax.set_title(name, fontsize=12, color=TEXT, pad=3)
        for tick_label in ax.get_xticklabels():
            tick_label.set_horizontalalignment("right")
        style(ax)

        probability_ax.set_yscale("log")
        probability_ax.set_ylim(probability_min, probability_max)
        probability_ax.tick_params(
            colors=TEXT,
            labelsize=10,
            width=1.15,
            length=4,
        )
        probability_ax.spines["right"].set_color(TEXT)
        probability_ax.spines["right"].set_linewidth(1.25)
        probability_ax.patch.set_visible(False)
        probability_ax.set_ylabel(
            "onset probability", fontsize=10, color=TEXT, labelpad=1
        )

    fig.subplots_adjust(left=0.10, right=0.87, bottom=0.25, top=0.85, wspace=0.72)
    fig.savefig(OUT / "mechanism.pdf")
    plt.close(fig)


def counterfactual(results):
    models = []
    for name, color, result in zip(
        ("Moshi", "PersonaPlex"), (MOSHI, PERSONAPLEX), results
    ):
        idle = np.asarray(
            [
                run["scores"][0]["score"]
                for run in result["calibration_idle"]
                if run["scores"]
            ]
        )
        speech = np.asarray(
            [
                run["score"]
                for run in result["calibration_responses"]
                if run["score"] is not None
            ]
        )
        boundaries = result["boundaries"]
        thresholds = (
            boundaries["response"],
            boundaries["balanced"],
            boundaries["suppression"],
        )
        models.append((name, color, (idle, speech), thresholds, thresholds[::2]))

    styles = ((0, (1, 1.5)), "-", (0, (5, 2)))
    labels = ("preserve", "balanced", "suppress")
    line_order = (1, 0, 2)
    fig, axes = plt.subplots(1, 2, figsize=(6.25, 2.1), sharey=True)
    positions = (0, 0.72)

    for model_index, (ax, (name, color, scores, thresholds, balanced)) in enumerate(
        zip(axes, models)
    ):
        for score_index, (y, values) in enumerate(zip(positions, scores)):
            if len(values) > 2 and np.ptp(values) > 0:
                violin = ax.violinplot(
                    values,
                    positions=[y],
                    vert=False,
                    widths=0.5,
                    showextrema=False,
                )["bodies"][0]
                violin.set_facecolor(color)
                violin.set_edgecolor(color)
                violin.set_alpha(0.22)
                vertices = violin.get_paths()[0].vertices
                vertices[:, 1] = np.maximum(vertices[:, 1], y)
            jitter = np.random.default_rng(10 * model_index + score_index).uniform(
                -0.14, -0.05, len(values)
            )
            ax.scatter(
                values,
                y + jitter,
                color=color,
                s=13,
                alpha=0.85,
                edgecolors="none",
            )
        ax.axvspan(*balanced, color="#b5b5b5", alpha=0.28, linewidth=0)
        for index in line_order:
            ax.axvline(
                thresholds[index],
                color="#b5b5b5" if index == 1 else TEXT,
                linestyle=styles[index],
                linewidth=2.6 if index == 1 else 1.5,
                zorder=4,
            )
        ax.set_title(name, fontsize=11, color=TEXT)
        upper = (
            np.ceil(10 * max(*(values.max() for values in scores), *thresholds)) / 10
        )
        ax.set_xlim(-0.015, upper)
        ax.set_ylim(-0.24, 1.02)
        ax.set_yticks(positions, ("spurious", "response"))
        ax.set_xlabel(r"divergence ($D_t$)", fontsize=11, color=TEXT)
        style(ax, grid_axis="x")
        ax.tick_params(axis="y", length=0)

    axes[0].legend(
        handles=[
            Line2D(
                [],
                [],
                color="#b5b5b5" if index == 1 else TEXT,
                linestyle=styles[index],
                linewidth=2.6 if index == 1 else 1.5,
                label=label,
            )
            for index, label in enumerate(labels)
        ],
        frameon=False,
        fontsize=8,
        handlelength=1.6,
        loc="lower right",
    )
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.29, top=0.83, wspace=0.12)
    fig.savefig(OUT / "counterfactual.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("moshi", type=Path)
    parser.add_argument("personaplex", type=Path)
    args = parser.parse_args()
    results = tuple(
        json.loads(path.read_text()) for path in (args.moshi, args.personaplex)
    )
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Avenir", "Helvetica", "Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelpad": 5,
        }
    )
    mechanism(results)
    counterfactual(results)


if __name__ == "__main__":
    main()

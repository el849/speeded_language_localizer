import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"

from constants import DATADIR, PLOTDIR

df = pd.read_csv(
    f"{DATADIR}/repeated_sessions_LH_lang.csv",
    index_col=0,
)
df = df.groupby(["Session", "Effect", "Version"]).agg("mean").reset_index()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

bar_width = 0.8
bar_colors = ["maroon", "darksalmon", "maroon", "darksalmon"]
subIDs = ["853", "837"]

version_order = {"Standard": 0, "Speeded": 1}
effect_order = {"S": 0, "N": 1}

for sub_num, subj in enumerate(subIDs):
    df_subj = df.loc[df.Session.str.startswith(subj)].reset_index(drop=True)
    grouped = df_subj.groupby("Session")

    bar_width = 0.2

    all_positions = []
    all_labels = []
    all_bars = []

    for i, (name, group) in enumerate(grouped):
        # Sort the group by custom order of 'Version' and 'Effect'
        sorting_key = (
            group["Version"].map(version_order) * len(effect_order)
            + group["Effect"].map(effect_order)
        ).argsort()
        sorted_group = group.iloc[sorting_key]

        # Plot bars for each 'Effect'
        for j, effect in enumerate(sorted_group["Effect"].unique()):
            effect_data = sorted_group[sorted_group["Effect"] == effect].reset_index(
                drop=True
            )
            positions = [
                i
                + i * (2 * bar_width)
                + j * bar_width
                + k * (len(sorted_group["Version"].unique()) + 1) * bar_width / 1.2
                for k in range(len(effect_data))
            ]

            bar = axes[sub_num].bar(
                positions,
                effect_data["EffectSize"],
                bar_width,
                label=f"{name}, {effect}",
                color=bar_colors[j],
                edgecolor="black",
                alpha=0.6,
                zorder=2,
            )
            all_bars += bar

            all_positions += list(positions)
            all_labels += [
                effect_data.Version[i] + "_" + effect_data.Effect[i]
                for i in range(len(effect_data))
            ]

    axes[sub_num].set_xticks(all_positions)
    axes[sub_num].set_xticklabels(all_labels, rotation=45, ha="right")

    for i in range(len(grouped)):
        axes[sub_num].text(
            all_bars[i * 4 + 2].get_x(),
            4.1,
            f"SESSION {i+1}",
            ha="left",
            va="bottom",
        )

    axes[sub_num].yaxis.set_tick_params(labelsize=16)
    axes[sub_num].yaxis.grid(alpha=0.4, zorder=0)

    axes[sub_num].set_ylabel("BOLD response", fontsize=14)
    axes[sub_num].set_ylim([-0.5, 4.5])
    axes[sub_num].set_title(
        f"Responses to sentences and nonwords\nmean of LH fROIs for three sessions in participant {subj}",
        fontsize=14,
    )


plt.tight_layout()

plt.savefig(f"{PLOTDIR}/consistency_across_sessions.svg")

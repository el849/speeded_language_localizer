# plot spcorr and dice coefficient analysis

import typing
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import os
from os.path import join
from scipy.stats import ttest_rel
from constants import D_COLOR_COND, DATADIR, PLOTDIR


date = datetime.now().strftime("%Y%m%d")

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"

VERSION_COND_ORDER = {
    "Standard SpCorr": 0,
    "Speeded SpCorr": 1,
    "Standard vs. Speeded SpCorr": 2,
    "Standard Dice (Thresh 90)": 4,
    "Speeded Dice (Thresh 90)": 5,
    "Standard vs. Speeded Dice (Thresh 90)": 6,
    "Standard Dice (Thresh 80)": 8,
    "Speeded Dice (Thresh 80)": 9,
    "Standard vs. Speeded Dice (Thresh 80)": 10,
    "Standard Dice (Thresh 70)": 12,
    "Speeded Dice (Thresh 70)": 13,
    "Standard vs. Speeded Dice (Thresh 70)": 14,
}

VERSION_COND_ORDER_WHOLEBRAIN = {
    "Standard SpCorr": 0,
    "Speeded SpCorr": 1,
    "Standard vs. Speeded SpCorr": 2,
    "Standard Dice (Thresh 90)": 4,
    "Speeded Dice (Thresh 90)": 5,
    "Standard vs. Speeded Dice (Thresh 90)": 6,
}

VERSION_COND_ORDER_DICE = {
    "Standard Dice (Thresh 60)": 1,
    "Speeded Dice (Thresh 60)": 2,
    "Standard vs. Speeded Dice (Thresh 60)": 3,
    "Standard Dice (Thresh 65)": 5,
    "Speeded Dice (Thresh 65)": 6,
    "Standard vs. Speeded Dice (Thresh 65)": 7,
    "Standard Dice (Thresh 70)": 9,
    "Speeded Dice (Thresh 70)": 10,
    "Standard vs. Speeded Dice (Thresh 70)": 11,
    "Standard Dice (Thresh 75)": 13,
    "Speeded Dice (Thresh 75)": 14,
    "Standard vs. Speeded Dice (Thresh 75)": 15,
    "Standard Dice (Thresh 80)": 17,
    "Speeded Dice (Thresh 80)": 18,
    "Standard vs. Speeded Dice (Thresh 80)": 19,
    "Standard Dice (Thresh 85)": 21,
    "Speeded Dice (Thresh 85)": 22,
    "Standard vs. Speeded Dice (Thresh 85)": 23,
    "Standard Dice (Thresh 90)": 25,
    "Speeded Dice (Thresh 90)": 26,
    "Standard vs. Speeded Dice (Thresh 90)": 27,
    "Standard Dice (Thresh 95)": 29,
    "Speeded Dice (Thresh 95)": 30,
    "Standard vs. Speeded Dice (Thresh 95)": 31,
}


def get_folder(network: str, hemi: str) -> str:
    """
    Returns folder that plots are saved in
    network: lang or MD
    hemi: LH or RH (only used if network is lang)
    """
    if network == "MD":
        return network
    else:
        return f"{hemi}_{network}"


def _make_bargraph(
    df: pd.DataFrame,
    df_subjectwise: pd.DataFrame,
    version_cond_order: typing.Dict[str, int],
    ax: "plt.Axes",
) -> None:
    """
    Bars effect size for each version condition and scatters individual subject effect sizes within the specified (functionally localized) `network`
    """

    # set index according to version_cond_order
    df.index = df["Version"]
    df = df.reindex(version_cond_order.keys())

    # make grid horizontal grid lines
    ax[0].yaxis.grid(alpha=0.4, zorder=0)
    ax[1].yaxis.grid(alpha=0.4, zorder=0)

    # plot bar graphs for effect sizes of each version condition with standard error of the mean error bars
    print(df)
    print(version_cond_order)
    xaxis_dist = [version_cond_order[x] for x in df["Version"]]
    ax[0].bar(
        xaxis_dist[:3],
        df[("Coefficient", "mean")][:3],
        yerr=df[("sem")][:3],
        capsize=3,
        color=D_COLOR_COND["spcorr"],
        alpha=0.6,
        zorder=2,
        width=0.6,
        edgecolor="black",
    )

    ax[1].bar(
        xaxis_dist[3:],
        df[("Coefficient", "mean")][3:],
        yerr=df[("sem")][3:],
        capsize=3,
        color=D_COLOR_COND["spcorr"],
        alpha=0.6,
        zorder=2,
        width=0.6,
        edgecolor="black",
    )

    # plot points for each individual subject's effect size
    for i, (k, v) in enumerate(
        version_cond_order.items()
    ):  # enumerate over version_cond in intended order, so errorbars and scatter matches
        ind_points = (df_subjectwise.loc[df_subjectwise["Version"] == k])[
            ("Coefficient")
        ].values
        cond_len = len(ind_points)
        jitter_array = np.random.rand(cond_len) * (0.1) - 0.1 / 2
        if i < 3:
            ax[0].scatter(v + jitter_array, ind_points, alpha=0.9, color="grey", s=15)
        else:
            ax[1].scatter(v + jitter_array, ind_points, alpha=0.9, color="grey", s=15)

    # set other plot properties
    ax[0].axhline(y=0, color="black", lw=0.5)
    ax[0].set_title("Fisher-Transformed Spatial Correlation", fontsize=16)
    ax[0].set_ylabel(
        "Mean Fisher-Transformed\nCorrelation Coefficient (± sem)", fontsize=16
    )
    ax[0].set_xticks(
        list(version_cond_order.values())[:3],
        labels=["Standard", "Speeded", "Standard vs.\nSpeeded"],
    )

    ax[1].axhline(y=0, color="black", lw=0.5)
    ax[1].set_title("Dice Overlap Coefficient", fontsize=16)
    ax[1].set_ylabel("Mean Dice Coefficient (± sem)", fontsize=16)
    ax[1].set_xticks(
        list(version_cond_order.values())[3:],
        labels=["Standard", "Speeded", "Standard vs.\nSpeeded"]
        * (len(version_cond_order) // 3 - 1),
        rotation=45,
        ha="right",
    )
    ax[0].yaxis.set_tick_params(labelsize=18)
    ax[1].yaxis.set_tick_params(labelsize=18)

    # Perform t-test between each pair of conditions in version_cond_order
    for i in range(len(version_cond_order) - 1):
        for j in range(i + 1, len(version_cond_order)):
            cond1 = list(version_cond_order.keys())[i]
            cond2 = list(version_cond_order.keys())[j]
            data1 = df_subjectwise.loc[df_subjectwise["Version"] == cond1][
                ("Coefficient")
            ].values
            data2 = df_subjectwise.loc[df_subjectwise["Version"] == cond2][
                ("Coefficient")
            ].values
            t_stat, p_value = ttest_rel(data1, data2)
            print(
                f"T-test between {cond1} and {cond2}: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}"
            )


def plot_avg_effectsize(
    df: pd.DataFrame,
    network: str = "lang",
    hemi: str = "LH",
    save_str: str = "n=?",
    save: bool = True,
) -> None:
    """Plot the average correlation (SpCorr, Dice) values within and across localizer versions for `network` `hemi`"""

    print("*" * 40, network, hemi, "*" * 40)
    df = df.copy()
    df = df.loc[df.Hemisphere == hemi]

    if network == "lang":
        df = df.loc[~df.ROI.str.contains("AngG")]  # exclude AngG from analysis

    ## POOL MANUAL APPROACH ##
    df_subjectwise = df.groupby(["Version", "UID"]).mean().reset_index()
    df_mean = (
        df_subjectwise.groupby(["Version"])
        .agg({"Coefficient": ["mean", "std", "count"]})
        .reset_index()
    )
    df_mean["sem"] = (
        df_mean["Coefficient"]["std"] / df_mean["Coefficient"]["count"] ** 0.5
    )

    if save:
        folder = get_folder(network, hemi)
        plot_dir = f"{PLOTDIR}/{folder}/spcorr_plots_{folder}"

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

    if network == "lang":
        fig, ax = plt.subplots(
            1, 2, figsize=(10, 6), gridspec_kw={"width_ratios": [1, 2]}
        )
        _make_bargraph(
            df_mean,
            df_subjectwise,
            version_cond_order=VERSION_COND_ORDER,
            ax=ax,
        )
    elif network == "wholebrain":
        fig, ax = plt.subplots(
            1, 2, figsize=(8, 6), gridspec_kw={"width_ratios": [1, 1]}
        )
        _make_bargraph(
            df_mean,
            df_subjectwise,
            version_cond_order=VERSION_COND_ORDER_WHOLEBRAIN,
            ax=ax,
        )

    fig.tight_layout()

    # save plots
    if save:
        fig.savefig(
            join(
                plot_dir,
                f"{folder}-network-avg_{save_str}.png",
            )
        )
        fig.savefig(
            join(
                plot_dir,
                f"{folder}-network-avg_{save_str}.svg",
            )
        )
    plt.close(fig)


def _make_bargraph_dice_si(
    df: pd.DataFrame,
    df_subjectwise: pd.DataFrame,
    version_cond_order: typing.Dict[str, int],
    ax: "plt.Axes",
) -> None:
    """
    Helper function for `plot_avg_effectsize_dice_si`
    """

    # set index according to version_cond_order
    df.index = df["Version"]
    df = df.reindex(version_cond_order.keys())

    # make grid horizontal grid lines
    ax.yaxis.grid(alpha=0.4, zorder=0)

    # plot bar graphs for effect sizes of each version condition with standard error of the mean error bars
    xaxis_dist = [version_cond_order[x] for x in df["Version"]]
    ax.bar(
        xaxis_dist,
        df[("Coefficient", "mean")],
        yerr=df[("sem")],
        capsize=3,
        color=D_COLOR_COND["spcorr"],
        alpha=0.6,
        zorder=2,
        width=0.6,
        edgecolor="black",
    )

    # plot points for each individual subject's effect size
    for i, (k, v) in enumerate(
        version_cond_order.items()
    ):  # enumerate over version_cond in intended order, so errorbars and scatter matches
        ind_points = (df_subjectwise.loc[df_subjectwise["Version"] == k])[
            ("Coefficient")
        ].values
        cond_len = len(ind_points)
        jitter_array = np.random.rand(cond_len) * (0.1) - 0.1 / 2
        ax.scatter(v + jitter_array, ind_points, alpha=0.9, color="grey", s=15)

    # set other plot properties

    ax.axhline(y=0, color="black", lw=0.5)
    ax.set_title("Dice Overlap Coefficient", fontsize=16)
    ax.set_ylabel("Mean Dice Coefficient (± sem)", fontsize=16)
    ax.set_xticks(
        list(version_cond_order.values()),
        labels=["Standard", "Speeded", "Standard vs.\nSpeeded"]
        * (len(list(version_cond_order.values())) // 3),
        rotation=45,
        size=12,
    )
    ax.yaxis.set_tick_params(labelsize=18)


def plot_avg_effectsize_dice_si(
    df: pd.DataFrame,
    network: str = "lang",
    hemi: str = "LH",
    save_str: str = "n=?",
    save: bool = True,
) -> None:
    """
    Plots the average Dice coefficient across and within localizer versions for a larger range of percentiles
    """
    print("*" * 80)
    df = df.copy()
    df = df.loc[df.Hemisphere == hemi]

    if network == "lang":
        df = df.loc[~df.ROI.str.contains("AngG")]  # exclude AngG from analysis

    ## POOL MANUAL APPROACH ##
    df_subjectwise = df.groupby(["Version", "UID"]).mean().reset_index()

    df_mean = (
        df_subjectwise.groupby(["Version"])
        .agg({"Coefficient": ["mean", "std", "count"]})
        .reset_index()
    )
    df_mean["sem"] = (
        df_mean["Coefficient"]["std"] / df_mean["Coefficient"]["count"] ** 0.5
    )

    if save:
        folder = get_folder(network, hemi)
        plot_dir = f"{PLOTDIR}/{folder}/dice_si_plots_{folder}"

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

    fig, ax = plt.subplots(figsize=(10, 6))

    _make_bargraph_dice_si(
        df_mean,
        df_subjectwise,
        version_cond_order=VERSION_COND_ORDER_DICE,
        ax=ax,
    )
    fig.tight_layout()

    # save plots
    if save:
        fig.savefig(
            join(
                plot_dir,
                f"{folder}-network-avg_{save_str}.png",
            )
        )
        fig.savefig(
            join(
                plot_dir,
                f"{folder}-network-avg_{save_str}.svg",
            )
        )
    plt.close(fig)


if __name__ == "__main__":
    ### LANG ###
    for network in ["lang", "wholebrain"]:
        for hemi in ["LH", "RH"]:
            df = pd.read_csv(join(DATADIR, f"spatial_corr_{network}.csv"), index_col=0)

            title_str_mean = f"Normal versus speeded langloc spatial correlation, mean across {hemi} lang fROIs, n=24"
            save_str_mean = f"n=24"
            plot_avg_effectsize(
                df,
                network=network,
                hemi=hemi,
                save_str=save_str_mean,
                save=True,
            )

        plot_avg_effectsize_dice_si(
            df,
            network=network,
            hemi="LH",
            save_str="n=24",
            save=True,
        )

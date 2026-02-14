# Plots the Effect Size per fROI
#   For each fROI per system:
#       group by langloc version, catergorized by condition (S vs. N)
#   Plot average effect across system

import argparse
import typing
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, ttest_1samp
import pandas as pd
import numpy as np
import os
from os.path import join
from constants import D_ROI_ORDER, D_COLOR_COND, DATADIR, PLOTDIR


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"


####### Helper Functions #######


def _get_folder(network: str, hemi: str) -> str:
    """
    Returns folder that plots are saved in
    network: lang or MD
    hemi: LH or RH (only used if network is lang)
    """
    return f"{hemi}_{network}" if network == "lang" else network


def _make_bargraph(
    df: pd.DataFrame,
    df_subjectwise: pd.DataFrame,
    title_str: str,
    version_cond_order: typing.Dict[str, int],
    ax: "plt.Axes",
    connect_subj_points: bool = False,
    colors: typing.List[str] = ["maroon", "darksalmon", "maroon", "darksalmon"],
) -> None:
    """
    Bars effect size for each version condition and scatters individual subject effect sizes within the specified (functionally localized) `network`
    """

    # set index according to version_cond_order
    df.index = df["Version_Condition"]
    df = df.reindex(version_cond_order)

    # make grid horizontal grid lines
    ax.yaxis.grid(alpha=0.4, zorder=0)

    # plot bar graphs for effect sizes of each version condition with standard error of the mean error bars
    ax.bar(
        df["Version_Condition"],
        df[("EffectSize", "mean")],
        yerr=df[("sem")],
        capsize=3,
        color=colors,
        alpha=0.6,
        zorder=2,
        edgecolor="black",
    )

    # plot points for each individual subject's effect size
    ind_points_arr = []
    for i, cond in enumerate(
        version_cond_order
    ):  # enumerate over version_cond in intended order, so errorbars and scatter matches
        ind_points = (df_subjectwise.loc[df_subjectwise["Version_Condition"] == cond])[
            ("EffectSize")
        ].values
        cond_len = len(ind_points)
        jitter_array = np.random.rand(cond_len) * (0.1) - 0.1 / 2
        ind_points_arr.append([i + jitter_array, ind_points])
        ax.scatter(i + jitter_array, ind_points, alpha=0.9, color="grey", s=15)

        if connect_subj_points and i % 2 == 1:
            for j in range(cond_len):
                plt.plot(
                    [ind_points_arr[-2][0][j], ind_points_arr[-1][0][j]],
                    [ind_points_arr[-2][1][j], ind_points_arr[-1][1][j]],
                    color="grey",
                    alpha=0.2,
                )

    # set other plot properties
    ax.axhline(y=0, color="black", lw=0.5)
    ax.yaxis.set_tick_params(labelsize=25)
    ax.xaxis.set_tick_params(labelsize=17)
    ax.set_title(title_str, fontsize=20)
    ax.set_ylabel("BOLD response\n(mean +- SEM across participants)", fontsize=16)


####### PLOT graphs #######


def plot_ROI_effectsize(
    df: pd.DataFrame,
    network: str = "lang",
    contrast: str = "S-N",
    title_str: str = "",
    save_str: str = "",
    hemi: str = "LH",
    save: bool = True,
) -> None:
    """
    For each ROI, plots the standard and speeded effect sizes for both sentences and non-words within the specified (functionally localized) `network`
    """
    print("######################################################################")
    print(f"###### PLOTTING ROI EFFECT SIZE FOR {network} {hemi} {contrast} ######")
    print("######################################################################")

    df_S_N = df.copy()
    contrasts_of_interest = contrast.split("-")
    df_S_N = df_S_N.loc[df_S_N["Condition"] != contrast]

    if (network == "MD" and contrast == "H-E") or (
        network == "dmn" and contrast == "E-H"
    ):
        version_cond_order = contrasts_of_interest
    else:
        version_cond_order = [
            f"Standard_{contrasts_of_interest[0]}",
            f"Standard_{contrasts_of_interest[1]}",
            f"Speeded_{contrasts_of_interest[0]}",
            f"Speeded_{contrasts_of_interest[1]}",
        ]

    # Obtain mean, sem, count statistics of the effect size across subjects
    df_grouped = (
        df_S_N.groupby(["Version_Condition", "ROI"])
        .agg({"EffectSize": ["mean", "std", "count"]})
        .reset_index()
    )
    df_grouped["sem"] = (
        df_grouped["EffectSize"]["std"] / df_grouped["EffectSize"]["count"] ** 0.5
    )
    folder = _get_folder(network, hemi)

    # Make plot for each ROI in network of interest
    row_counter = 0
    col_counter = 0
    if network == "lang":
        rows = 2
        cols = 3
        fig, axs = plt.subplots(rows, cols, figsize=(20, 10), sharey=True)
    elif network == "MD":
        rows = 4
        cols = 5
        fig, axs = plt.subplots(rows, cols, figsize=(34, 16), sharey=True)
    elif network == "dmn":
        rows = 3
        cols = 4
        fig, axs = plt.subplots(rows, cols, figsize=(28, 14), sharey=True)
    elif network == "extendedLang":
        rows = 4
        cols = 7
        fig, axs = plt.subplots(rows, cols, figsize=(52, 24))
    else:
        raise Exception(f"Network {network} not supported")

    for roi_idx, roi in enumerate(
        [
            roi_name for roi_name in D_ROI_ORDER[folder] if roi_name
        ]  # Extended lang parcel exclude ROI 9
    ):  # iterate over ROIs in intended order

        if network == "lang" and "AngG" in roi:  # exclude AngG from analysis
            continue
        df_grouped_preagg = df_S_N.loc[df_S_N["ROI"] == roi]
        df_plot = df_grouped.loc[df_grouped["ROI"] == roi]

        _make_bargraph(
            df_plot,
            df_grouped_preagg,
            roi,
            version_cond_order,
            axs[row_counter, col_counter],
            colors=D_COLOR_COND[f"{network}_{contrast}"],
        )

        if (roi_idx + 1) % cols == 0:
            row_counter += 1
            col_counter = 0
        else:
            col_counter += 1

    fig.suptitle(f"{title_str}", fontsize=15)
    fig.tight_layout()

    # Define paths and save plots
    if save:
        PLOT_DIR = f"{PLOTDIR}/{folder}/effectsize_plots_{folder}/{contrast}"

        if not os.path.exists(PLOT_DIR):
            os.makedirs(PLOT_DIR)

        fig.savefig(join(PLOT_DIR, f"{folder}-{contrast}-ROIs_{save_str}.png"))
        fig.savefig(join(PLOT_DIR, f"{folder}-{contrast}-ROIs_{save_str}.svg"))

    plt.close(fig)


# Plots a graph from a dataframe
def plot_avg_effectsize(
    df: pd.DataFrame,
    network: str = "lang",
    contrast: str = "S-N",
    title_str: str = "",
    save_str: str = "",
    hemi: str = "LH",
    save: bool = True,
) -> None:
    """
    Plots the standard and speeded effect sizes for both sentences and non-words averaged across ROIs within the specified (functionally localized) `network`
    """
    print("######################################################################")
    print(f"###### PLOTTING AVG EFFECT SIZE FOR {network} {hemi} {contrast} ######")
    print("######################################################################")
    df = df.copy()

    # set Version_Condition to contain both the language localizer version (standard vs speeded) and the localizer condition (sentence vs nonword)
    # For MD, average LH and RH separately
    contrasts = contrast.split("-")
    connect_subj_points = False
    if network == "lang" or network == "extendedLang":
        version_cond_order = [
            f"Standard_{contrasts[0]}",
            f"Standard_{contrasts[1]}",
            f"Speeded_{contrasts[0]}",
            f"Speeded_{contrasts[1]}",
        ]
        df = df.loc[
            ~df["ROI"].isnull()
        ]  # exclude null ROIs (extendedLang excludes parcel 9)
        df = df.loc[~df["ROI"].str.contains("AngG")]  # exclude AngG from analysis
    elif (network == "MD" or network == "dmn") and contrast == "S-N":
        if hemi is None:
            version_cond_order = [
                f"Standard_{contrasts[0]}",
                f"Standard_{contrasts[1]}",
                f"Speeded_{contrasts[0]}",
                f"Speeded_{contrasts[1]}",
            ]
            fig, ax = plt.subplots(figsize=(7, 5))
        elif hemi == "LH" or hemi == "RH":
            version_cond_order = [
                f"Standard_{contrasts[0]}\n{hemi}",
                f"Standard_{contrasts[1]}\n{hemi}",
                f"Speeded_{contrasts[0]}\n{hemi}",
                f"Speeded_{contrasts[1]}\n{hemi}",
            ]
            df["Version_Condition"] = df["Version_Condition"] + "\n" + df["Hemisphere"]
            save_str += f"_{hemi}"
            connect_subj_points = True

    elif (network == "MD" and contrast == "H-E") or (
        network == "dmn" and contrast == "E-H"
    ):
        version_cond_order = [
            f"{contrasts[0]}_LH",
            f"{contrasts[1]}_LH",
            f"{contrasts[0]}_RH",
            f"{contrasts[1]}_RH",
        ]

        df["Version_Condition"] = df["Version_Condition"] + "_" + df["Hemisphere"]

    else:
        raise Exception(f"Network {network} not supported")

    # Get the mean sentence and nonword effect sizes value across ROIs for each subject for both the standard and speeded versions of the localizer task
    df_subjectwise = df.groupby(["Version_Condition", "UID"]).mean()
    df_subjectwise = df_subjectwise.reset_index()

    # Compare H and E for standard and speeded
    if network == "lang" and contrast == "H-E":

        for version_condition in df["Version_Condition"].unique():
            t_stat, p = ttest_1samp(
                df_subjectwise[
                    df_subjectwise["Version_Condition"] == version_condition
                ]["EffectSize"],
                0,
            )
            print(
                f"T-test for comparing to baseline (0) for {hemi} {network} {version_condition} n={len(df_subjectwise[df_subjectwise['Version_Condition'] == version_condition])}: t={t_stat}, p={p}"
            )

        t_stat, p = ttest_ind(
            df_subjectwise.loc[df_subjectwise["Version_Condition"] == f"Standard_H"][
                "EffectSize"
            ],
            df_subjectwise.loc[df_subjectwise["Version_Condition"] == f"Standard_E"][
                "EffectSize"
            ],
        )
        print(
            f"ttest t-stat for Standard H-E {_get_folder(network, hemi)}: {t_stat}, p: {p}"
        )

        t_stat, p = ttest_ind(
            df_subjectwise.loc[df_subjectwise["Version_Condition"] == f"Speeded_H"][
                "EffectSize"
            ],
            df_subjectwise.loc[df_subjectwise["Version_Condition"] == f"Speeded_E"][
                "EffectSize"
            ],
        )
        print(
            f"ttest t-stat for Speeded H-E {_get_folder(network, hemi)}: {t_stat}, p: {p}"
        )

    # Obtain mean, sem, count statistics of the effect size across subjects
    df_mean = (
        df_subjectwise.groupby(["Version_Condition"])
        .agg({"EffectSize": ["mean", "std", "count"]})
        .reset_index()
    )
    df_mean["sem"] = (
        df_mean["EffectSize"]["std"] / df_mean["EffectSize"]["count"] ** 0.5
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    print(df_mean)
    _make_bargraph(
        df_mean,
        df_subjectwise,
        title_str,
        version_cond_order,
        ax,
        connect_subj_points=connect_subj_points,
        colors=D_COLOR_COND[f"{network}_{contrast}"],
    )
    fig.tight_layout()

    # Define paths and save plots
    if save:
        folder = _get_folder(network, hemi)
        PLOT_DIR = f"{PLOTDIR}/{folder}/effectsize_plots_{folder}/{contrast}"

        fig.savefig(join(PLOT_DIR, f"{folder}-network-{contrast}-avg_{save_str}.png"))
        fig.savefig(join(PLOT_DIR, f"{folder}-network-{contrast}-avg_{save_str}.svg"))

    plt.close(fig)


# Plots S-N graph from a dataframe
def plot_contrast_effectsize(
    df: pd.DataFrame,
    network: str = "lang",
    contrast: str = "S-N",
    title_str: str = "Standard versus speeded langloc S-N contrast, mean across LH lang fROIs",
    save_str: str = "n=?",
    hemi: str = "LH",
    save: bool = True,
) -> None:
    """
    Plots the standard and speeded effect sizes for sentences > non-words averaged across ROIs within the specified (functionally localized) `network`
    """

    df = df.copy()

    # set Version_Condition to contain the localizer condition (sentence vs nonword)
    # For MD, average LH and RH separately
    if network == "lang" or network == "extendedLang":
        version_cond_order = [f"Standard_{contrast}", f"Speeded_{contrast}"]

        df = df.loc[
            ~df["ROI"].isnull()
        ]  # exclude null ROIs (extendedLang excludes parcel 9)
        df = df.loc[~df["ROI"].str.contains("AngG")]  # exclude AngG from analysis
    elif (network == "MD" or network == "dmn") and contrast == "S-N":
        version_cond_order = [
            f"Standard_{contrast}_LH",
            f"Standard_{contrast}_RH",
            f"Speeded_{contrast}_LH",
            f"Speeded_{contrast}_RH",
        ]
        df["Version_Condition"] = df["Version_Condition"] + "_" + df["Hemisphere"]
    elif (network == "MD" and contrast == "H-E") or (
        network == "dmn" and contrast == "E-H"
    ):
        version_cond_order = [f"{contrast}_LH", f"{contrast}_RH"]
        df["Version_Condition"] = df["Version_Condition"] + "_" + df["Hemisphere"]
    else:
        raise Exception(f"Network {network} not supported")

    df_SN = df.copy()
    df_SN = df_SN.loc[df_SN["Condition"] == contrast]

    # Get the mean sentence > nonword effect size value across ROIs for each subject for both the standard and speeded versions of the localizer task
    df_subjectwise = df_SN.groupby(["Version_Condition", "UID"]).mean()
    df_subjectwise = df_subjectwise.reset_index()

    # Obtain mean, sem, count statistics of the effect size across subjects
    df_mean = (
        df_subjectwise.groupby(["Version_Condition"])
        .agg({"EffectSize": ["mean", "std", "count"]})
        .reset_index()
    )
    df_mean["sem"] = (
        df_mean["EffectSize"]["std"] / df_mean["EffectSize"]["count"] ** 0.5
    )

    # Perform t-test to determine if there is a significant difference between the sentence > nonword effect between standard and speeded version of the localzier
    if network == "lang":
        t_stat, p = ttest_ind(
            df_subjectwise.loc[
                df_subjectwise["Version_Condition"] == f"Standard_{contrast}"
            ]["EffectSize"],
            df_subjectwise.loc[
                df_subjectwise["Version_Condition"] == f"Speeded_{contrast}"
            ]["EffectSize"],
        )
        print(
            f"ttest t-stat for {contrast} (standard vs. speeded) {_get_folder(network, hemi)}: {t_stat}, p: {p}"
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    _make_bargraph(
        df_mean,
        df_subjectwise,
        title_str,
        version_cond_order,
        ax,
        colors=D_COLOR_COND[f"{network}_{contrast}"],
    )
    fig.tight_layout()

    # Define paths and make plots
    if save:
        folder = _get_folder(network, hemi)
        PLOT_DIR = f"{PLOTDIR}/{folder}/effectsize_plots_{folder}/{contrast}"
        fig.savefig(join(PLOT_DIR, f"{folder}-network-{contrast}_{save_str}.png"))
        fig.savefig(join(PLOT_DIR, f"{folder}-network-{contrast}_{save_str}.svg"))

    plt.close(fig)


def load_csvs(
    network: str, contrast: str, hemi: typing.Optional[str] = None
) -> pd.DataFrame:
    """
    Load the csvs for the given `network` and `contrast` in hempisphere `hemi`
    """

    df = pd.read_csv(join(DATADIR, f"mROI_{network}_{contrast}.csv"), index_col=0)
    if hemi is not None:
        df = df.loc[df.Hemisphere == hemi]
    return df


if __name__ == "__main__":
    ###### PLOT LANG network (separately for LH and RH ROIs) ######

    network = "lang"
    for hemi in ["LH", "RH"]:
        contrast = "S-N"

        df_lang = load_csvs(network, contrast, hemi)
        title_str_roi = f"Standard vs. speeded language localizer effect size across {hemi} lang fROIs, n={len(np.unique(list(df_lang.UID)))}"
        save_str_roi = f"n={len(np.unique(list(df_lang.UID)))}"
        title_str_mean = f"Standard vs. speeded language localizer effect size, mean across {hemi} lang fROIs, n={len(np.unique(list(df_lang.UID)))}"
        save_str_mean = f"n={len(np.unique(list(df_lang.UID)))}"

        plot_ROI_effectsize(
            df_lang,
            network=network,
            contrast=contrast,
            title_str=title_str_roi,
            save_str=save_str_roi,
            hemi=hemi,
            save=True,
        )
        plot_avg_effectsize(
            df_lang,
            network=network,
            contrast=contrast,
            title_str=title_str_mean,
            save_str=save_str_mean,
            hemi=hemi,
            save=True,
        )
        plot_contrast_effectsize(
            df_lang,
            network=network,
            contrast=contrast,
            title_str=title_str_mean,
            save_str=save_str_mean,
            hemi=hemi,
            save=True,
        )

        contrast = "H-E"
        df_lang = load_csvs(network, contrast, hemi)

        title_str_roi = f"Standard vs. speeded MD localizer effect size across {hemi} lang fROIs, n={len(np.unique(list(df_lang.UID)))}"
        save_str_roi = f"n={len(np.unique(list(df_lang.UID)))}"
        title_str_mean = f"Standard vs. speeded MD localizer effect size, mean across {hemi} lang fROIs, n={len(np.unique(list(df_lang.UID)))}"
        save_str_mean = f"n={len(np.unique(list(df_lang.UID)))}"

        plot_ROI_effectsize(
            df_lang,
            network=network,
            contrast=contrast,
            title_str=title_str_roi,
            save_str=save_str_roi,
            hemi=hemi,
            save=True,
        )
        plot_avg_effectsize(
            df_lang,
            network=network,
            contrast=contrast,
            title_str=title_str_mean,
            save_str=save_str_mean,
            hemi=hemi,
            save=True,
        )
        plot_contrast_effectsize(
            df_lang,
            network=network,
            contrast=contrast,
            title_str=title_str_mean,
            save_str=save_str_mean,
            hemi=hemi,
            save=True,
        )

    ###### PLOT MD Network ######

    network = "MD"
    contrast = "S-N"

    df_MD = load_csvs(network, contrast, hemi=None)

    for hemi in ["LH", "RH", None]:
        title_str_roi = f"Responses to sentences and nonwords in the {hemi if hemi is not None else 'LH and RH'} multiple demand (MD) network"
        save_str_roi = f"n={len(np.unique(list(df_MD.UID)))}"
        title_str_mean = f"Responses to sentences and nonwords in the {hemi if hemi is not None else 'LH and RH'} multiple demand (MD) network"
        save_str_mean = f"n={len(np.unique(list(df_MD.UID)))}"

        plot_ROI_effectsize(
            df_MD,
            network=network,
            contrast=contrast,
            title_str=title_str_roi,
            save_str=save_str_roi,
            hemi=hemi,
            save=True,
        )
        plot_avg_effectsize(
            df_MD,
            network=network,
            contrast=contrast,
            title_str=title_str_mean,
            save_str=save_str_mean,
            hemi=hemi,
            save=True,
        )
        plot_contrast_effectsize(
            df_MD,
            network=network,
            contrast=contrast,
            title_str=title_str_mean,
            save_str=save_str_mean,
            hemi=hemi,
            save=True,
        )

    contrast = "H-E"
    df_MD = load_csvs(network, contrast, hemi=None)

    title_str_roi = f"Hard vs. easy SpatialFIN effect size across MD fROIs, n={len(np.unique(list(df_MD.UID)))}"
    save_str_roi = f"n={len(np.unique(list(df_MD.UID)))}"
    title_str_mean = f"Hard vs. easy SpatialFIN effect size, mean across MD fROIs, n={len(np.unique(list(df_MD.UID)))}"
    save_str_mean = f"n={len(np.unique(list(df_MD.UID)))}"

    plot_ROI_effectsize(
        df_MD,
        network=network,
        contrast=contrast,
        title_str=title_str_roi,
        save_str=save_str_roi,
        hemi=None,
        save=True,
    )
    plot_avg_effectsize(
        df_MD,
        network=network,
        contrast=contrast,
        title_str=title_str_mean,
        save_str=save_str_mean,
        hemi=None,
        save=True,
    )
    plot_contrast_effectsize(
        df_MD,
        network=network,
        contrast=contrast,
        title_str=title_str_mean,
        save_str=save_str_mean,
        hemi=None,
        save=True,
    )

    ###### PLOT DMN Network ######
    network = "dmn"
    contrast = "S-N"

    df_dmn = load_csvs(network, contrast, None)
    title_str_roi = f"Standard vs. speeded language localizer effect size across DMN fROIs, n={len(np.unique(list(df_dmn.UID)))}"
    save_str_roi = f"n={len(np.unique(list(df_dmn.UID)))}"
    title_str_mean = f"Standard vs. speeded language localizer effect size, mean across DMN fROIs, n={len(np.unique(list(df_dmn.UID)))}"
    save_str_mean = f"n={len(np.unique(list(df_dmn.UID)))}"

    plot_ROI_effectsize(
        df_dmn,
        network=network,
        contrast=contrast,
        title_str=title_str_roi,
        save_str=save_str_roi,
        hemi=None,
        save=True,
    )
    plot_avg_effectsize(
        df_dmn,
        network=network,
        contrast=contrast,
        title_str=title_str_mean,
        save_str=save_str_mean,
        hemi=None,
        save=True,
    )
    plot_contrast_effectsize(
        df_dmn,
        network=network,
        contrast=contrast,
        title_str=title_str_mean,
        save_str=save_str_mean,
        hemi=None,
        save=True,
    )

    contrast = "E-H"
    df_dmn = load_csvs(network, contrast, None)

    title_str_roi = f"Easy vs. hard SpatialFIN effect size across DMN fROIs, n={len(np.unique(list(df_dmn.UID)))}"
    save_str_roi = f"n={len(np.unique(list(df_dmn.UID)))}"
    title_str_mean = f"Easy vs. hard SpatialFIN effect size, mean across DMN fROIs, n={len(np.unique(list(df_dmn.UID)))}"
    save_str_mean = f"n={len(np.unique(list(df_dmn.UID)))}"

    plot_ROI_effectsize(
        df_dmn,
        network=network,
        contrast=contrast,
        title_str=title_str_roi,
        save_str=save_str_roi,
        hemi=None,
        save=True,
    )
    plot_avg_effectsize(
        df_dmn,
        network=network,
        contrast=contrast,
        title_str=title_str_mean,
        save_str=save_str_mean,
        hemi=None,
        save=True,
    )
    plot_contrast_effectsize(
        df_dmn,
        network=network,
        contrast=contrast,
        title_str=title_str_mean,
        save_str=save_str_mean,
        hemi=None,
        save=True,
    )

    ###### PLOT Extended Lang Network ######
    network = "extendedLang"
    contrast = "S-N"

    df_extendedLang = load_csvs(network, contrast, None)
    title_str_roi = f"Standard vs. speeded language localizer effect size across Extended Lang fROIs, n={len(np.unique(list(df_extendedLang.UID)))}"
    save_str_roi = f"n={len(np.unique(list(df_extendedLang.UID)))}"
    title_str_mean = f"Standard vs. speeded language localizer effect size, mean across Extended Lang fROIs, n={len(np.unique(list(df_extendedLang.UID)))}"
    save_str_mean = f"n={len(np.unique(list(df_extendedLang.UID)))}"

    plot_ROI_effectsize(
        df_extendedLang,
        network=network,
        contrast=contrast,
        title_str=title_str_roi,
        save_str=save_str_roi,
        hemi=None,
        save=True,
    )
    plot_avg_effectsize(
        df_extendedLang,
        network=network,
        contrast=contrast,
        title_str=title_str_mean,
        save_str=save_str_mean,
        hemi=None,
        save=True,
    )
    plot_contrast_effectsize(
        df_extendedLang,
        network=network,
        contrast=contrast,
        title_str=title_str_mean,
        save_str=save_str_mean,
        hemi=None,
        save=True,
    )

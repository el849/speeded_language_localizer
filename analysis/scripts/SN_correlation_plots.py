import typing
import pandas as pd
import numpy as np
from os.path import join
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import scipy as sp
from constants import DATADIR, PLOTDIR


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
sns.set_style("whitegrid")

EXCLUDE_SPEEDED = ["870_FED_20220218c_3T1_PL2017"]  # outlier participant


def get_dataframe_even_odd(exclude_outlier: bool = False) -> pd.DataFrame:
    """
    :param exclude_outlier whether to exclude the outlier
    :return the loaded dataframe of response values
    """
    df = pd.read_csv(join(DATADIR, "mROI_lang_S-N_evenodd.csv"))
    df = df.loc[df.Hemisphere == "LH"]

    if exclude_outlier:
        df = df.loc[~df["Subject"].isin(EXCLUDE_SPEEDED)]

    return df


def plot_even_odd_corr(
    df_roi: pd.DataFrame, ax: "plt.Axes", lim: typing.List[int, int]
) -> float:
    """
    :param df_roi the dataframe of response values to plot
    :param ax the axes on which to plot
    :param lim the axes boundaries
    :return the pearonsr correlation of the odd and even runs of the given task. Also scatterplots the even and odd runs from `df_roi` on `ax`
    """
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    ax.set_aspect("equal")
    ax.scatter(df_roi["EffectSize_ODD"], df_roi["EffectSize_EVEN"], s=5, c="black")

    pearsonr, p = sp.stats.pearsonr(df_roi["EffectSize_ODD"], df_roi["EffectSize_EVEN"])
    ax.text(
        0.05,
        0.8,
        "r={:.2f}\np={:.2g}".format(pearsonr, p),
        transform=ax.transAxes,
        fontsize=8,
    )
    return pearsonr


def plot_even_odd_corr_all_ROIs(
    df_standard: pd.DataFrame,
    df_speeded: pd.DataFrame,
    exclude_outlier: bool = False,
) -> None:
    """
    Plots thec correlation between even and odd runs of a localizer version
    :param df_standard DataFrame containing responses to the standard localizer
    :param df_speeded DataFrame containing the responses to the speeded localizer
    :param exclude_outlier whether to exclude outliers in the plotting
    """
    ROIs = df_standard["ROI"].unique()
    ROWS = 2
    COLS = len(ROIs) + 1
    fig, axes = plt.subplots(ROWS, COLS, figsize=(16, 5.2))

    all_values = (
        list(df_standard["EffectSize_ODD"])
        + list(df_speeded["EffectSize_EVEN"])
        + list(df_standard["EffectSize_ODD"])
        + list(df_speeded["EffectSize_EVEN"])
    )
    lim = [np.min(all_values) - 0.1, np.max(all_values) + 0.1]

    for r in range(ROWS):
        df = df_standard if r == 0 else df_speeded
        pearson_r_values = []
        for c, roi in enumerate(ROIs):
            df_roi = df.loc[df["ROI"] == roi]
            pearsonr = plot_even_odd_corr(df_roi, axes[r, c + 1], lim)
            pearson_r_values.append(pearsonr)
        if r == 0:
            print(
                f"Standard mean pearsonr: {np.mean(pearson_r_values)}, median: {np.median(pearson_r_values)}, std: {np.std(pearson_r_values, ddof=1)}"
            )
        elif r == 1:
            print(
                f"Speeded mean pearsonr: {np.mean(pearson_r_values)}, median: {np.median(pearson_r_values)}, std: {np.std(pearson_r_values, ddof=1)}"
            )

    for ax, col in zip(axes[0], ["LH_Lang"] + list(ROIs)):
        ax.set_title(col)

    for ax, row in zip(axes[:, 0], ["Standard", "Speeded"]):
        ax.set_ylabel(row, size="large")

    plot_even_odd_corr_avg(df_standard, df_speeded, axes[:, 0], lim)

    fig.supxlabel("Odd Run Effect Size", fontsize=14, x=0.5, y=0.01)
    fig.supylabel("Even Run Effect Size", fontsize=14, x=0.01, y=0.5)

    fig.tight_layout()
    plt.savefig(
        join(
            PLOTDIR,
            f"lang_SN_EVEN_ODD_corr_excludeoutlier={exclude_outlier}.png",
        )
    )
    plt.savefig(
        join(
            PLOTDIR,
            f"lang_SN_EVEN_ODD_corr_excludeoutlier={exclude_outlier}.svg",
        )
    )


def plot_even_odd_corr_avg(df_standard, df_speeded, axes, lim):
    """
    Plots thec correlation between even and odd runs of the standard and speeded localizers
    """
    for r in range(2):
        df = df_standard if r == 0 else df_speeded
        df = df.groupby("Subject").mean()
        plot_even_odd_corr(df, axes[r], lim)


def main():
    for exclude_outlier in [True, False]:
        df = get_dataframe_even_odd(exclude_outlier)
        df_standard = df.loc[df.Version == "Standard"]
        df_speeded = df.loc[df.Version == "Speeded"]
        plot_even_odd_corr_all_ROIs(df_standard, df_speeded, exclude_outlier)

        df = get_dataframe_even_odd(exclude_outlier)
        df_standard = df.loc[df.Version == "Standard"]
        df_speeded = df.loc[df.Version == "Speeded"]
        plot_even_odd_corr_all_ROIs(df_standard, df_speeded, exclude_outlier)


if __name__ == "__main__":
    main()

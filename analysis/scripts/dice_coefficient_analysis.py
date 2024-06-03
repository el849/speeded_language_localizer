import os
import typing
import copy
from tqdm import tqdm
from collections import namedtuple, defaultdict

import h5py
import numpy as np
import pandas as pd
from scipy import io
import nibabel as nib

from constants import (
    FIRSTLEVELDIR,
    FL_CONTRAST,
    ROIDIR,
    D_ROI_ORDER,
    UID_LIST,
    DICERESULTDIR,
)

### Define more constants for this analysis
ODD = "ODD_S-N"
EVEN = "EVEN_S-N"
ALLRUNS = "S-N"
STANDARD = "langlocSN"
SPEEDED = "langlocSN_speeded"

NETWORKS = ["lang", "wholebrain"]
THRESHOLDS_TO_COMPUTE = [60, 65, 70, 75, 80, 85, 90, 95]


DiceParams = namedtuple(
    "DiceParams",
    [
        "identifier",
        "descrip",
        "version1",
        "version2",
        "contrast1",
        "contrast2",
        "thresh",
    ],
)


## Helper functions to load files


def _load_nii_flat(file: str) -> "np.ndarray":
    """
    :param file: filename of nifti to load
    :return: flattened nifti array

    """
    return np.array(nib.load(file).dataobj).flatten()


def _get_spmT_file_from_contrast(
    subid: str, firstlevel: str = "langlocSN", contrast: str = "S-N"
) -> str:
    """
    :param subid: the subject UID
    :param firstlevel: the experiment to use (langlocSN, langlocSN_speeded)
    :return: path corresponding to where the firstlevel files live
    """
    firstlevel_path = f"{FIRSTLEVELDIR}/firstlevel_{subid}/{firstlevel}"

    try:
        with h5py.File(os.path.join(firstlevel_path, "SPM.mat"), "r") as f:
            loaded_file = f["/SPM/xCon/name"]
            for i in range(len(loaded_file)):
                final_str = ""
                for j in range(len(f[loaded_file[i][0]])):
                    final_str += chr(f[loaded_file[i][0]][j][0])

                if final_str == contrast:
                    xcon_number = i + 1
                    return os.path.join(
                        firstlevel_path, f"spmT_{'{:04d}'.format(xcon_number)}.nii"
                    )
            else:
                raise Exception(f"Contrast {contrast} is not in firstlevel")
    except:
        loaded_file = io.loadmat(
            os.path.join(firstlevel_path, "SPM.mat"), struct_as_record=False
        )["SPM"][0, 0].xCon[0]
        for i in range(len(loaded_file)):
            final_str = loaded_file[i].name[0]
            if final_str == contrast:
                xcon_number = i + 1
                return os.path.join(
                    firstlevel_path, f"spmT_{'{:04d}'.format(xcon_number)}.nii"
                )
        else:
            raise Exception(f"Contrast {contrast} is not in firstlevel")


def get_thresh(samples: np.ndarray, thresh: int = 90) -> float:
    """ "Get percentile value, and take all values that are above that"""
    return np.percentile(samples, int(thresh))


## Functions to compute the dice coefficient


def _get_roi_masks(
    network: np.ndarray,
    parc: np.ndarray,
    netw_of_interest: str = "lang",
    rois_of_interest: typing.Union[None, list, np.ndarray] = None,
    func_threshold: int = 90,
    d_parcel_name_map: dict = None,
) -> typing.Tuple["np.ndarray", "np.ndarray"]:
    """

    :param network: ndarray, the first level contrast masked with parcels. Not thresholding masked.
    :param parc: ndarray, parcel denotations. Not thresholding masked.
    :param netw_of_interest: string
    :param rois_of_interest: None (if taking all) or list of strings, denoting which ROIs to fetch
    :param func_threshold: int
    :param d_parcel_name_map: dict, mapping parcel number to parcel name

    :return:
            mask_roi_all_netw_str: ndarray of strings, denoting ROI name
            mask_entire_netw: ndarray boolean, denoting where the entire lang network is
                    Shape of the the mask_* ndarrays are of length (91*109*91)=902629
    """

    mask_roi_all_netw = np.zeros(network.shape).astype(
        int
    )  # for storing masks for every ROI
    mask_entire_netw = np.zeros(network.shape)  # for storing the entire network

    for roi in np.unique(parc):
        roi = int(roi)
        if roi > 0:
            roi_name = d_parcel_name_map[roi]
            if rois_of_interest:
                if not roi_name in rois_of_interest:
                    continue  # do not extract voxels

            samples = network[(parc == roi) & (~(np.isnan(network)))]
            percentile_val_used = get_thresh(samples, func_threshold)

            # If percentile is 0 or below, it means that this contrast was not significant for this ROIm extract voxels with t-values > 0
            if percentile_val_used <= 0:
                roi_voxels = (parc == roi) & (network >= 0)
            else:
                roi_voxels = (parc == roi) & (
                    network >= get_thresh(samples, thresh=func_threshold)
                )

            mask_entire_netw += roi_voxels
            mask_roi_all_netw += roi_voxels * roi

        mask_roi_all_netw_str = copy.deepcopy(mask_roi_all_netw).astype(str)

    # Add the roi names into the mask_roi_all_netw_str array (converting the integers to strings)
    for roi in np.unique(parc):
        if roi > 0:
            roi_name = d_parcel_name_map[int(roi)]
            mask_roi_all_netw_str[mask_roi_all_netw == int(roi)] = (
                f"{netw_of_interest}_{roi_name}"
            )
    mask_entire_netw = mask_entire_netw.astype(bool)

    return mask_roi_all_netw_str, mask_entire_netw


def _get_fl_in_parcel(
    ROIDIR: str,
    spmT_file: str,
    network: str,
) -> typing.Tuple["np.ndarray", "np.ndarray"]:
    """
    :param ROIDIR directory containing the parcels
    :param spmT_file the filename for the contrast file of interest
    :param network the name of the parcel  to load (`lang`, `wholebrain`, `md`)
    :return network: ndarray of float, the voxel t-values for voxels belonging in the parcel
            parc: ndarray of int, the parcel containing integers denoting the ROIs
    """
    PATH_parc = f"{ROIDIR}/{network}.nii"
    parc = np.rint(_load_nii_flat(PATH_parc))

    network = _load_nii_flat(spmT_file)
    # return the first level network where there is a parcel
    return network * (parc > 0), parc


def _get_voxels_helper(netw_of_interest, fl, spmT_file, func_threshold, UID_SESSION):
    """
    helper function for `get_voxels()`
    :param netw_of_interest: str which parcel to use (lang, md, dmn, aud, vis)
    :param fl: str which firstlevel to use - firstlevel_{fl} must exist in SUBJECTDIR (ie. langlocSN)
    :param contrast_num: int which firstlevel contrast to use
    :param func_threshold: int
    :param UID_SESSION: str
    :return mask_roi: ndarray of strings, denoting ROI name, '0' if the voxel does not belong to an ROI
            mask_netw: ndarray boolean, denoting where the entire `network_of_interest` is
                    Shape of the the mask_* ndarrays are of length (91*109*91)=902629. If False, then the mask_* ndarrays are of length are subtracted with
                    the number of voxels in the thresh_mask.
    """
    # load parcels for netw_of_interest and return first-level netw_of_interest masked with parcel
    parc_fl, parc = _get_fl_in_parcel(
        ROIDIR=ROIDIR, spmT_file=spmT_file, network=netw_of_interest
    )

    # get netw_of_interest contrast in ROIs
    roi_name_map = dict(
        zip(
            list(range(1, len(D_ROI_ORDER[netw_of_interest]) + 1)),
            D_ROI_ORDER[netw_of_interest],
        )
    )
    mask_roi, mask_netw = _get_roi_masks(
        network=parc_fl,
        parc=parc,
        netw_of_interest=netw_of_interest,
        func_threshold=func_threshold,
        d_parcel_name_map=roi_name_map,
    )

    # 902629 (total voxels)
    # 235539 (thresholded with brain)
    # 13588 (size of lang parcel)
    # masks_roi: length 902629, 1364 non-zero values
    # mask_netw: length 902629, 1364 True values -- top 10 % in lang parcel

    return mask_roi, mask_netw


def get_overlap_area(voxels_1, voxels_2):
    """
    :param voxels_1: ndarray boolean, denoting where voxels are in run 1 to compare
    :param_voxels_2: ndarray boolean, denoting where voxels are in run 2 to compare
    :return: overlap_area: ndarray (1,0), denoting intersection of `voxels_1` and `voxels_2`
             total_covered_area: ndarray (1,0), denoting union of `voxels_1` and `voxels_2`
             one_only_area: ndarray (1,0), denoting where `voxels_1` and not `voxels_2`
             two_only_area: ndarray (1,0), denoting where `voxels_2` and not `voxels_1`
    """
    overlap_area = np.where(voxels_1, 1, 0) * np.where(
        voxels_2, 1, 0
    )  # np.where returns 1 where True and thus by multiplying we obtain the voxels where both sessions were selected
    total_covered_area = np.where(
        np.where(voxels_1, 1, 0) + np.where(voxels_2, 1, 0) > 0, 1, 0
    )
    one_only_area = np.where(
        np.where(voxels_1, 1, 0) - np.where(voxels_2, 1, 0) > 0, 1, 0
    )
    two_only_area = np.where(
        np.where(voxels_2, 1, 0) - np.where(voxels_1, 1, 0) > 0, 1, 0
    )
    return overlap_area, total_covered_area, one_only_area, two_only_area


def dice_coefficient(
    roi_voxels: "np.ndarray", fl_voxels: "np.ndarray"
) -> typing.Dict[str, float]:
    """
    :param roi_voxels: arr containing two (for each run to compare) ndarray of strings, denoting ROI name
    :param fl_voxels: arr containing two (for each run to compare) ndarray boolean, denoting where regions to
                      compare overlaps for are
    :return: The Dice coefficient values. Returns dictionary mapping each ROI to the Dice coefficient values
    """
    dice_coefficients = {}

    assert np.sum(np.where((roi_voxels == "0") & fl_voxels[0])) == 0
    assert np.sum(np.where((roi_voxels == "0") & fl_voxels[1])) == 0

    overlap_area, _, _, _ = get_overlap_area(fl_voxels[0], fl_voxels[1])

    print("overlap area: ", np.sum(overlap_area))
    print("1 area: ", np.sum(np.where(fl_voxels[0], 1, 0)))
    print("2 area: ", np.sum(np.where(fl_voxels[1], 1, 0)))

    dice_coefficients["all_rois"] = (
        2
        * np.sum(overlap_area)
        / (np.sum(np.where(fl_voxels[0], 1, 0)) + np.sum(np.where(fl_voxels[1], 1, 0)))
    )

    for roi in np.unique(np.array(roi_voxels)):
        if roi == "0":
            continue

        overlap_area, _, _, _ = get_overlap_area(
            (roi_voxels[0] == roi) & (fl_voxels[0]),
            (roi_voxels[1] == roi) & (fl_voxels[1]),
        )
        dice_coefficients[roi] = (
            2
            * np.sum(overlap_area)
            / (
                np.sum(np.where((fl_voxels[0]) & (roi_voxels[0] == roi), 1, 0))
                + np.sum(np.where((fl_voxels[1]) & (roi_voxels[0] == roi), 1, 0))
            )
        )
    return dice_coefficients


def get_dice_coefficients(
    network: str, dice_params: typing.Type[DiceParams]
) -> typing.Dict[str, typing.Any]:
    """
    :param network the network to compute the Dice Coeffients for
    :param dice_params the parameters to use in the Dice computation. See NamedTuple `DiceParams` (includes localizer version, threshold value)
    :return dictionary mapping params to the computed Dice coefficients
    """
    dice_dict = {
        "ROI": [],
        "Subject": [],
        "Effects": [],
        "Identifier": [],
        "Identifier": [],
        "Version": [],
        "UID": [],
        "Dice Coefficient": [],
    }

    for i in tqdm(range(len(UID_LIST))):
        contrast1 = _get_spmT_file_from_contrast(
            UID_LIST[i], dice_params.version1, dice_params.contrast1
        )

        contrast2 = _get_spmT_file_from_contrast(
            UID_LIST[i], dice_params.version2, dice_params.contrast2
        )
        roi_voxels_1, fl_voxels_1 = _get_voxels_helper(
            network,
            dice_params.version1,
            contrast1,
            dice_params.thresh,
            UID_LIST[i],
        )
        roi_voxels_2, fl_voxels_2 = _get_voxels_helper(
            network,
            dice_params.version2,
            contrast2,
            dice_params.thresh,
            UID_LIST[i],
        )

        dice = dice_coefficient(
            [roi_voxels_1, roi_voxels_2], [fl_voxels_1, fl_voxels_2]
        )

        for k, v in dice.items():
            if (
                k == "all_rois"
            ):  # don't include the entire combined parcel in the output
                continue

            dice_dict["ROI"].append(k)
            dice_dict["Subject"].append(i + 1)
            dice_dict["Effects"].append(
                f"{dice_params.contrast1}/{dice_params.contrast2}"
            )
            dice_dict["Identifier"].append(dice_params.identifier)
            dice_dict["Version"].append(dice_params.descrip)
            dice_dict["UID"].append(UID_LIST[i])
            dice_dict["Dice Coefficient"].append(v)
    return dice_dict


## Functions for computing the number of voxels above the threshold value


def get_number_significant_voxels(
    spmT_file: np.ndarray,
    netw_of_interest: str = "lang",
    rois_of_interest: typing.Union[None, list, np.ndarray] = None,
    func_threshold: int = 90,
    d_parcel_name_map: typing.Dict[int, str] = None,
) -> int:
    """
    :param spmT_file firstlevel file to use
    :param netw_of_interest parcel to use
    :param rois_of_interest the ROIs to retrieve this value for
    :param func_threshold the percentage cutoff to use
    :param d_parcel_name_map dict mapping the int to str representation of each ROI
    :return the number of voxels above the threshold set (or voxels for which the t-value of the contrast of interest is >0 for contrasts where the t-value threshold is negative at the given percentage)
    """

    parc_fl, parc = _get_fl_in_parcel(
        ROIDIR=ROIDIR, spmT_file=spmT_file, network=netw_of_interest
    )

    number_significant_voxels = {}

    for roi in np.unique(parc):
        roi = int(roi)
        if roi > 0:
            roi_name = d_parcel_name_map[int(roi)]
            if rois_of_interest:
                if not roi_name in rois_of_interest:
                    continue
            samples = parc_fl[(parc == roi) & (~(np.isnan(parc_fl)))]

            percentile_val_used = get_thresh(samples, func_threshold)
            if percentile_val_used <= 0:
                roi_voxels = (parc == roi) & (parc_fl >= 0)
            else:
                roi_voxels = (parc == roi) & (parc_fl >= percentile_val_used)
            # print(f"ROI number: {roi_name}, number of voxels {np.sum(roi_voxels)}")
            number_significant_voxels[roi_name] = np.sum(roi_voxels)

    return number_significant_voxels


def compile_number_significant_voxels() -> pd.DataFrame:
    """
    See `get_number_signicant_voxels` function. Compiles results from all `THRESHOLDS_TO_COMPUTE` in lang and wholebrain parcels for standard and speeded localizer tasks
    """
    for network in ["lang", "wholebrain"]:
        df = pd.DataFrame()
        for fl in [STANDARD, SPEEDED]:
            for contrast in [ALLRUNS, ODD, EVEN]:
                for uid in UID_LIST:
                    spm_file = _get_spmT_file_from_contrast(uid, fl, contrast)

                    significant_voxels = defaultdict(list)
                    roi_name_map = dict(
                        zip(
                            list(range(1, len(D_ROI_ORDER[network]) + 1)),
                            D_ROI_ORDER[network],
                        )
                    )
                    for thresh in THRESHOLDS_TO_COMPUTE:
                        sig_vox_thresh = get_number_significant_voxels(
                            spmT_file=spm_file,
                            netw_of_interest=network,
                            rois_of_interest=None,
                            func_threshold=thresh,
                            d_parcel_name_map=roi_name_map,
                        )
                        significant_voxels = {
                            key: significant_voxels[key] + [value]
                            for key, value in sig_vox_thresh.items()
                            if (key.startswith("LH") and network == "lang")
                            or (network == "wholebrain")
                        }
                    df_temp = pd.DataFrame(significant_voxels)
                    df_temp["fl"] = [fl for i in range(len(df_temp))]
                    df_temp["contrast"] = [contrast for i in range(len(df_temp))]
                    df_temp["session"] = [uid for i in range(len(df_temp))]
                    df_temp["threshold"] = [100 - x for x in THRESHOLDS_TO_COMPUTE]
                    df = pd.concat([df, df_temp])

        df = df.groupby(["fl", "contrast", "threshold"], sort=False).agg(
            ["mean", "std"]
        )
        df.to_csv(f"{DICERESULTDIR}/significant_voxels_{network}.csv")
        df = df.round(decimals=3)

        df.to_latex(f"{DICERESULTDIR}/significant_voxels_{network}.txt")


if __name__ == "__main__":
    if not os.path.isdir(DICERESULTDIR):
        os.makedirs(DICERESULTDIR)

    for THRESH in THRESHOLDS_TO_COMPUTE:
        for network in NETWORKS:
            print(
                f"******************** starting dice for {network}, thresh={THRESH} ********************"
            )
            ######## compare within langloc (ODD_S-N, EVEN_S-N) ########
            params_all_normal = DiceParams(
                identifier="all_normal",
                descrip="Normal\n(odd vs even)",
                version1=STANDARD,
                version2=STANDARD,
                contrast1=ODD,
                contrast2=EVEN,
                thresh=THRESH,
            )
            df = pd.DataFrame(get_dice_coefficients(network, params_all_normal))

            ######## compare wtihin langloc speeded (ODD_S-N, EVEN_S-N) #######
            params_all_speeded = DiceParams(
                identifier="all_speeded",
                descrip="Speeded\n(odd vs even)",
                version1=SPEEDED,
                version2=SPEEDED,
                contrast1=ODD,
                contrast2=EVEN,
                thresh=THRESH,
            )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(get_dice_coefficients(network, params_all_speeded)),
                ]
            )

            ######## compare langloc vs. langloc speeded ########

            # (S-N langloc, S-N speeded)
            params_normal_v_speeded = DiceParams(
                identifier="all_normal_v_speeded",
                descrip="Normal vs. Speeded\n(all runs)",
                version1=STANDARD,
                version2=SPEEDED,
                contrast1=ALLRUNS,
                contrast2=ALLRUNS,
                thresh=THRESH,
            )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        get_dice_coefficients(network, params_normal_v_speeded)
                    ),
                ]
            )

            # (ODD_S-N langloc, ODD_S-N speeded)
            params_normal_odd_speeded_even = DiceParams(
                identifier="normal_odd_speeded_even",
                descrip="Normal (odd) vs.\nSpeeded (even)",
                version1=STANDARD,
                version2=SPEEDED,
                contrast1=ODD,
                contrast2=EVEN,
                thresh=THRESH,
            )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        get_dice_coefficients(network, params_normal_odd_speeded_even),
                    ),
                ]
            )

            # (EVEN_S-N langloc, EVEN_S-N speeded)
            params_normal_even_speeded_odd = DiceParams(
                identifier="normal_even_speeded_odd",
                descrip="Normal (even) vs.\nSpeeded (odd)",
                version1=STANDARD,
                version2=SPEEDED,
                contrast1=EVEN,
                contrast2=ODD,
                thresh=THRESH,
            )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        get_dice_coefficients(network, params_normal_even_speeded_odd),
                    ),
                ]
            )

            # (ODD_S-N langloc, EVEN_S-N speeded)
            params_normal_odd_speeded_odd = DiceParams(
                identifier="normal_odd_speeded_odd",
                descrip="Normal (odd) vs.\nSpeeded (odd)",
                version1=STANDARD,
                version2=SPEEDED,
                contrast1=ODD,
                contrast2=ODD,
                thresh=THRESH,
            )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        get_dice_coefficients(network, params_normal_odd_speeded_odd),
                    ),
                ]
            )

            # (EVEN_S-N langloc, ODD_S-N speeded)
            params_normal_odd_speeded_odd = DiceParams(
                identifier="normal_even_speeded_even",
                descrip="Normal (even) vs.\nSpeeded (even)",
                version1=STANDARD,
                version2=SPEEDED,
                contrast1=EVEN,
                contrast2=EVEN,
                thresh=THRESH,
            )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        get_dice_coefficients(network, params_normal_odd_speeded_odd),
                    ),
                ]
            )

            df.to_csv(
                f"{DICERESULTDIR}/dice_coefficients_{THRESH}percent_{network}.csv"
            )
    compile_number_significant_voxels()

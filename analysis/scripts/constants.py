DATADIR = "analysis/data"
PLOTDIR = "anslysis/plots"
FIRSTLEVELDIR = "analysis/data/firstlevel"
ROIDIR = "analysis/data/roi_parcels"
DICERESULTDIR = "analysis/dice_results"

D_ROI_ORDER = {
    "lang": [
        "LIFGorb",
        "LIFG",
        "LMFG",
        "LAntTemp",
        "LPostTemp",
        "LAngG",
        "RIFGorb",
        "RIFG",
        "RMFG",
        "RAntTemp",
        "RPostTemp",
        "RAngG",
    ],
    "LH_lang": ["LIFGorb", "LIFG", "LMFG", "LAntTemp", "LPostTemp", "LAngG"],
    "RH_lang": ["RIFGorb", "RIFG", "RMFG", "RAntTemp", "RPostTemp", "RAngG"],
    "MD": [
        "LH_Precentral_A_precG",
        "LH_Precentral_B_IFGop",
        "LH_antParietal",
        "LH_insula",
        "LH_medialFrontal",
        "LH_midFrontal",
        "LH_midFrontalOrb",
        "LH_midParietal",
        "LH_postParietal",
        "LH_supFrontal",
        "RH_Precentral_A_precG",
        "RH_Precentral_B_IFGop",
        "RH_antParietal",
        "RH_insula",
        "RH_medialFrontal",
        "RH_midFrontal",
        "RH_midFrontalOrb",
        "RH_midParietal",
        "RH_postParietal",
        "RH_supFrontal",
    ],
    "wholebrain": ["wholebrain_LH", "wholebrain_RH"],
}

D_COLOR_COND = {
    "lang_S-N": ["maroon", "darksalmon", "maroon", "darksalmon"],
    "lang_H-E": [
        "indigo",
        "mediumpurple",
        "indigo",
        "mediumpurple",
    ],
    "MD_H-E": ["darkslategray", "paleturquoise", "darkslategray", "paleturquoise"],
    "MD_S-N": ["mediumblue", "lightsteelblue", "mediumblue", "lightsteelblue"],
    "spcorr": ["maroon", "darksalmon", "gray"],
}

FL_CONTRAST = {
    "langlocSN": {
        "con_0001": "S",
        "con_0002": "N",
        "con_0003": "S-N",
        "con_0004": "N-S",
    },
    "langlocSN_speeded": {
        "con_0001": "S",
        "con_0002": "N",
        "con_0003": "S-N",
        "con_0004": "N-S",
    },
}

UID_LIST = [
    834,
    837,
    838,
    841,
    851,
    853,
    854,
    856,
    863,
    865,
    866,
    870,
    872,
    873,
    875,
    876,
    880,
    887,
    943,
    946,
    947,
    950,
    952,
    958,
]

from __future__ import division, absolute_import, print_function

import pandas as pd


def panel_fillna(panel, type="bfill"):
    """
    fill nan along the 3rd axis
    :param panel: the panel to be filled
    :param type: bfill or ffill
    """
    new_panel = panel.copy()
    for feature in panel.feature:
        for asset in panel.asset:
            if type == "both":
                new_panel.loc[feature, asset] = pd.Series(panel.loc[feature, asset]).fillna(method="ffill"). \
                    fillna(method="bfill")
            else:
                new_panel.loc[feature, asset] = pd.Series(panel.loc[feature, asset]).fillna(method=type)
    return new_panel

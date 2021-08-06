# SPDX-License-Identifier: MIT

"""
This script plots the accuracy results.
"""

import matplotlib.pyplot as plt
import pyarrow.feather

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def main(): # pylint: disable=too-many-locals, too-many-statements
    """
    The main function.
    """

    #***********************************************************************************************
    # Load the results and place them in long form.
    #***********************************************************************************************
    profiles = pyarrow.feather.read_feather("data/ckt5_default_load_profile.feather")

    #***********************************************************************************************
    # Set font sizes.
    #***********************************************************************************************
    #fontsize = 40
    #plt.rc("axes", labelsize=fontsize)
    #plt.rc("xtick", labelsize=fontsize)
    #plt.rc("ytick", labelsize=fontsize)

    #***********************************************************************************************
    # Plot the results.
    #***********************************************************************************************
    (fig_com_md, ax_com_md) = plt.subplots()
    (fig_com_sm, ax_com_sm) = plt.subplots()
    (fig_res, ax_res) = plt.subplots()

    fig_com_md.tight_layout()
    fig_com_sm.tight_layout()
    fig_res.tight_layout()

    fig_com_md.suptitle("Commercial: Medium")
    fig_com_sm.suptitle("Commercial: Small")
    fig_res.suptitle("Residential")

    ax_com_md.plot(profiles["Commercial_MD"])
    ax_com_sm.plot(profiles["Commercial_SM"])
    ax_res.plot(profiles["Residential"])

    ax_com_md.set_xlabel("Time (Hour)")
    ax_com_sm.set_xlabel("Time (Hour)")
    ax_res.set_xlabel("Time (Hour)")

    ax_com_md.set_ylabel("Value")
    ax_com_sm.set_ylabel("Value")
    ax_res.set_ylabel("Value")

    plt.show()

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()

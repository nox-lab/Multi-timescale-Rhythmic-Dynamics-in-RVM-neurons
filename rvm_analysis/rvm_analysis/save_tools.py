"""
Save tools makes saving images ready for inclusion in reports much easier,
by saving pdfs and svgs of the same image at a fixed dpi, leaving the
notebook image at low resolution.
"""

import os
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt


def save_in_folder(
    filename, basePath: Union[Path,str]="Images/", svg=False, dpi=300, show=True, save=True,pad_inches=0.0
):
    """
    Saves a matplotlib image to a directory, keeping the pdf/svg and
    the png image separate. Shows the image after saving, else clears the memory.
    `save` allows for showing the plot without saving it, without writing this
    in every function where save in folder is.
    """
    if save:
        save_path_pdf = (
            os.path.join(basePath, "svgs") if svg else os.path.join(basePath, "pdfs")
        )
        os.makedirs(save_path_pdf, exist_ok=True)
        name = filename + ".svg" if svg else filename + ".pdf"
        plt.savefig(
            os.path.join(save_path_pdf, name), transparent=True, bbox_inches="tight",pad_inches=pad_inches
        )

        save_path_png = os.path.join(basePath, "pngs")
        os.makedirs(save_path_png, exist_ok=True)
        plt.savefig(
            os.path.join(save_path_png, filename + ".png"),
            transparent=True,
            dpi=dpi,
            bbox_inches="tight",pad_inches=pad_inches
        )
    if show:
        plt.show()
    else:
        plt.cla()
        plt.clf()

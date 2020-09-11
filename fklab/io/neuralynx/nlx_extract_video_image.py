"""
===========================================================================
Nlx_extract_video_image (:mod:`fklab.io.neuralynx.nlx_extract_video_image`)
===========================================================================

.. currentmodule:: fklab.io.neuralynx.nlx_extract_video_image

Processing functions for neuralinx video/images


"""

__all__ = [
    "get_nlx_video_time_lut",
    "get_nlx_video_time",
    "nlx_extract_video_image",
    "extract_video_image",
]

import os
import glob
import numpy as np
import yaml
import scipy.interpolate


def get_nlx_video_time_lut(path, target="VT1"):
    """Load timesample from a time_lut.npy file or open a video file and extract timesample from it and transform it in second.

    Parameters
    ----------
    path : str
        path of the time sample file
    target : str, optional
        name of the video associated

    Returns
    -------
    ts : timesample in second

    """
    import bs4

    lut_file = os.path.join(path, target + "_time_lut.npy")

    if os.path.isfile(lut_file):
        # load lut
        ts = np.load(lut_file, mmap_mode="r")
    else:
        filename = os.path.join(path, target + ".smi")

        if not os.path.isfile(filename):
            raise ValueError("No video caption file found: " + filename)

        ts = []

        with open(filename) as fid:
            soup = bs4.BeautifulSoup(fid, "lxml")
            for k in soup.find_all("sync"):
                ts.append([int(k["start"]), int(k.p.text)])

        if len(ts) == 0:
            ts = np.zeros((0, 2))
        else:
            ts = np.array(ts)
            ts = ts / [[1000.0, 1000000.0]]  # convert to seconds

        # save to target_time_lut.npy
        np.save(lut_file, ts)

    return ts


def get_nlx_video_time(path, t, target="VT1"):
    """Construct or load the summary file (video_summary.yaml).

    for each video file with a smi file :
    information gathered are: path, basename, video filename, caption filename, time_lut filename, start time and stop time

    Parameters
    ----------
    path : str
        path of the file

    t:

    target : str, optional
        name of the video associated

    """
    # do we have summary file?
    summary_file = os.path.join(path, "video_summary.yaml")

    if not os.path.isfile(summary_file):
        summary = []
        # construct summary file and LUTs
        # list all mpg and smi files
        video_files = glob.glob(os.path.join(path, target + "*.mpg"))
        video_files = [
            os.path.splitext(x)[0] for x in video_files
        ]  # remove the .mpg from all filenames
        smi_files = glob.glob(os.path.join(path, target + "*.smi"))
        smi_files = [
            os.path.splitext(x)[0] for x in smi_files
        ]  # remove the .smi from all filenames

        # only work with cases where we have both mpg and smi files
        video_files = sorted(
            list(set(video_files).intersection(smi_files))
        )  # why do we need the sorted function ? should not be already sorted by the set()

        # for each video, create LUT
        for k in video_files:
            try:
                ts = get_nlx_video_time_lut(path, target=k)
            except Exception as error:
                print("read nlx video time lut: " + str(error))
                continue

            start_time = np.inf if ts.shape[0] == 0 else ts[0, 1]
            stop_time = -np.inf if ts.shape[0] == 0 else ts[-1, 1]

            basepath, base = os.path.split(k)

            summary.append(
                dict(
                    path=basepath,
                    base=base,
                    video_file=base + ".mpg",
                    caption_file=base + ".smi",
                    time_lut_file=base + "_time_lut.npy",
                    start_time=float(start_time),
                    stop_time=float(stop_time),
                )
            )

        with open(summary_file, "w") as f:
            yaml.dump(summary, stream=f)

    else:

        with open(summary_file, "r") as f:
            summary = yaml.safe_load(f)

    # TODO: Understand this part ---> What is happening here ??
    # reload timesample for each videofile and check if the time specified in input is in the timestample
    for k in summary:
        ts = get_nlx_video_time_lut(k["path"], target=k["base"])
        if t > k["start_time"] and t < k["stop_time"]:
            f = scipy.interpolate.interp1d(
                ts[:, 1], ts[:, 0], kind="linear"
            )  # set the interpolation function
            video_t = f(t)  # give the new time stample interpolated
            return video_t, k

    return None, None


def nlx_extract_video_image(path, t, outputfile, overwrite=False):
    """Manage which file and if the video time sample exist, before extracting the video.

    Parameters
    ----------
    path : str
        path of the file

    t:

    outputfile: str

    overwrite : bool, optional

    """
    # TODO: Add message error ?
    if os.path.isfile(outputfile) and not overwrite:
        raise (ValueError("The file exists and I don't have the right to overwrite it"))

    video_t, video = get_nlx_video_time(path, t)
    if video_t is None:
        return

    extract_video_image(
        os.path.join(video["path"], video["video_file"]), video_t, outputfile
    )


def extract_video_image(inputfile, t, outputfile):
    # TODO: Function really useful ? should be combine with the function before

    import subprocess

    command = ["avconv", "-ss", str(t), "-i", inputfile, "-frames:v", "1", outputfile]
    with open(os.devnull, "w") as f:
        if subprocess.call(["which", "avconv"], stdout=f, stderr=f) == 0:
            subprocess.call(command, stdout=f, stderr=f)
        else:
            print(
                "Cannot extract images from video file. Please install avconv using `sudo apt-get install libav-tools`"
            )

"""Copy synthesis to test the audio processing code"""

import argparse
import os

import numpy as np

import audio
from config import Config as cfg


def copy_synthesis(wav_file, out_path):
    """Perform copy synthesis on the wav file and write the synthesized wav to disk at out_path
    """
    filename = os.path.splitext(os.path.basename(wav_file))[0]

    y = audio.load_wav(wav_file)
    if cfg.rescaling:
        y = y / np.abs(y).max() * cfg.rescaling_max

    mag = audio.spectrogram(y)

    y_hat = audio.inv_spectrogram(mag)

    out_path = os.path.join(out_path, filename + "_synthesized.wav")
    print(f"Writing {out_path} to disk")
    audio.save_wav(y_hat, out_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Do copy synthesis on wav file")

    parser.add_argument(
        "--wav_file",
        help="Path to wav file on which to perform copy synthesis",
        required=True)

    parser.add_argument(
        "--out_path",
        help="Path where the synthesized wav file will be written to disk",
        required=True)

    args = parser.parse_args()
    copy_synthesis(args.wav_file, args.out_path)

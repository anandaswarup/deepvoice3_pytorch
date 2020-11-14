"""Preprocess dataset"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import cpu_count

import numpy as np
from tqdm import tqdm

import audio
from config import Config as cfg


def _process_utterance(mag_dir, mel_dir, wav_path, text):
    """Preprocesses a single utterance audio/text pair.
    This writes the mel and linear scale spectrograms to disk and returns a tuple to write
    to the train.txt file.

    Args:
      mag_dir: The directory to write the log magnitude spectrograms into
      mel_dir: The directory to write the mel spectrograms into
      wav_path: Path to the audio file containing the speech input
      text: The text spoken in the input audio file

    Returns:
      A (filename, text, num_frames) tuple to write to train.txt
    """
    filename = os.path.splitext(os.path.basename(wav_path))[0]

    # Load the audio to a numpy array
    wav = audio.load_wav(wav_path)

    if cfg.rescaling:
        wav = wav / np.abs(wav).max() * cfg.rescaling_max

    # Compute the linear-scale spectrogram from the wav
    spectrogram = audio.spectrogram(wav).astype(np.float32)
    num_frames = spectrogram.shape[1]

    # Compute a mel-scale spectrogram from the wav
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

    # Write the spectrograms to disk
    np.save(os.path.join(mag_dir, filename + ".npy"),
            spectrogram.T,
            allow_pickle=False)

    np.save(os.path.join(mel_dir, filename + ".npy"),
            mel_spectrogram.T,
            allow_pickle=False)

    # Return a tuple describing this training example:
    return (filename, text, num_frames)


def build_from_path_ljspeech(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    """Preprocesses the LJ Speech dataset from a given input path into a given output directory.

      Args:
        in_dir: The directory where you have downloaded the LJ Speech dataset
        out_dir: The directory to write the output into
        num_workers: Optional number of worker processes to parallelize across
        tqdm: You can optionally pass tqdm to get a nice progress bar

      Returns:
        A list of tuples describing the training examples. This should be written to train.txt
    """
    mag_dir = os.path.join(out_dir, "mag")
    os.makedirs(mag_dir, exist_ok=True)

    mel_dir = os.path.join(out_dir, "mel")
    os.makedirs(mel_dir, exist_ok=True)

    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []

    with open(os.path.join(in_dir, "metadata.csv"), "r") as file_reader:
        for line in file_reader:
            parts = line.strip().split('|')
            wav_path = os.path.join(in_dir, 'wavs', '%s.wav' % parts[0])
            text = parts[2]
            futures.append(
                executor.submit(
                    partial(_process_utterance, mag_dir, mel_dir, wav_path,
                            text)))

    return [future.result() for future in tqdm(futures)]


def preprocess(in_dir, out_dir, num_workers):
    os.makedirs(out_dir, exist_ok=True)

    if cfg.dataset == "ljspeech":
        metadata = build_from_path_ljspeech(in_dir,
                                            out_dir,
                                            num_workers,
                                            tqdm=tqdm)
    else:
        raise NotImplementedError

    write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, "train.txt"), "w") as file_writer:
        for m in metadata:
            file_writer.write('|'.join([str(x) for x in m]) + '\n')

    frames = sum([m[2] for m in metadata])
    frame_shift_ms = cfg.hop_size / cfg.sample_rate * 1000
    hours = frames * frame_shift_ms / (3600 * 1000)

    print(
        f"Wrote {len(metadata)} utterances, {frames} frames ({hours:2f} hours)"
    )
    print(f"Max input length: {max(len(m[1]) for m in metadata)}")
    print(f"Max output length: {max(m[2] for m in metadata)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset")

    parser.add_argument("--dataset_dir",
                        help="Path to the dataset dir",
                        required=True)

    parser.add_argument("--out_dir",
                        help="Path to the output dir",
                        required=True)

    args = parser.parse_args()
    num_workers = cpu_count()

    dataset_dir = args.dataset_dir
    out_dir = args.out_dir
    preprocess(dataset_dir, out_dir, num_workers)

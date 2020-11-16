"""Synthesis script"""

import argparse
import os
import sys
from os.path import basename, join, splitext

import numpy as np
import torch

import audio
from config import Config as cfg
from frontend import english
from train import build_model, plot_alignment

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def tts(model, text, speaker_id=None, fast=False):
    """Convert text to speech waveform given a deepvoice3 model.
    """
    model = model.to(device)
    model.eval()

    if fast:
        model.make_generation_fast_()

    if cfg.frontend == "en":
        sequence = np.array(english.text_to_sequence(text))
    else:
        raise NotImplementedError

    sequence = torch.from_numpy(sequence).unsqueeze(0).long().to(device)
    text_positions = torch.arange(1,
                                  sequence.size(-1) +
                                  1).unsqueeze(0).long().to(device)

    speaker_ids = None if speaker_id is None else torch.LongTensor(
        [speaker_id]).to(device)

    # Greedy decoding
    with torch.no_grad():
        mel_outputs, linear_outputs, alignments, done = model(
            sequence, text_positions=text_positions, speaker_ids=speaker_ids)

    linear_output = linear_outputs[0].cpu().data.numpy()
    spectrogram = audio._denormalize(linear_output)
    alignment = alignments[0].cpu().data.numpy()
    mel = mel_outputs[0].cpu().data.numpy()
    mel = audio._denormalize(mel)

    # Predicted audio signal
    waveform = audio.inv_spectrogram(linear_output.T)

    return waveform, alignment, spectrogram, mel


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synthesize text using a trained DeepVoice3 model")

    parser.add_argument("--out_dir",
                        help="Output dir where synthesis output will be saved",
                        required=True)

    parser.add_argument("--text_file",
                        help="Path to the file sentences to be synthesized",
                        required=True)

    parser.add_argument("--checkpoint_path",
                        help="Path to the checkpoint to synthesize from",
                        required=True)

    parser.add_argument("--max_decoder_steps",
                        help="The max number of steps to run the decoder",
                        required=True)

    parser.add_argument("--speaker_id",
                        help="Speaker id (in case of multispeaker synthesis)",
                        required=False)

    args = parser.parse_args()

    out_dir = args.out_dir
    text_file = args.text_file
    checkpoint_path = args.checkpoint_path
    max_decoder_steps = args.max_decoder_steps

    speaker_id = int(args.speaker_id) if args.speaker_id else None

    # Model
    model = build_model()

    # Load checkpoints
    checkpoint = _load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    checkpoint_name = splitext(basename(checkpoint_path))[0]

    model.seq2seq.decoder.max_decoder_steps = max_decoder_steps

    os.makedirs(out_dir, exist_ok=True)

    with open(text_file, "r") as file_reader:
        lines = file_reader.readlines()
        lines = [line.strip("\n") for line in lines]

        for idx, line in enumerate(lines):
            text = line[:-1]
            waveform, alignment, _, _ = tts(model,
                                            text,
                                            speaker_id=speaker_id,
                                            fast=True)
            out_wav_path = join(out_dir,
                                f"{idx}_{checkpoint_name}_synthesized.wav")

            out_alignment_path = join(
                out_dir, f"{idx}_{checkpoint_name}_synthesized_alignment.png")

            plot_alignment(alignment.T,
                           out_alignment_path,
                           info=f"{cfg.builder}, {basename(checkpoint_path)}")

            audio.save_wav(waveform, out_wav_path)

    print(f"Synthesis complete. Generated audio samples saved in {out_dir}")

    sys.exit(0)

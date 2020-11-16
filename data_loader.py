"""Model data loader"""

import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from config import Config as cfg
from frontend import english


def _pad(seq, max_len, constant_values=0):
    return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant',
                  constant_values=constant_values)


def _pad_2d(x, max_len, b_pad=0):
    x = np.pad(x, [(b_pad, max_len - len(x) - b_pad), (0, 0)],
               mode="constant",
               constant_values=0)
    return x


def _load_training_instances(training_file):
    """Load training instances from file into memory
    """
    with open(training_file, "r") as file_reader:
        training_instances = file_reader.readlines()

    training_instances = [
        instance.strip("\n") for instance in training_instances
    ]
    training_instances = [
        instance.split("|") for instance in training_instances
    ]

    return training_instances


class TTSDataset(Dataset):
    """TTS dataset
    """
    def __init__(self, data_dir):
        """Initialize the dataset
        """
        self.training_instances = _load_training_instances(
            os.path.join(data_dir, "train.txt"))
        self.data_dir = data_dir
        self.frame_lengths = [int(x[2]) for x in self.training_instances]
        self.multi_speaker = len(self.training_instances[0]) == 4

    def __len__(self):
        return len(self.training_instances)

    def __getitem__(self, idx):
        return self.load_data(idx)

    def load_data(self, idx):
        """Load the data
        """
        if self.multi_speaker:
            filename, text, num_frames, speaker_id = self.training_instances[
                idx]
        else:
            filename, text, num_frames = self.training_instances[idx]

        # Transform text to sequence of ids
        if cfg.frontend == "en":
            text_seq = np.asarray(english.text_to_sequence(text),
                                  dtype=np.int32)
        else:
            raise NotImplementedError

        # Load the log-magnitude and mel-spectrograms
        mag = np.load(os.path.join(self.data_dir, "mag",
                                   filename + ".npy")).astype(np.float32)
        mel = np.load(os.path.join(self.data_dir, "mel",
                                   filename + ".npy")).astype(np.float32)

        assert int(num_frames) == mag.shape[0] == mel.shape[0]

        if self.multi_speaker:
            return text_seq, mag, mel, int(speaker_id)
        else:
            return text_seq, mag, mel

    def collate_fn(self, batch):
        """Create batch
        """
        r = cfg.outputs_per_step
        downsample_step = cfg.downsample_step

        # Lengths
        input_lengths = [len(x[0]) for x in batch]
        max_input_len = max(input_lengths)

        target_lengths = [len(x[1]) for x in batch]

        max_target_len = max(target_lengths)
        if max_target_len % r != 0:
            max_target_len += r - max_target_len % r
            assert max_target_len % r == 0
        if max_target_len % downsample_step != 0:
            max_target_len += downsample_step - max_target_len % downsample_step
            assert max_target_len % downsample_step == 0

        # Set 0 for zero beginning padding
        # imitates initial decoder states
        b_pad = r
        max_target_len += b_pad * downsample_step

        a = np.array([_pad(x[0], max_input_len) for x in batch], dtype=np.int)
        x_batch = torch.LongTensor(a)

        input_lengths = torch.LongTensor(input_lengths)
        target_lengths = torch.LongTensor(target_lengths)

        b = np.array(
            [_pad_2d(x[1], max_target_len, b_pad=b_pad) for x in batch],
            dtype=np.float32)
        mel_batch = torch.FloatTensor(b)

        c = np.array(
            [_pad_2d(x[2], max_target_len, b_pad=b_pad) for x in batch],
            dtype=np.float32)
        y_batch = torch.FloatTensor(c)

        # text positions
        text_positions = np.array(
            [_pad(np.arange(1,
                            len(x[0]) + 1), max_input_len) for x in batch],
            dtype=np.int)
        text_positions = torch.LongTensor(text_positions)

        max_decoder_target_len = max_target_len // r // downsample_step

        # frame positions
        s, e = 1, max_decoder_target_len + 1
        # if b_pad > 0:
        #    s, e = s - 1, e - 1
        # NOTE: needs clone to supress RuntimeError in dataloarder...
        # ref: https://github.com/pytorch/pytorch/issues/10756
        frame_positions = torch.arange(s, e).long().unsqueeze(0).expand(
            len(batch), max_decoder_target_len).clone()

        # done flags
        done = np.array([
            _pad(np.zeros(len(x[1]) // r // downsample_step - 1),
                 max_decoder_target_len,
                 constant_values=1) for x in batch
        ])
        done = torch.FloatTensor(done).unsqueeze(-1)

        if self.multi_speaker:
            speaker_ids = torch.LongTensor([x[3] for x in batch])
        else:
            speaker_ids = None

        return x_batch, input_lengths, mel_batch, y_batch, \
            (text_positions, frame_positions), done, target_lengths, speaker_ids


class RandomizedLengthSampler(Sampler):
    """Partially randmoized sampler
        1. Sort by lengths
        2. Pick a small patch and randomize it
        3. Permutate mini-batchs
    """
    def __init__(self,
                 lengths,
                 batch_size=16,
                 batch_group_size=None,
                 permutate=True):
        self.lengths, self.sorted_indices = torch.sort(
            torch.LongTensor(lengths))
        self.batch_size = batch_size
        if batch_group_size is None:
            batch_group_size = min(batch_size * 32, len(self.lengths))
            if batch_group_size % batch_size != 0:
                batch_group_size -= batch_group_size % batch_size

        self.batch_group_size = batch_group_size
        assert batch_group_size % batch_size == 0
        self.permutate = permutate

    def __iter__(self):
        indices = self.sorted_indices.clone()
        batch_group_size = self.batch_group_size
        s, e = 0, 0
        for i in range(len(indices) // batch_group_size):
            s = i * batch_group_size
            e = s + batch_group_size
            random.shuffle(indices[s:e])

        # Permutate batches
        if self.permutate:
            perm = np.arange(len(indices[:e]) // self.batch_size)
            random.shuffle(perm)
            indices[:e] = indices[:e].view(-1,
                                           self.batch_size)[perm, :].view(-1)

        # Handle last elements
        s += batch_group_size
        if s < len(indices):
            random.shuffle(indices[s:])

        return iter(indices)

    def __len__(self):
        return len(self.sorted_indices)

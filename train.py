"""Model training script"""

import argparse
import os
import sys
from datetime import datetime
from os.path import join

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader

import audio
from config import Config as cfg
from data_loader import RandomizedLengthSampler, TTSDataset
from frontend import english
from loss import guided_attentions, sequence_mask, spec_loss
from model_builder import deepvoice3_model

global_step = 0
global_epoch = 0

use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False


def noam_learning_rate_decay(init_lr, global_step, warmup_steps=4000):
    """Noam learning rate decay
    """
    warmup_steps = float(warmup_steps)
    step = global_step + 1.
    lr = init_lr * warmup_steps**0.5 * np.minimum(step * warmup_steps**-1.5,
                                                  step**-0.5)

    return lr


def plot_alignment(alignment, path, info=None):
    fig, ax = plt.subplots()
    im = ax.imshow(alignment,
                   aspect='auto',
                   origin='lower',
                   interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    plt.savefig(path, format='png')
    plt.close()


def time_string():
    return datetime.now().strftime('%Y-%m-%d %H:%M')


def save_alignment(path, attn):
    plot_alignment(attn.T,
                   path,
                   info="{}, {}, step={}".format(cfg.builder, time_string(),
                                                 global_step))


def eval_model(device, model, global_step, logs_dir, ismultispeaker):
    """Evaluate the model
    """
    import synthesis

    # Hardcoded sentences for evaluation
    texts = [
        "Scientists at the CERN laboratory say they have discovered a new particle.",
        "There's a way to measure the acute emotional intelligence that has never gone out of style.",
        "President Trump met with other leaders at the Group of Twenty conference.",
        "Generative adversarial network or variational auto-encoder.",
        "Please call Stella.",
        "Some have accepted this as a miracle without any physical explanation.",
    ]

    eval_output_dir = join(logs_dir, "eval")
    os.makedirs(eval_output_dir, exist_ok=True)

    # Prepare model for evaluation
    model_eval = build_model().to(device)
    model_eval.load_state_dict(model.state_dict())

    # hard coded
    speaker_ids = [0, 1, cfg.n_speakers - 1] if ismultispeaker else [None]

    for speaker_id in speaker_ids:
        speaker_str = "multispeaker{}".format(
            speaker_id) if speaker_id is not None else "single"

        for idx, text in enumerate(texts):
            signal, alignment, _, _ = synthesis.tts(model_eval,
                                                    text,
                                                    speaker_id=speaker_id,
                                                    fast=True)
            signal /= np.max(np.abs(signal))

            # Alignment
            path = join(
                eval_output_dir,
                f"step{global_step:09d}_text{idx}_{speaker_str}_alignment.png")
            save_alignment(path, alignment)

            # Audio
            path = join(
                eval_output_dir,
                "step{global_step:09d}_text{idx}_{speaker_str}_predicted.wav")
            audio.save_wav(signal, path)


def save_states(global_step, attn, linear_outputs, input_lengths, logs_dir):
    """Save intermediate states
    """
    print(f"Save intermediate states at step {global_step:09d}")

    idx = min(1, len(input_lengths) - 1)

    # Alignment
    # Multi-hop attention
    if attn is not None and attn.dim() == 4:
        for i, alignment in enumerate(attn):
            # Save alignment to disk
            alignment = alignment[idx].cpu().data.numpy()

            alignment_dir = join(logs_dir, f"alignment_layer{i + 1}")
            os.makedirs(alignment_dir, exist_ok=True)

            path = join(alignment_dir,
                        f"step{global_step:09d}_layer_{i + 1}_alignment.png")

            save_alignment(path, alignment)

        # Save averaged alignment
        alignment_dir = join(checkpoint_dir, "alignment_ave")
        os.makedirs(alignment_dir, exist_ok=True)

        path = join(alignment_dir,
                    f"step{global_step:09d}_layer_alignment.png")

        alignment = attn.mean(0)[idx].cpu().data.numpy()

        save_alignment(path, alignment)

    linear_output = linear_outputs[idx].cpu().data.numpy()

    # Predicted audio signal
    signal = audio.inv_spectrogram(linear_output.T)
    signal /= np.max(np.abs(signal))
    path = join(checkpoint_dir, f"step{global_step:09d}_predicted.wav")

    audio.save_wav(signal, path)


def train(device,
          model,
          data_loader,
          optimizer,
          init_lr=0.002,
          checkpoint_dir=None,
          logs_dir=None,
          checkpoint_interval=None,
          nepochs=None,
          clip_thresh=1.0):
    """Train the model
    """
    linear_dim = model.linear_dim
    r = cfg.outputs_per_step
    downsample_step = cfg.downsample_step
    current_lr = init_lr

    binary_criterion = nn.BCELoss()

    global global_step, global_epoch

    while global_epoch < nepochs:
        for _, (x, input_lengths, mel, y, positions, done, target_lengths,
                speaker_ids) in enumerate(data_loader):

            model.train()

            ismultispeaker = speaker_ids is not None

            # Learning rate schedule
            if cfg.lr_schedule == "noam_learning_rate_decay":
                current_lr = noam_learning_rate_decay(init_lr,
                                                      global_step,
                                                      warmup_steps=4000)

                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr
            optimizer.zero_grad()

            # Used for Position encoding
            text_positions, frame_positions = positions

            # Downsample mel spectrogram
            if downsample_step > 1:
                mel = mel[:, 0::downsample_step, :].contiguous()

            # Lengths
            input_lengths = input_lengths.long().numpy()
            decoder_lengths = target_lengths.long().numpy(
            ) // r // downsample_step

            max_seq_len = max(input_lengths.max(), decoder_lengths.max())
            if max_seq_len >= cfg.max_positions:
                raise RuntimeError(
                    """Input length or decoder target length exceeded the maximum length. Change the value of
                    max_positions""")

            # Data device placement
            x = x.to(device)
            text_positions = text_positions.to(device)
            frame_positions = frame_positions.to(device)
            y = y.to(device)
            mel, done = mel.to(device), done.to(device)
            target_lengths = target_lengths.to(device)
            speaker_ids = speaker_ids.to(device) if ismultispeaker else None

            # Create mask (for masked loss)
            if cfg.masked_loss_weight > 0:
                # decoder target mask
                decoder_target_mask = sequence_mask(
                    target_lengths // (r * downsample_step),
                    max_len=mel.size(1)).unsqueeze(-1)

                # spectrogram mask
                if downsample_step > 1:
                    target_mask = sequence_mask(
                        target_lengths, max_len=y.size(1)).unsqueeze(-1)
                else:
                    target_mask = decoder_target_mask

                # shift mask
                decoder_target_mask = decoder_target_mask[:, r:, :]
                target_mask = target_mask[:, r:, :]
            else:
                decoder_target_mask, target_mask = None, None

            # Forward pass
            mel_outputs, linear_outputs, attn, done_hat = model(
                x,
                mel,
                speaker_ids=speaker_ids,
                text_positions=text_positions,
                frame_positions=frame_positions,
                input_lengths=input_lengths)

            # Loss computation
            w = cfg.binary_divergence_weight

            # mel loss
            mel_l1_loss, mel_binary_div = spec_loss(mel_outputs[:, :-r, :],
                                                    mel[:, r:, :],
                                                    decoder_target_mask)

            mel_loss = (1 - w) * mel_l1_loss + w * mel_binary_div

            # stop token loss
            done_loss = binary_criterion(done_hat, done)

            # linear loss
            n_priority_freq = int(cfg.priority_freq / (cfg.sample_rate * 0.5) *
                                  linear_dim)

            linear_l1_loss, linear_binary_div = spec_loss(
                linear_outputs[:, :-r, :],
                y[:, r:, :],
                target_mask,
                priority_bin=n_priority_freq,
                priority_w=cfg.priority_freq_weight)

            linear_loss = (1 - w) * linear_l1_loss + w * linear_binary_div

            # Overall loss
            loss = mel_loss + linear_loss + done_loss

            # Guided attention loss
            if cfg.use_guided_attention:
                soft_mask = guided_attentions(input_lengths,
                                              decoder_lengths,
                                              attn.size(-2),
                                              g=cfg.guided_attention_sigma)

                soft_mask = torch.from_numpy(soft_mask).to(device)
                attn_loss = (attn * soft_mask).mean()

                loss += attn_loss

            if global_step > 0 and global_step % checkpoint_interval == 0:
                # Save attention states
                save_states(global_step, attn, linear_outputs, input_lengths,
                            logs_dir)

                # Save checkpoint
                save_checkpoint(model, optimizer, global_step, global_epoch,
                                checkpoint_dir)

            if global_step > 0 and global_step % cfg.eval_interval == 0:
                eval_model(device, model, global_step, logs_dir,
                           ismultispeaker)

            # Update (backward pass)
            loss.backward()
            if clip_thresh > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.get_trainable_parameters(), clip_thresh)
            optimizer.step()

            # Log training
            log_str = (
                f"Epoch: {global_epoch:09d}, Step: {global_step:09d}, Total loss: {loss.item():09f}, "
                f"Mel loss: {mel_loss.item():09f}, Linear loss: {linear_loss.item():09f}, "
                f"Done loss: {done_loss.item():09f} ")

            if cfg.use_guided_attention:
                log_str += f"Attention loss: {attn_loss.item():09f}"

            print(log_str)

            global_step += 1

        global_epoch += 1


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):
    """Write the checkpoint to disk
    """
    checkpoint_path = join(checkpoint_dir, f"checkpoint_step{step:09d}.pth")

    optimizer_state = optimizer.state_dict(
    ) if cfg.save_optimizer_state else None

    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer_state,
            "global_step": step,
            "global_epoch": epoch,
        }, checkpoint_path)

    print(f"Saved checkpoint: {checkpoint_path}")


def build_model():
    if cfg.frontend == "en":
        num_chars = english.num_chars()
    else:
        raise NotImplementedError

    model = deepvoice3_model(
        n_vocab=num_chars,
        embed_dim=cfg.text_embed_dim,
        mel_dim=cfg.num_mels,
        linear_dim=cfg.fft_size // 2 + 1,
        r=cfg.outputs_per_step,
        downsample_step=cfg.downsample_step,
        n_speakers=cfg.num_speakers,
        speaker_embed_dim=cfg.speaker_embed_dim,
        padding_idx=cfg.padding_idx,
        dropout=cfg.dropout,
        kernel_size=cfg.kernel_size,
        encoder_channels=cfg.encoder_channels,
        decoder_channels=cfg.decoder_channels,
        converter_channels=cfg.converter_channels,
        query_position_rate=cfg.query_position_rate,
        key_position_rate=cfg.key_position_rate,
        use_memory_mask=cfg.use_memory_mask,
        trainable_positional_encodings=cfg.trainable_positional_encodings,
        force_monotonic_attention=cfg.force_monotonic_attention,
        use_decoder_state_for_postnet_input=cfg.
        use_decoder_state_for_postnet_input,
        max_positions=cfg.max_positions,
        embedding_weight_std=cfg.embedding_weight_std,
        speaker_embedding_weight_std=cfg.speaker_embedding_weight_std,
        freeze_embedding=cfg.freeze_embedding,
        window_ahead=cfg.window_ahead,
        window_backward=cfg.window_backward,
        key_projection=cfg.key_projection,
        value_projection=cfg.value_projection,
    )

    return model


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model, optimizer):
    """Load the checkpoint and set the optimizer state (in order to resume training)
    """
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer_state = checkpoint["optimizer"]

    if optimizer_state is not None:
        print("Load optimizer state from {}".format(path))
        optimizer.load_state_dict(checkpoint["optimizer"])

    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the DeepVoice3 model")

    parser.add_argument(
        "--data_dir",
        help="Path to the data dir containing the training data",
        required=True)

    parser.add_argument(
        "--checkpoint_dir",
        help="Path to the dir, where training checkpoints will be saved",
        required=True)

    parser.add_argument(
        "--logs_dir",
        help="Path to the dir, where training logs will be written",
        required=True)

    parser.add_argument(
        "--checkpoint_path",
        help="If specified load checkpoint and restart training from that point"
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    checkpoint_dir = args.checkpoint_dir
    logs_dir = args.logs_dir
    checkpoint_path = args.checkpoint_path

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Setup dataloader
    dataset = TTSDataset(data_dir)
    frame_lengths = dataset.frame_lengths
    sampler = RandomizedLengthSampler(frame_lengths, batch_size=cfg.batch_size)
    dataloader = DataLoader(dataset,
                            batch_size=cfg.batch_size,
                            num_workers=cfg.num_workers,
                            sampler=sampler,
                            collate_fn=dataset.collate_fn,
                            pin_memory=cfg.pin_memory,
                            drop_last=True)

    # Setup device
    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    model = build_model().to(device)

    optimizer = optim.Adam(model.get_trainable_parameters(),
                           lr=cfg.initial_learning_rate,
                           betas=(cfg.adam_beta1, cfg.adam_beta2),
                           eps=cfg.adam_eps,
                           weight_decay=cfg.weight_decay)

    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer)

    # Train the model
    try:
        train(device,
              model,
              dataloader,
              optimizer,
              init_lr=cfg.initial_learning_rate,
              checkpoint_dir=checkpoint_dir,
              logs_dir=logs_dir,
              checkpoint_interval=cfg.checkpoint_interval,
              nepochs=cfg.nepochs,
              clip_thresh=cfg.clip_thresh)
    except KeyboardInterrupt:
        save_checkpoint(model, optimizer, global_step, checkpoint_dir,
                        global_epoch)

    print("Training complete")
    sys.exit(0)

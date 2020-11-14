"""Configuration parameters / Hyperparameters"""


class Config:
    # Model [deepvoice3 or deepvoice3_multispeaker]
    builder = "deepvoice3"

    # Dataset [currently supports ljspeech and libritts]
    dataset = "ljspeech"

    # Frontend [currently supports only English (en)]
    frontend = 'en'

    # Must be configured based on the dataset and model being used
    # if builder = deepvoice3_multispeaker then dataset == libritts and num_speakers = 247
    # if builder == deepvoice3 then dataset == ljspeech and num_speakers = 1
    num_speakers = 1
    speaker_embed_dim = 16

    # Audio processing parameters
    num_mels = 80
    fmin = 125
    fmax = 7600
    fft_size = 1024
    hop_size = 256
    sample_rate = 22050
    preemphasis_coef = 0.97
    min_level_db = -100
    ref_level_db = 20

    # whether to rescale waveform or not.
    # Let x is an input waveform rescaled waveform y is given by:
    # y  =  x / np.abs(x).max() * rescaling_max
    rescaling = False
    rescaling_max = 0.999

    # mel-spectrogram is normalized to [0 1] for each utterance and clipping may
    # happen depends on min_level_db and ref_level_db causing clipping noise.
    # If False assertion is added to ensure no clipping happens.
    allow_clipping_in_normalization = True

    # Model architecture
    downsample_step = 4
    outputs_per_step = 1

    embedding_weight_std = 0.1
    speaker_embedding_weight_std = 0.01
    padding_idx = 0

    max_positions = 512  # Max input text length (try setting larger value if you want to give very long text input)

    dropout = 1 - 0.95

    text_embed_dim = 128

    kernel_size = 3
    encoder_channels = 256
    decoder_channels = 256
    converter_channels = 256

    query_position_rate = 1.0
    key_position_rate = 1.385
    key_projection = False
    value_projection = False
    use_memory_mask = True
    trainable_positional_encodings = False
    freeze_embedding = False

    use_decoder_state_for_postnet_input = True

    # Data loader
    pin_memory = True
    num_workers = 2

    # Loss
    masked_loss_weight = 0.5  # (1 - w) * loss + w * masked_loss
    priority_freq = 3000  # heuristic: prioritize [0 ~ priority_freq] for linear loss
    priority_freq_weight = 0.0  # (1 - w) * linear_loss + w * priority_linear_loss
    binary_divergence_weight = 0.1  # set 0 to disable
    use_guided_attention = True
    guided_attention_sigma = 0.2

    # Training parameters
    batch_size = 16
    adam_beta1 = 0.5
    adam_beta2 = 0.9
    adam_eps = 1e-6
    initial_learning_rate = 5e-4
    lr_schedule = "noam_learning_rate_decay"
    nepochs = 500
    weight_decay = 0.0
    clip_thresh = 0.1

    # Training monitoring
    checkpoint_interval = 10000
    eval_interval = 10000
    save_optimizer_state = True

    # Synthesis parameters
    force_monotonic_attention = True
    window_ahead = 3  # Attention constraint for incremental decoding
    window_backward = 1  # 0 tends to prevent word repretetion but sometime causes skip words
    power = 1.4  # Power to raise magnitudes to prior to phase retrieval

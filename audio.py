import librosa
import librosa.filters
import lws
import numpy as np
from scipy import signal
from scipy.io import wavfile

from config import Config as cfg

_mel_basis = None


def _lws_processor():
    return lws.lws(cfg.fft_size, cfg.hop_size, mode="speech")


def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)


def _build_mel_basis():
    if cfg.fmax is not None:
        assert cfg.fmax <= cfg.sample_rate // 2
    return librosa.filters.mel(cfg.sample_rate,
                               cfg.fft_size,
                               fmin=cfg.fmin,
                               fmax=cfg.fmax,
                               n_mels=cfg.num_mels)


def _amp_to_db(x):
    min_level = np.exp(cfg.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _normalize(S):
    return np.clip((S - cfg.min_level_db) / -cfg.min_level_db, 0, 1)


def _denormalize(S):
    return (np.clip(S, 0, 1) * -cfg.min_level_db) + cfg.min_level_db


def load_wav(path):
    """Read the wav file from disk and load into memory
    """
    return librosa.core.load(path, sr=cfg.sample_rate)[0]


def save_wav(wav, path):
    """Write wav file to disk
    """
    wav = wav * 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, cfg.sample_rate, wav.astype(np.int16))


def preemphasis(x):
    """Apply preemphasis on signal
    """
    b = np.array([1., -cfg.preemphasis_coef], x.dtype)
    a = np.array([1.], x.dtype)

    return signal.lfilter(b, a, x)


def inv_preemphasis(x):
    """Invert the preemphasis
    """
    b = np.array([1.], x.dtype)
    a = np.array([1., -cfg.preemphasis_coef], x.dtype)

    return signal.lfilter(b, a, x)


def spectrogram(y):
    """Compute log-magnitude spectrogram from signal
    """
    D = _lws_processor().stft(preemphasis(y)).T
    S = _amp_to_db(np.abs(D)) - cfg.ref_level_db

    return _normalize(S)


def melspectrogram(y):
    """Compute the mel-spectrogram from signal
    """
    D = _lws_processor().stft(preemphasis(y)).T
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - cfg.ref_level_db
    if not cfg.allow_clipping_in_normalization:
        assert S.max() <= 0 and S.min() - cfg.min_level_db >= 0

    return _normalize(S)


def inv_spectrogram(spectrogram):
    """Invert the spectrogram back to signal by iterative estimation of phase
    """
    S = _db_to_amp(_denormalize(spectrogram) +
                   cfg.ref_level_db)  # Convert back to linear
    processor = _lws_processor()
    D = processor.run_lws(S.astype(np.float64).T**cfg.power)
    y = processor.istft(D).astype(np.float32)

    return inv_preemphasis(y)

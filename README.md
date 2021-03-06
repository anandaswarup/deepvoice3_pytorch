# Convolutional Neural Network based TTS models

PyTorch implementation of convolutional neural network based text-to-speech (TTS) models. The code in this repository is a modified version of [Ryuichi Yamamoto's implementation of DeepVoice3](https://github.com/r9y9/deepvoice3_pytorch)

## Highlights

- Convolutional sequence-to-sequence model with attention for text-to-speech synthesis
- Multi-speaker and single speaker versions of DeepVoice3
- Preprocessor for [LJSpeech (en)](https://keithito.com/LJ-Speech-Dataset/), and [LibriTTS (en, multi-speaker)](http://www.openslr.org/60/) datasets
- Language-dependent frontend text processor for English

## Getting started

### Configuration

Configuration parameters / hyperparameters are specified in `config.py`, and these values are used across all scripts.

### 0. Download dataset

- LJSpeech (en): https://keithito.com/LJ-Speech-Dataset/
- LibriTTS (en, multi-speaker): http://www.openslr.org/60/

### 1. Preprocessing

Usage:

```
python preprocess.py --dataset_path <path to the dataset dir> --out_dir <output dir>
```

Supported datasets are:

- `ljspeech` (en, single speaker)
- `libritts` (en, multi-speaker)

When this is done, you will see extracted features (mel-spectrograms and linear spectrograms) in `out_dir`.

### 2. Training
Usage:
```
python train.py --data_dir <path to the training data dir> \
		--checkpoint_dir <path to the dir, where training checkpoints will be saved> \
		--logs_dir <path to the dir, where training logs will be written> \
		--checkpoint_path <if specified load checkpoint and restart training from that point>
```

## Acknowledgements

1. [Ryuichi Yamamoto's implementation of DeepVoice3](https://github.com/r9y9/deepvoice3_pytorch)

## References
1. [arXiv:1710.07654](https://arxiv.org/abs/1710.07654): Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning.
2. [arXiv:1710.08969](https://arxiv.org/abs/1710.08969): Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention.

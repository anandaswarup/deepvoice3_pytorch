# DeepVoice3

PyTorch implementation of convolutional networks-based text-to-speech synthesis models based on

1. [arXiv:1710.07654](https://arxiv.org/abs/1710.07654): Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning.
2. [arXiv:1710.08969](https://arxiv.org/abs/1710.08969): Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention.

The code in this repository is a modified version of [Ryuichi Yamamoto's implementation of DeepVoice3](https://github.com/r9y9/deepvoice3_pytorch)

## Highlights

- Convolutional sequence-to-sequence model with attention for text-to-speech synthesis
- Multi-speaker and single speaker versions of DeepVoice3
- Preprocessor for [LJSpeech (en)](https://keithito.com/LJ-Speech-Dataset/), and [LibriTTS (en, multi-speaker)](http://www.openslr.org/60/) datasets
- Language-dependent frontend text processor for English

## Getting started

### Preset parameters

Configuration parameters / hyperparameters are provided by means of json files in the `config` directory. The following scripts

1. `preprocess.py`
2. `train.py`
3. `synthesis.py`

accepts `--config <path to json file>` as an  optional parameter, which specifies the path to a json config file. The same `--config=<path to json file>` must be used throughout preprocessing, training and evaluation.

### 0. Download dataset

- LJSpeech (en): https://keithito.com/LJ-Speech-Dataset/
- LibriTTS (en, multi-speaker): http://www.openslr.org/60/

### 1. Preprocessing

Usage:

```
python preprocess.py --config <path to json config file> --dataset_path <path to the dataset dir> --out_dir <output dir>
```

Supported datasets are:

- `ljspeech` (en, single speaker)
- `libritts` (en, multi-speaker)

When this is done, you will see extracted features (mel-spectrograms and linear spectrograms) in `out_dir`.

## Acknowledgements

1. [Ryuichi Yamamoto's implementation of DeepVoice3](https://github.com/r9y9/deepvoice3_pytorch)


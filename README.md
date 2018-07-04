# Wave-U-Net
Implementation of the [Wave-U-Net](https://arxiv.org/abs/1806.03185) for audio source separation.

## Listening examples

Listen to vocal separation results [here](https://sisec18.unmix.app/#/unmix/Side%20Effects%20Project%20-%20Sing%20With%20Me/STL1) and to multi-instrument separation results [here](https://sisec18.unmix.app/#/unmix/Side%20Effects%20Project%20-%20Sing%20With%20Me/STL2)

## What is the Wave-U-Net?
The Wave-U-Net is a convolutional neural network applicable to audio source separation tasks, which works directly on the raw audio waveform, presented in [this paper](https://arxiv.org/abs/1806.03185).

The Wave-U-Net is an adaptation of the U-Net architecture to the one-dimensional time domain to perform end-to-end audio source separation. Through a series of downsampling and upsampling blocks, which involve 1D convolutions combined with a down-/upsampling process, features are computed on multiple scales/levels of abstraction and time resolution, and combined to make a prediction.

See the diagram below for a summary of the network architecture.

<img src="./waveunet.png" width="500">

### Participation in the SiSec separation competition

The Wave-U-Net also participated in the [SiSec separation campaign](https://sisec.inria.fr/) as submissions [STL1](https://github.com/sigsep/sigsep-mus-2018/blob/master/submissions/STL1/description.md) and [STL2](https://github.com/sigsep/sigsep-mus-2018/blob/master/submissions/STL2/description.md) and achieved a good performance, especially considering the limited dataset we used compared to many other submissions despite having a more data-hungry end-to-end approach (we have to learn the frequency decomposition front-end from data as well).

### Installation

## Requirements

GPU strongly recommended to avoid very long training times.
Python 2.7
Python package requirements below:

```
sacred==0.7.3
tensorflow-gpu==1.8.0
librosa==0.6.1
scikit-image==0.13.1
soundfile==0.10.2
scikits.audiolab==0.11.0
lxml==4.2.1
musdb==0.2.3
museval==0.2.0
```

These required packages are also saved in the file requirements.txt located in this repository, so you can clone the repository and then execute the following in the downloaded repository's path to install all the required packages at once:

``pip install -r requirements.txt``

### Download datasets

To reproduce the experiments in the paper, you need to download the datasets below. You can of course use your own datasets for separation, but for this you would need to modify the code manually, which will not be discussed here.

#### MUSDB18

Download the full MUSDB18 dataset from https://sigsep.github.io/datasets/musdb.html and extract it into a folder of your choice. It should have two subfolders: "test" and "train" as well as a README.md file.

#### CCMixter (only required for vocal separation experiments)

If you want to replicate the vocal separation experiments and not only the multi-instrument experiments, you also need to download the CCMixter vocal separation database from https://members.loria.fr/ALiutkus/kam/. Extract this dataset into a folder of your choice. Its main folder should contain one subfolder for each song.

### Set-up filepaths

Now you need to set up the correct file paths for the datasets and the location where source estimates should be saved.

Open the ``Training.py`` file, and set the ``musdb_path`` entry of the ``model_config`` dictionary to the location of the main folder of the MUSDB18 dataset.
Also set the ``estimates_path`` entry of the same ``model_config`` dictionary to the path pointing to an empty folder where you want the final source estimates of the model to be saved into.

If you use CCMixter, open the ``CCMixter.xml`` in the main repository folder, and replace the given file path tagged as ``databaseFolderPath`` with your path to the main folder of CCMixter.

## Training/running the experiments

Since the paper investigates many model variants of the Wave-U-Net and also trains the [U-Net proposed for vocal separation](https://ismir2017.smcnus.org/wp-content/uploads/2017/10/171_Paper.pdf), which achieved state-of-the-art performance, as a comparison, we give a list of model variants to train and the command needed to start training them:

| Model name (from paper) | Description                                             | Separate vocals or multi-instrument? | Command for training                          |
|-------------------------|---------------------------------------------------------|--------------------------------------|-----------------------------------------------|
| M1                      | Baseline Wave-U-Net model                               | Vocals                               | ``python Training.py``                            |
| M2                      | M1 + difference output layer                            | Vocals                               | ``python Training.py with baseline_diff``         |
| M3                      | M2 + proper input context                               | Vocals                               | ``python Training.py with baseline_context``      |
| M4                      | BEST-PERFORMING: M3 + Stereo I/O                        | Vocals                               | ``python Training.py with baseline_stereo``       |
| M5                      | M4 + Learned upsampling layer                           | Vocals                               | ``python Training.py with full``                  |
| M6                      | M4 applied to multi-instrument sep.                     | Multi-instrument                     | ``python Training.py with full_multi_instrument`` |
| M7                      | Wave-U-Net model to compare with SotA models U7,U7a     | Vocals                               | ``python Training.py with baseline_comparison``   |
| U7                      | U-Net replication from prior work, audio-based MSE loss | Vocals                               | ``python Training.py with unet_spectrogram``      |
| U7a                     | Like U7, but with L1 magnitude loss                     | Vocals                               | ``python Training.py with unet_spectrogram_l1``   |

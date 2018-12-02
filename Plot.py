import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

import Utils
from Evaluate import compute_mean_metrics


def draw_violin_sdr(json_folder):
    acc, voc = compute_mean_metrics(json_folder, compute_averages=False)
    acc = acc[~np.isnan(acc)]
    voc = voc[~np.isnan(voc)]
    data = [acc, voc]
    inds = [1,2]

    fig, ax = plt.subplots()
    ax.violinplot(data, showmeans=True, showmedians=False, showextrema=False, vert=False)
    ax.scatter(np.percentile(data, 50, axis=1),inds, marker="o", color="black")
    ax.set_title("Segment-wise SDR distribution")
    ax.vlines([np.min(acc), np.min(voc), np.max(acc), np.max(voc)], [0.8, 1.8, 0.8, 1.8], [1.2, 2.2, 1.2, 2.2], color="blue")
    ax.hlines(inds, [np.min(acc), np.min(voc)], [np.max(acc), np.max(voc)], color='black', linestyle='--', lw=1, alpha=0.5)

    ax.set_yticks([1,2])
    ax.set_yticklabels(["Accompaniment", "Vocals"])

    fig.set_size_inches(8, 3.)
    fig.savefig("sdr_histogram.pdf", bbox_inches='tight')

def draw_spectrogram(example_wav="musb_005_angela thomas wade_audio_model_without_context_cut_28234samples_61002samples_93770samples_126538.wav"):
    y, sr = Utils.load(example_wav, sr=None)
    spec = np.abs(librosa.stft(y, 512, 256, 512))
    norm_spec = librosa.power_to_db(spec**2)
    black_time_frames = np.array([28234, 61002, 93770, 126538]) / 256.0

    fig, ax = plt.subplots()
    img = ax.imshow(norm_spec)
    plt.vlines(black_time_frames, [0, 0, 0, 0], [10, 10, 10, 10], colors="red", lw=2, alpha=0.5)
    plt.vlines(black_time_frames, [256, 256, 256, 256], [246, 246, 246, 246], colors="red", lw=2, alpha=0.5)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(img, cax=cax)

    ax.xaxis.set_label_position("bottom")
    #ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * 256.0 / sr))
    #ax.xaxis.set_major_formatter(ticks_x)
    ax.xaxis.set_major_locator(ticker.FixedLocator(([i * sr / 256. for i in range(len(y)//sr + 1)])))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(([str(i) for i in range(len(y)//sr + 1)])))

    ax.yaxis.set_major_locator(ticker.FixedLocator(([float(i) * 2000.0 / (sr/2.0) * 256. for i in range(6)])))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter([str(i*2) for i in range(6)]))

    ax.set_xlabel("t (s)")
    ax.set_ylabel('f (KHz)')

    fig.set_size_inches(7., 3.)
    fig.savefig("spectrogram_example.pdf", bbox_inches='tight')


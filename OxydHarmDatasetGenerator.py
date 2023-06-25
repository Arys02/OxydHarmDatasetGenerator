import random

import matplotlib.pyplot as plt
import os

import numpy
import numpy as np
import librosa.display
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-i", "--in", type=str, help="path for the folder containing mp3 files electronique")
parser.add_argument("-i2", "--in2", type=str, help="path for the folder containing mp3 files accapella")
parser.add_argument("-i3", "--in3", type=str, help="path for the folder containing mp3 files accoustique")
parser.add_argument("-o", "--out", type=str, help="Path for the output folder")
parser.add_argument("-l", "--label", type=str, help="label of dataset")
parser.add_argument("-d", "--duration", type=int, help="duration of the sample 1-5secondes")
parser.add_argument("-img", "--image", action="store_true", help="set if image should be generated")
parser.add_argument("-c", "--csv", action="store_true", help="set if csv should be generated")
parser.add_argument("-s", "--size", type=float, help="size of output square image x*x")

args = parser.parse_args()
config = vars(args)
print(config)

in_folder = config['in']
in2_folder = config['in2']
in3_folder = config['in3']
out_folder = config['out']
duration = config['duration']
image_out = config['image']
csv_out = config['image']
label = config['label']
size = config['size']

audio1_fpath = in_folder
audio2_fpath = in2_folder
audio3_fpath = in3_folder


# for 1 folder

def build_audio_dict(audio_fpath):
    audio_dict = {}
    audio_clips = os.listdir(audio_fpath)
    audio_label = os.path.dirname(audio_fpath).split("/")[-1]
    random.shuffle(audio_clips)
    audio_trainset = audio_clips[0:(len(audio_clips) // 4) * 3]
    audio_testset = audio_clips[(len(audio_clips) // 4) * 3:]
    audio_dict["label"] = audio_label
    audio_dict["fpath"] = audio_fpath
    audio_dict["train"] = audio_trainset
    audio_dict["train_data"] = []
    audio_dict["test"] = audio_testset
    audio_dict["test_data"] = []
    return audio_dict


audio1_dict = build_audio_dict(audio1_fpath)
audio2_dict = build_audio_dict(audio2_fpath)
audio3_dict = build_audio_dict(audio3_fpath)
print(audio1_dict)


def addDictNarray(audio_dict, label="train", withImg=False, offset=1.0, duration=3):
    if label not in ["train", "test"]:
        label = "train"
    for audio_clip in tqdm(audio_dict[label]):
        x, sr = librosa.load(audio_dict['fpath'] + audio_clip, sr=44100, offset=offset, duration=duration)

        # plt.figure(figsize=(14, 5))
        # librosa.display.waveshow(x, sr=sr)
        # plt.show()

        X = librosa.stft(x)
        Xdb = librosa.amplitude_to_db(abs(X))

        # openCV

        plt.figure()
        fig = plt.figure()
        fig.set_size_inches(size, size, forward=False)

        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()

        fig.add_axes(ax)

        librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz', cmap='gray')

        if (withImg):
            plt.savefig(out_folder + audio_clip + ".png")
        plt.show()

        array_from_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        array_from_img_sliced = array_from_img[0:len(array_from_img):3]
        audio_dict[label + '_data'].append(array_from_img_sliced)
    audio_dict[label + "input_size"] = len(audio_dict[label + '_data'])


addDictNarray(audio1_dict, label="train", withImg=False)
addDictNarray(audio1_dict, label="test", withImg=False)

addDictNarray(audio2_dict, label="train", withImg=False)
addDictNarray(audio2_dict, label="test", withImg=False)

addDictNarray(audio3_dict, label="train", withImg=False)
addDictNarray(audio3_dict, label="test", withImg=False)


def buildDataset(main_dict, second_dict1, second_dict2, label):
    data_neg_1 = second_dict1[label + "_data"]
    random.shuffle(data_neg_1)

    data_neg_2 = second_dict2[label + "_data"]
    random.shuffle(data_neg_2)

    data_neg = numpy.concatenate((data_neg_1[0: len(data_neg_1) // 2], data_neg_2[len(data_neg_2) // 2:]))

    positiv_array = [[1]] * len(main_dict[label + "_data"])
    negativ_array = [[0]] * len(data_neg)

    dataset_positiv = numpy.hstack((positiv_array, main_dict[label + "_data"]))
    dataset_negativ = numpy.hstack((negativ_array, data_neg))

    dataset = np.concatenate((dataset_positiv, dataset_negativ))
    return dataset


dataset1_train = buildDataset(audio1_dict, audio2_dict, audio3_dict, "train")
dataset1_test = buildDataset(audio1_dict, audio2_dict, audio3_dict, "test")

dataset2_train = buildDataset(audio2_dict, audio1_dict, audio3_dict, "train")
dataset2_test = buildDataset(audio2_dict, audio1_dict, audio3_dict, "test")

dataset3_train = buildDataset(audio3_dict, audio1_dict, audio2_dict, "train")
dataset3_test = buildDataset(audio3_dict, audio1_dict, audio2_dict, "test")

def buildfolder(data_dict, label, img=False):
    np.savetxt(out_folder + label + '.csv', data_dict, fmt='%d', delimiter=",")

buildfolder(dataset1_test, "electro_test")
buildfolder(dataset1_train, "electro_train")

buildfolder(dataset2_test, "acappella_test")
buildfolder(dataset2_train, "acappella_train")

buildfolder(dataset3_test, "accoust_test")
buildfolder(dataset3_train, "accoust_train")
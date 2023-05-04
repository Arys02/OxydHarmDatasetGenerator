import matplotlib.pyplot as plt
import os
import numpy as np
import librosa.display
import argparse


parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-i", "--in", type=str, help="path for the folder containing mp3 files")
parser.add_argument("-o", "--out", type=str,help="Path for the output folder")
parser.add_argument("-l", "--label", type=str, help="label of dataset")
parser.add_argument("-d", "--duration", type=int, help="duration of the sample 1-5secondes")
parser.add_argument("-img", "--image", action="store_true", help="set if image should be generated")
parser.add_argument("-c", "--csv", action="store_true", help="set if csv should be generated")

args = parser.parse_args()
config = vars(args)
print(config)

in_folder = config['in']
out_folder = config['out']
duration = config['duration']
image_out = config['image']
csv_out = config['image']
label = config['label']


audio_fpath = in_folder

audio_clips = os.listdir(audio_fpath)

offset = 10.0


x, sr = librosa.load(audio_fpath + audio_clips[0], sr=44100, offset=offset, duration=duration)

print(type(x), type(sr))
print(x.shape, sr)

plt.figure(figsize=(14, 5))
librosa.display.waveshow(x, sr=sr)

plt.show()

X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))


plt.figure()
#

sizes = np.shape(Xdb)

fig = plt.figure()
fig.set_size_inches(0.1, 0.1, forward=False)

ax = plt.Axes(fig, [0., 0., 0.1, 0.1])
ax.set_axis_off()

fig.add_axes(ax)


#
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz', cmap='gray')
plt.savefig(out_folder + "8.png")
plt.show()


array_from_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
array_from_img_sliced = array_from_img[0:len(array_from_img):3]
np.savetxt('test32.csv', array_from_img_sliced, fmt='%d', delimiter=",")





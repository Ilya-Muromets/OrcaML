import argparse, os
import glob
import ntpath
import numpy as np
import skimage
import scipy.io.wavfile
import scipy.signal as signal

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=dir_path)
args = parser.parse_args()
path = args.path
filenames = glob.glob(path + "*.wav")
save_path = path + "processed/"
os.makedirs(save_path, exist_ok=True)

for i, filename in enumerate(filenames):
    if i % (len(filenames)//25) == 0:
        print("Finished {0}/{1}".format(i,len(filenames)))
    
    wav = scipy.io.wavfile.read(filename)
    if (len(wav[1])/wav[0] - 2) >= 0.1:
        print(filename, " Skipped. Length over 2 seconds.")
        continue
    wav = signal.resample(wav[1], 44100) # resample to 44100 points, 2 seconds at 22050Hz
    scipy.io.wavfile.write(save_path + ntpath.basename(filename), 22050, wav)    
    
    try:
        f, t, Zxx = signal.stft(wav[:,0], wav[0], nperseg=1000)
    except: 
        f, t, Zxx = signal.stft(wav, wav[0], nperseg=1000)
    
    Zxx = (np.log(np.abs(Zxx))) # log weighted for less scaling noise
    Zxx = skimage.transform.resize(Zxx, (128,128)) # resize image
    # scale to uint8 0-255
    Zxx = ((Zxx - Zxx.min()) * (1/(Zxx.max() - Zxx.min())) * 255).astype('uint8') 
    
    new_filename =  save_path + ntpath.basename(filename)
    new_filename = new_filename.split(".wav")[0] # cut off .wav

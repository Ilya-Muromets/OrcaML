import numpy as np
import glob
import torch
import time
import natsort
import skimage.transform
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

class SpectrogramLoader(Dataset):
    def __init__(self, filepath, resize=(128,128)):
        self.spectrograms = []
        self.labels = []
        for i, folder in enumerate(natsort.natsorted(glob.glob(filepath + "*/"))):
            filenames = glob.glob(folder + "*.npy")
            for filename in filenames:
                spectrogram = np.load(filename)
                if spectrogram.shape != resize:
                    spectrogram = skimage.transform.resize(spectrogram, resize)
                self.spectrograms.append(spectrogram)
                self.labels.append(i)
                
    def __len__(self):
        return len(self.spectrograms)
    
    def __getitem__(self, idx):
        return self.spectrograms[idx], self.labels[idx]
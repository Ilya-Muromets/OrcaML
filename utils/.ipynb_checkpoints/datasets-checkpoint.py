import numpy as np
import glob
import torch
import time
import natsort
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

class ClassComplexLoader(Dataset):
    def __init__(self, data_path='', T1_path='', T2_path='', num_classes=8, max_scans=335):
        self.data_shape = None
        self.T1_class_counts = np.zeros(num_classes)
        self.T2_class_counts = np.zeros(num_classes)
        self.T1_filenames = None
        self.T2_filenames = None
        self.data_filenames = None
        self.T1_class_counts = np.load("data/T1_class_counts.npy").reshape(-1,num_classes)
        self.T1_class_counts = np.mean(self.T1_class_counts, axis=1)
        self.T2_class_counts = np.load("data/T2_class_counts.npy").reshape(-1,num_classes)
        self.T2_class_counts = np.mean(self.T2_class_counts, axis=1)
        
        self.num_classes = num_classes


        self.T1_filenames = natsort.natsorted(glob.glob(T1_path))[0:max_scans]
        print("Found ", len(self.T1_filenames), " T1 files.")

        self.T2_filenames = natsort.natsorted(glob.glob(T2_path))[0:max_scans]
        print("Found ", len(self.T2_filenames), " T2 files.")

        self.data_filenames = natsort.natsorted(glob.glob(data_path))[0:max_scans]
        print("Found ", len(self.data_filenames), " MRF files.")

        test = np.load(self.T1_filenames[0])[0]
        self.data_shape = test.shape
        print("Data shape: ", self.data_shape)

        # check we loaded correct number and shape
        assert len(self.T1_filenames) == len(self.T2_filenames) >= len(self.data_filenames)

        self.T1_filenames = np.array(self.T1_filenames)
        self.T2_filenames = np.array(self.T2_filenames)
        self.data_filenames = np.array(self.data_filenames)

    def __len__(self):
        return len(self.data_filenames)*np.product(self.data_shape)

    def __getitem__(self, idx):
        #retrieve indices
        list_idx = idx//np.product(self.data_shape)
        matrix_idx = idx%np.product(self.data_shape)
        row_idx = matrix_idx//self.data_shape[1]
        column_idx = matrix_idx%self.data_shape[1]
    


        # print(list_idx, column_idx, row_idx)



        MRF = np.load(self.data_filenames[list_idx], mmap_mode="r")
        complex_datum = np.array(MRF[column_idx, row_idx, :]) # index into matrix
        # print(complex_datum.shape)
        # print(time.time() - start)

        # print(complex_datum.shape)
        # stack real component on top of imaginary component of data, shape now (2,len/2)
        complex_datum = complex_datum.reshape(2,500).astype(np.float32)
        # print(time.time() - start)


        T1_file = np.load(self.T1_filenames[list_idx], mmap_mode="r")
        T1 = T1_file[0,row_idx,column_idx]
        T1 = T1.astype(np.float64)/(2**16-1) # assuming uint16
        T1 = np.round((self.num_classes-0.51)*T1, 0).astype(np.int)



        T2_file = np.load(self.T2_filenames[list_idx], mmap_mode="r")
        T2 = T2_file[0,row_idx,column_idx]
        T2 = T2.astype(np.float64)/(2**16-1) # assuming uint16
        T2 = np.round((self.num_classes-0.51)*T2, 0).astype(np.int)


        return complex_datum, T1, T2

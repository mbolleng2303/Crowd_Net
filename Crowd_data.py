
import numpy as np
from torch.utils.data import Dataset


class Crowd_Dataset(Dataset):
    def __init__(self, num_class, num_feature):
        self.data = np.reshape(np.loadtxt("Data/M2.csv", delimiter=",", dtype=float), (-1, num_feature+1))
        self.input = self.data[:, 1:]
        self.nb_people = self.data[:, 0]
        self.num_class = num_class
        self.num_sample = len(self.nb_people)
        self.label = np.zeros_like(self.nb_people)
        self.bin_label = np.zeros((self.num_sample,self.num_class))
        self._make_label()

    def _make_label(self):
        print("label generation : ", self.num_class)
        nbr_smpl_per_class = int(self.num_sample/self.num_class)
        for i in range(self.num_class):
            if i == self.num_class-1:
                self.label[i * nbr_smpl_per_class:] = i
            else:
                self.label[i * nbr_smpl_per_class: (i + 1) * nbr_smpl_per_class] = i
        for i in range(len(self.label)):
            self.bin_label[i, int(self.label[i])] = 1


    def __len__(self):
        return self.num_sample

    def __getitem__(self, idx):
        feature = self.input[idx, :]
        label = self.bin_label[idx,:]

        return feature, label
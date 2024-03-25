import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import copy

class DigitsDataset(Dataset):
    def __init__(self, data_path, channels, percent=0.1, filename=None, setname=None, transform=None, seed=None, known_class=6):
        
        self.known_class=known_class
        self.total_class=10
        self.setname=setname
        #random choose knwon and unknown class
        np.random.seed(seed)
        self.total_classes_perm=np.arange(self.total_class)
        np.random.shuffle(self.total_classes_perm)
        
        self.known_class_list=self.total_classes_perm[:self.known_class]
        self.unknown_class_list=self.total_classes_perm[self.known_class:]
        # print('Known class list:',self.known_class_list,'Unknown class list',self.unknown_class_list)
        
        if filename is None:
            if percent >= 0.1:
                for part in range(int(percent*10)):
                    if part == 0:
                        self.train_images, self.train_labels = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                    else:
                        images, labels = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                        self.train_images = np.concatenate([self.train_images,images], axis=0)
                        self.train_labels = np.concatenate([self.train_labels,labels], axis=0)
            else:
                self.train_images, self.train_labels = np.load(os.path.join(data_path, 'partitions/train_part0.pkl'), allow_pickle=True)
                data_len = int(self.train_images.shape[0] * percent*10)
                self.train_images = self.train_images[:data_len]
                self.train_labels = self.train_labels[:data_len]

            self.test_images, self.test_labels = np.load(os.path.join(data_path, 'test.pkl'), allow_pickle=True)
        else:
            self.train_images, self.train_labels = np.load(os.path.join(data_path, filename), allow_pickle=True)

        self.transform = transform
        self.channels = channels
        self.train_labels = self.train_labels.astype(np.long).squeeze()
        self.test_labels = self.test_labels.astype(np.long).squeeze()

        #relabel dataset
        self.knowndict={}
        self.unknowndict={}
        for i in range(len(self.known_class_list)):
            self.knowndict[self.known_class_list[i]]=i
        for j in range(len(self.unknown_class_list)):
            self.unknowndict[self.unknown_class_list[j]]=j+len(self.known_class_list)
        # if setname=='train':
        #     print(self.knowndict,self.unknowndict)

        self.copytrainy=copy.deepcopy(self.train_labels)
        self.copytesty=copy.deepcopy(self.test_labels)
        for i in range(len(self.known_class_list)):
            self.train_labels[self.copytrainy==self.known_class_list[i]]=self.knowndict[self.known_class_list[i]]
            self.test_labels[self.copytesty==self.known_class_list[i]]=self.knowndict[self.known_class_list[i]]
        for j in range(len(self.unknown_class_list)):
            self.train_labels[self.copytrainy==self.unknown_class_list[j]]=self.unknowndict[self.unknown_class_list[j]]
            self.test_labels[self.copytesty==self.unknown_class_list[j]]=self.unknowndict[self.unknown_class_list[j]]
        self.origin_known_list=self.known_class_list
        self.origin_unknown_list=self.unknown_class_list
        self.new_known_list=np.arange(self.known_class)
        self.new_unknown_list=np.arange(self.known_class,self.known_class+len(self.unknown_class_list))


        self.trian_data_known_index=[]
        self.test_data_known_index=[]
        for item in self.new_known_list:
            index=np.where(self.train_labels==item)
            index=list(index[0])
            self.trian_data_known_index=self.trian_data_known_index+index
            index=np.where(self.test_labels==item)
            index=list(index[0])
            self.test_data_known_index=self.test_data_known_index+index
        
        self.train_data_index_perm=np.arange(len(self.train_labels))
        self.train_data_unknown_index=np.setdiff1d(self.train_data_index_perm,self.trian_data_known_index)
        self.test_data_index_perm=np.arange(len(self.test_labels))
        self.test_data_unknown_index=np.setdiff1d(self.test_data_index_perm,self.test_data_known_index)
        
        assert (len(self.test_data_unknown_index)+len(self.test_data_known_index)==len(self.test_labels))


        self.transform = transform


        if setname=="train":
            self.datax=(self.train_images[self.trian_data_known_index])
            self.datay=(self.train_labels[self.trian_data_known_index])
        elif setname=="testclose":
            self.datax=(self.test_images[self.test_data_known_index])
            self.datay=(self.test_labels[self.test_data_known_index])
        elif setname=="testopen":
            self.datax=(self.test_images[self.test_data_unknown_index])
            self.datay=(self.test_labels[self.test_data_unknown_index])

    def __len__(self):
        return len(self.datay)

    def known_class_show(self):
        return self.origin_known_list,self.origin_unknown_list

    def __getitem__(self, idx):
        image = self.datax[idx]
        label = self.datay[idx]
        if self.channels == 1:
            image = Image.fromarray(image, mode='L')
        elif self.channels == 3:
            image = Image.fromarray(image, mode='RGB')
        else:
            raise ValueError("{} channel is not allowed.".format(self.channels))

        if self.transform is not None:
            image = self.transform(image)

        return image, label
        

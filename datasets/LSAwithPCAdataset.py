import torch

from torch.utils.data import Dataset, Subset, ConcatDataset
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import torchaudio
import torchio as tio
import os
import numpy as np
import random
from collections import defaultdict

from tqdm import tqdm
import nibabel as nib



class LSAwithPCADataset(Dataset):

    def __init__(self,
                 LSAPath = "datas/LSA_with_pca",
                 device = 'cpu',
                 Foldids: list = None,
                 MeanFoldInstrument = False,
                 Augmentation = False,
                 BrainMask = False,
                 PCAModel = None
                 ):
        self.BrainImagePaths = []
        self.BrainImageMetadata = defaultdict(dict)
        self.all_Labels = []
        self.augmentation = Augmentation
        self.brainMask = BrainMask
        self.PCAModel = PCAModel
        self.eps = 1e-15
        self.LSAPath = LSAPath

        for fileName in os.listdir(self.LSAPath):
            if fileName.endswith('nii') and not fileName.startswith('amask'):
                filePath = os.path.join(os.getcwd(), self.LSAPath, fileName)
                self.BrainImagePaths.append(filePath)

                xSplit = fileName.split("_")

            
                Instrument = xSplit[2]
                self.all_Labels.append(Instrument)

                
                Fold = xSplit[1]

                self.BrainImageMetadata[filePath] = {
                                            "Filename":fileName,
                                            "Instrument":Instrument,
                                            "Fold":Fold
                                            }
        # Select folds
        if Foldids:
            BrainImagePathsTemp = []
            for filePath in self.BrainImagePaths:
                if int(self.BrainImageMetadata[filePath]['Fold']) in Foldids:
                    BrainImagePathsTemp.append(filePath)

            self.BrainImagePaths = BrainImagePathsTemp
                    

        if MeanFoldInstrument:
            self.BrainImagePaths = []
            for dirPath, _, fileNames in os.walk("MeanFoldInstrumentLSA"):
                for fileName in fileNames:
                    #print(fileName.split('_')[0].split('d'))
                    if int(fileName.split('_')[0].split('d')[1]) in Foldids:
                        self.BrainImagePaths.append(dirPath+'/'+fileName)
                        self.BrainImageMetadata[dirPath+'/'+fileName]['Instrument'] = fileName.split('_')[1].split('.')[0]
        
        
        
        self.le = LabelEncoder()
        self.le.fit(self.all_Labels)


    def __len__(self):
        return len(self.BrainImagePaths)

    def __getitem__(self, index):
        path = self.BrainImagePaths[index]
        try:
            # Read file
            BrainImage = nib.load(path)
            # Get raw data
            BrainImage = BrainImage.get_fdata()
            BrainImage = np.array(BrainImage)
            BrainImage = torch.from_numpy(BrainImage)
            BrainImage = BrainImage.unsqueeze(0)
        except:
            print('Cannot open file:', path)

        genre = self.BrainImageMetadata[path]['Instrument']
        OneHotGenre = self._GetOneHotCode(genre)
        
        
        if self.augmentation:
            random_affine = tio.RandomAffine(degrees=20,
                                            scales=0,
                                            )
            BrainImage = random_affine(BrainImage)

        if self.brainMask:
            BrainImage = self._AC_Masking(BrainImage)

        #BrainImage = torch.where(BrainImage>1.0,1.0,BrainImage)
        #BrainImage = torch.where(BrainImage<0.5,0.5,BrainImage)

        return BrainImage, OneHotGenre


    def _GetOneHotCode(self, genre):
        label = self.le.transform([genre])
        label = torch.tensor(label)
        label = F.one_hot(label, num_classes=10)
        #print(label[index], self._inverse_one_hot_encode(label, index))
        return label[0]
    

    def _getGenre(self, OneHotGenre):
        return self.le.inverse_transform([np.array(OneHotGenre).argmax()])[0]

    def _AC_Masking(self, BrainImage):
        BrainMask = nib.load('./LSA/mask/AC_mask_downsampled.nii')
        BrainMask = BrainMask.get_fdata()
        BrainMask = np.array(BrainMask)
        BrainMask = torch.from_numpy(BrainMask)
        BrainMask = BrainMask.unsqueeze(0)

        BrainImage*=BrainMask
        return BrainImage
    

    

    


if __name__ == "__main__":
    Images = torch.Tensor([])

    for fid in range(1,17):

        LSA = LSAwithPCADataset(Foldids=[fid], MeanFoldInstrument=False, Augmentation=False, BrainMask=False)
        print('Brain images:', len(LSA), f'Fold {fid}')
        for v,g in LSA:
            
            print(v.min(), v.max(), v.mean(), v.std(), LSA._getGenre(g))
    

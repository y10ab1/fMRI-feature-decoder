import torch
import torchaudio
import torchio as tio
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, ConcatDataset


from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
import random
from collections import defaultdict, Counter

from tqdm import tqdm
import nibabel as nib

from utils import image_masking_3D, clip_brainImage, get_brainImage, load_dict



class LSA_local_brain_dataset(Dataset):

    def __init__(self,
                 LSAPath = "datas/LSA_better",
                 device = 'cpu',
                 Foldids: list = None,
                 MeanFoldInstrument = False,
                 Augmentation = False,
                 BrainMask = False,
                 PCAModel = None,
                 genre_type = 'MidiCode',
                 coordinate = (0,0,0),
                 side_length = -1
                 ):

        self.BrainImagePaths = []
        self.BrainImageMetadata = defaultdict(dict)
        self.all_Labels = []
        self.augmentation = Augmentation
        self.brainMask = BrainMask
        self.PCAModel = PCAModel
        self.LSAPath = LSAPath
        self.genre_type = genre_type
        self.side_length = side_length
        self.coordinate = coordinate
        self.device = device

        print("Load from:", self.LSAPath)
        for fileName in os.listdir(self.LSAPath):
            
            if fileName.endswith('nii') and not fileName.startswith('amask'):
                filePath = os.path.join(os.getcwd(), self.LSAPath, fileName)
                self.BrainImagePaths.append(filePath)

                xSplit = fileName.split("_")
                xSplit2 = xSplit[-1].split("-")

                if len(xSplit) > 3:
                    # Instrument = xSplit[0] + '_' + xSplit[1]
                    Instrument =  xSplit[1] # zrep_instrument -> instrument
                else:
                    Instrument = xSplit[0]

                MainTimbre = xSplit[-2]
                SubTimbre = xSplit2[0]
                MidiCode = xSplit2[1]
                #MidiCode = 'Low' if int(MidiCode) < 74 else 'High' # Classify between only high and low pitch

                Fold = xSplit2[2].split('.')[0]

                self.BrainImageMetadata[filePath] = {
                                            "Filename":fileName,
                                            "Instrument":Instrument,
                                            "MainTimbre":MainTimbre, 
                                            "SubTimbre":SubTimbre, 
                                            "MidiCode":MidiCode, 
                                            "Fold":Fold
                                            }

                self.all_Labels.append(self.BrainImageMetadata[filePath][self.genre_type])

        # Select folds
        if Foldids:
            BrainImagePathsTemp = []
            for filePath in self.BrainImagePaths:
                if int(self.BrainImageMetadata[filePath]['Fold']) in Foldids:
                    BrainImagePathsTemp.append(filePath)

            self.BrainImagePaths = BrainImagePathsTemp
                    

        if MeanFoldInstrument:
            self.BrainImagePaths = []
            for dirPath, _, fileNames in os.walk("MeanFoldInstrumentLSA_better"):
                for fileName in fileNames:
                    #print(fileName.split('_')[0].split('d'))
                    if int(fileName.split('_')[0].split('d')[1]) in Foldids:
                        self.BrainImagePaths.append(dirPath+'/'+fileName)
                        self.BrainImageMetadata[dirPath+'/'+fileName]['Instrument'] = fileName.split('_')[1].split('.')[0]
        
        
        
        self.le = LabelEncoder()
        self.le.fit(self.all_Labels)
        self.num_classes = len(Counter(self.all_Labels))

        self.all_BrainImages = self.get_all_BrainImages()
        self.all_OneHotGenres = self.get_all_OneHotGenres()

        #print(self.le.classes_)
        #print((self.le.inverse_transform(list(range(25)))), list(range(25)))


    def __len__(self):
        return len(self.BrainImagePaths)

    def __getitem__(self, index):
        BrainImage = self.all_BrainImages[index]
        OneHotGenre = self.all_OneHotGenres[index]
        
        # Set nan(if any) to 0
        BrainImage = torch.where(BrainImage.isnan(), 0.0, BrainImage)


        if self.augmentation:
            random_affine = tio.RandomAffine(degrees=20,
                                            scales=0,
                                            )
                                            
            BrainImage = random_affine(BrainImage)
            BrainImage = BrainImage.float()

        if self.brainMask:
            BrainImage = self._AC_Masking(BrainImage)

        BrainImage = BrainImage[:,:,:,10:-10] # Up and down clipping along z axis
        #print(Counter(self.all_Labels))

        if self.side_length > 0:
            BrainImage = clip_brainImage(BrainImage.squeeze(), self.coordinate, self.side_length).unsqueeze(0)          
        return BrainImage, OneHotGenre
    
    def get_all_BrainImages(self):
        all_BrainImages = []

        for path in tqdm(self.BrainImagePaths):
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

            #all_BrainImages = torch.cat((all_BrainImages, BrainImage.unsqueeze(0)), 0)
            all_BrainImages.append(BrainImage)
        all_BrainImages = torch.stack(all_BrainImages)

        return all_BrainImages.to(self.device)
        

    def get_all_OneHotGenres(self):
        all_OneHotGenres = []
        for path in tqdm(self.BrainImagePaths):
            genre = self.BrainImageMetadata[path][self.genre_type]
            OneHotGenre = self._GetOneHotCode(genre)
            
            all_OneHotGenres.append(OneHotGenre)
        all_OneHotGenres = torch.stack(all_OneHotGenres)
        
        return all_OneHotGenres.to(self.device)


    def _GetOneHotCode(self, genre):
        label = self.le.transform([genre])
        label = torch.tensor(label)
        label = F.one_hot(label, num_classes=self.num_classes)
        #print(label[index], self._inverse_one_hot_encode(label, index))
        return label[0]
    
    

    def _getGenre(self, OneHotGenre):
        return self.le.inverse_transform([np.array(OneHotGenre.cpu()).argmax()])[0]

    def _AC_Masking(self, BrainImage):
        BrainMask = nib.load('./LSA/mask/AC_mask_downsampled.nii')
        BrainMask = BrainMask.get_fdata()
        BrainMask = np.array(BrainMask)
        BrainMask = torch.from_numpy(BrainMask)
        BrainMask = BrainMask.unsqueeze(0)

        BrainImage*=BrainMask
        return BrainImage
    
def MeanFoldInstrumentPreprocess(LSA):
    SumGenreImage = {}
    SumGenreImageCount = defaultdict(int)

    for idx, (image, genre) in enumerate(LSA):
        #print(genre, idx)
        #print(image.shape, type(image))
        #print(image.max(), image.min(), image.mean())
        try:
            SumGenreImage[tuple(genre.tolist())]+=image
            SumGenreImageCount[tuple(genre.tolist())]+=1
        except:
            SumGenreImage[tuple(genre.tolist())]=image
            SumGenreImageCount[tuple(genre.tolist())]=1

    for k,v in SumGenreImage.items():
        v /= SumGenreImageCount[k]
        print(k, SumGenreImageCount[k])
        #print(v)
        print(v.shape, v.max(), v.min())
    
        ima = nib.Nifti1Image(v[0].numpy(), np.eye(4))
        #ima.set_data_dtype(np.my_dtype)

        nib.save(ima, f'MeanFoldInstrumentLSA_better/Fold{fid}_{LSA._getGenre(k)}.nii')



    


    


if __name__ == "__main__":
    LSA = LSA_local_brain_Dataset(LSAPath = "../datas/LSA_better", Foldids=list(range(1,2)))
    print(LSA)
    # Images = torch.Tensor([])
    # for fid in range(1,2):

    #     LSA = LSA_local_brain_Dataset(Foldids=[fid], MeanFoldInstrument=False, Augmentation=False, BrainMask=False)
    #     print('Brain images:', len(LSA), f'Fold {fid}')

        
    #     for v,g in LSA:
    #         v = torch.where(v.isnan(),0.0,v)
    #         new_v = v.squeeze()
    #         #new_v = image_masking_3D(new_v.cuda())
    #         from pca_preprocess import Save_3DImage_to_nii
    #         #Save_3DImage_to_nii(new_v.cpu(), 'testmasking-5x5x5-60.nii')
    #         print(new_v.shape, g)
    #         print('Max:',new_v.max(),'Min:', new_v.min())
    #         print('Voxels of nan:', torch.sum(torch.isnan(new_v)))
    #         print('----------------------------------------------')

    #         from utils import scan_the_brain, clip_brainImage, integrate_brainImage_clips
    #         side_length = 8

    #         coords = scan_the_brain(new_v, side_length = side_length, hop_length = side_length)
    #         print(coords, len(coords))
    #         new_image = integrate_brainImage_clips(new_v, coords, side_length)

    #         Save_3DImage_to_nii(new_image.cpu()*10, 'clipping.nii')
    #         #print(new_image, new_image.shape)
    #         break
        
            
        # '''
        # SumGenreImage = {}
        # SumGenreImageCount = defaultdict(int)

        # for idx, (image, genre) in enumerate(LSA):
        #     #print(genre, idx)
        #     #print(image.shape, type(image))
        #     #print(image.max(), image.min(), image.mean())
        #     try:
        #         SumGenreImage[tuple(genre.tolist())]+=image
        #         SumGenreImageCount[tuple(genre.tolist())]+=1
        #     except:
        #         SumGenreImage[tuple(genre.tolist())]=image
        #         SumGenreImageCount[tuple(genre.tolist())]=1
        
            
        
            
        # print(LSA.le.classes_)
        # #???std
        # #????????????????????????????????? (??????????????????nii????????????header ???????????????header)
        # for k,v in SumGenreImage.items():
        #     v /= SumGenreImageCount[k]
        #     print(k, SumGenreImageCount[k])
        #     #print(v)
        #     print(v.shape, v.max(), v.min())
        
        #     ima = nib.Nifti1Image(v[0].numpy(), np.eye(4))
        #     #ima.set_data_dtype(np.my_dtype)

        #     nib.save(ima, f'MeanFoldInstrumentLSA/Fold{fid}_{LSA._getGenre(k)}.nii')
        # '''

        #over sampling
        #mean instrument
        

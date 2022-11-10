import os
print('GPUs:', os.environ.get("CUDA_VISIBLE_DEVICES"))
import argparse
import torch
import torchaudio
import numpy as np
import time
from sklearn.metrics import f1_score

from LSARegressionTrainer import model_trainer
from models.model_for_local_brain_regression import AAEEncoder

from statistics import mean
#from datasets.LSAdataset import LSADataset
from datasets.LSA_local_brain_dataset import LSA_local_brain_dataset
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset, random_split

from scipy import stats
from utils import generate_local_brain_dataset, scan_the_brain, Save_3DImage_to_nii, get_templateImage, plot_samplepoints,write_training_config_Csv

import shap

# Seeds
torch.manual_seed(0)
np.random.seed(0)

def get_parser():
    
    parser = argparse.ArgumentParser("CNN", description='Hyperparameter setting.')
    parser.add_argument('--BATCH_SIZE', default=128, type=int, help='')
    parser.add_argument('--EPOCHS', default=200, type=int, help='')
    parser.add_argument('--LEARNING_RATE', default=0.005, type=float, help='')
    parser.add_argument('--L1_WEIGHT', default=1e-5, type=float, help='')
    parser.add_argument('--L2_WEIGHT', default=1e-5, type=float, help='')
    parser.add_argument('--EPS', default=1e-15, type=float, help='')
    parser.add_argument('--MASK', default=False, type=bool, help='')
    parser.add_argument('--NUM_MASK', default=15, type=int, help='')
    parser.add_argument('--MASK_SIZE', default=2, type=int, help='')
    parser.add_argument('--DATE', default=time.ctime(), type=str, help='')
    parser.add_argument('--SUFFIX', default='', type=str, help='')
    parser.add_argument('--MODEL_NAME', default='model.pth', type=str, help='')
    parser.add_argument('--GENRE_TYPE', default='MidiCode', type=str, choices=['Instrument', 'MidiCode', 'MainTimbre', 'SubTimbre'], help='')
    parser.add_argument('--LOCAL_BRAIN_SIDE_LENGTH', default=20, type=int, help='')
    parser.add_argument('--LOCAL_BRAIN_HOP_LENGTH', default=20, type=int, help='')
    parser.add_argument('--SAVE_DIR_PATH', default=f'Results/{time.ctime().replace(" ","-")}', type=str, help='')
    parser.add_argument('--LSA_DIR_PATH', default=f'', type=str, help='')
    parser.add_argument('--GPU', default=f"0", type=str, help='')

    return parser



def prepare_local_brain_LSAdataset(device, genre_type, coordinate, side_length, trainFolds=None, testFolds=None, lsa_path=None, vecotrPointsPath=None):
    
    # Load LSA dataset
    test_set = LSA_local_brain_dataset(device=device, 
                                        Foldids = testFolds,
                                        LSAPath = lsa_path,
                                        genre_type=genre_type, 
                                        coordinate=coordinate, 
                                        side_length=side_length)

    trainFolds, validFolds = random_split(trainFolds, (8,1))

    

    train_set = LSA_local_brain_dataset(device=device, 
                                        Foldids = trainFolds, 
                                        genre_type=genre_type, 
                                        coordinate=coordinate, 
                                        side_length=side_length,
                                        LSAPath = lsa_path)
    valid_set = LSA_local_brain_dataset(device=device, 
                                        Foldids = validFolds, 
                                        genre_type=genre_type, 
                                        coordinate=coordinate, 
                                        side_length=side_length,
                                        LSAPath = lsa_path)
    
    return (train_set, valid_set, test_set)




if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    
    

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    print(f"Using {device}")
        
    # Creating saving folder.
    try:
        os.makedirs(f'{args.SAVE_DIR_PATH}')
        print(f'{args.SAVE_DIR_PATH} created.')
    except OSError as error:
        print(f'{args.SAVE_DIR_PATH} existed.')

    
    four_folds = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]

    

    for fold in range(4):
        

        
        trainFolds = four_folds[(fold+1)%4] + four_folds[(fold+2)%4] + four_folds[(fold+3)%4]

        mini_four_folds = trainFolds.copy()
        for mini_fold in range(0,len(mini_four_folds),3):
            mini_testFolds = mini_four_folds[mini_fold:mini_fold+3]
            mini_trainFolds = list(set(mini_four_folds) - set(mini_testFolds))

            # Dataset part

            sample_brain_image = get_templateImage()


            side_length = args.LOCAL_BRAIN_SIDE_LENGTH # Determine local brain cube side length
            coordinates = scan_the_brain(image = sample_brain_image.squeeze(), 
                                        side_length = args.LOCAL_BRAIN_SIDE_LENGTH , 
                                        hop_length = args.LOCAL_BRAIN_HOP_LENGTH)
            print(coordinates)
            train_set, valid_set, test_set = prepare_local_brain_LSAdataset(device = device,
                                                                            trainFolds = mini_trainFolds, 
                                                                            testFolds = mini_testFolds, 
                                                                            genre_type = args.GENRE_TYPE,
                                                                            coordinate = (0,0,0),
                                                                            side_length = side_length,
                                                                            lsa_path = args.LSA_DIR_PATH)
            for coordinate in coordinates:
                # if coordinate != (60, 40,20) and coordinate !=(0,40,20):
                #     continue
                train_set.coordinate = coordinate
                valid_set.coordinate = coordinate
                test_set.coordinate = coordinate

                train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=False)

                os.makedirs(f'{args.SAVE_DIR_PATH}/fold{fold}/minifold{mini_fold//3}/{coordinate}',exist_ok=True)
                print(f'{args.SAVE_DIR_PATH}/fold{fold}/minifold{mini_fold//3}/{coordinate} created.')
            


                # Load model from model.pth
                # Model part
                model = AAEEncoder(num_classes=1)
                model = torch.nn.DataParallel(model)
                model.to(device)
                model.load_state_dict(torch.load(f'{args.SAVE_DIR_PATH}/fold{fold}/minifold{mini_fold//3}/{coordinate}/model.pth'))
                model.eval()

                # Compute SHAP values
                shap_values_list = []
                for brainimage, label in train_loader:
                
                    brainimage = brainimage.float().to(device)



                    explainer = shap.DeepExplainer(model, brainimage)
                    shap_values = explainer.shap_values(brainimage)
                    print("Shap values shape:",shap_values.shape) # (128,1, 20,20,20)
                    print("Shap values type:",type(shap_values)) # numpy.ndarray
                    shap_values_list.append(shap_values)
                
                shap_values = np.concatenate(shap_values_list, axis=0).squeeze()
                print("Shap values shape:",shap_values.shape) # (400, 20,20,20)
                # Absoulte value of shap values
                abs_shap_values = np.abs(shap_values)
                # Print min and max of absolute shap values
                print("Min of absolute shap values:",np.min(abs_shap_values))
                print("Max of absolute shap values:",np.max(abs_shap_values))
                # Average over first dimension

                shap_values = np.mean(abs_shap_values, axis=0)
                # Save shap values
                np.save(f'{args.SAVE_DIR_PATH}/fold{fold}/minifold{mini_fold//3}/{coordinate}/shap_values.npy', shap_values)
                torch.cuda.empty_cache() 

            # Remove dataset tensor to free GPU memory
            del train_set.all_BrainImages, valid_set.all_BrainImages, test_set.all_BrainImages
            del train_set.all_OneHotGenres, valid_set.all_OneHotGenres, test_set.all_OneHotGenres
            torch.cuda.empty_cache() 

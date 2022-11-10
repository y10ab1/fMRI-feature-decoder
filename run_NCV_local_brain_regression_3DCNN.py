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

    # Model part
    
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

                os.makedirs(f'{args.SAVE_DIR_PATH}/fold{fold}/minifold{mini_fold//3}/{coordinate}',exist_ok=True)
                print(f'{args.SAVE_DIR_PATH}/fold{fold}/minifold{mini_fold//3}/{coordinate} created.')
            

                # training part
                trainer = model_trainer(TRAIN_SET = train_set, 
                                        VALID_SET = valid_set, 
                                        TEST_SET = test_set, 
                                        AAEEncoder = AAEEncoder,
                                        EPOCHS = args.EPOCHS,
                                        BATCH_SIZE = args.BATCH_SIZE,
                                        FOLDID = trainFolds,
                                        LEARNING_RATE = args.LEARNING_RATE,
                                        L2_WEIGHT = args.L2_WEIGHT,
                                        L1_WEIGHT = args.L1_WEIGHT,
                                        MASK = args.MASK,
                                        NUM_MASK = args.NUM_MASK,
                                        MASK_SIZE = args.MASK_SIZE,
                                        DATE = args.DATE,
                                        GENRE_TYPE = args.GENRE_TYPE,
                                        MODEL_NAME = args.MODEL_NAME,
                                        COORDINATE = coordinate,
                                        SUFFIX = args.SUFFIX+f'coordinate-{coordinate}',
                                        SAVE_DIR_PATH = f'{args.SAVE_DIR_PATH}/fold{fold}/minifold{mini_fold//3}/{coordinate}')
                trainer.start_training()
                write_training_config_Csv(f"{args.SAVE_DIR_PATH}/fold{fold}/minifold{mini_fold//3}/{coordinate}/trainingConfig.csv",                    
                                            test_loss = trainer.test_loss,
                                            Date = args.DATE,
                                            TrainFolds = trainFolds,
                                            TestFolds = list(set(range(1,17))-set(trainFolds)),
                                            mini_TrainFolds = mini_trainFolds,
                                            mini_TestFolds = mini_testFolds,
                                            Epochs = args.EPOCHS,
                                            BatchSize = args.BATCH_SIZE,
                                            LearningRate = args.LEARNING_RATE,
                                            side_length = args.LOCAL_BRAIN_SIDE_LENGTH,
                                            hop_length = args.LOCAL_BRAIN_HOP_LENGTH,
                                            genre_type = args.GENRE_TYPE,
                                            LSAPath = args.LSA_DIR_PATH,
                                            loss = trainer.test_loss,
                                            Coordinate = coordinate,
                                            corrcoef = trainer.corrcoef,
                                            P_value = trainer.p_value,
                                            spearman_p_value = trainer.spear_p_value,
                                            spearman_corrcoef = trainer.spearcoef)



                # Highlighting local brain
                templateImage = get_templateImage()
                x,y,z = coordinate
                templateImage[x:x+side_length, y:y+side_length, z:z+side_length]=500
                
                x_length, y_length, z_length = templateImage.shape
                save_image = (torch.ones((x_length, y_length, z_length+20))).double()
                save_image[:,:,10:-10] = templateImage.double()
                Save_3DImage_to_nii(save_image, Path= f'{args.SAVE_DIR_PATH}/fold{fold}/minifold{mini_fold//3}/{coordinate}/localBrain-Highlight.nii')


                torch.cuda.empty_cache() 

            # Remove dataset tensor to free GPU memory
            del train_set.all_BrainImages, valid_set.all_BrainImages, test_set.all_BrainImages
            del train_set.all_OneHotGenres, valid_set.all_OneHotGenres, test_set.all_OneHotGenres
            torch.cuda.empty_cache() 

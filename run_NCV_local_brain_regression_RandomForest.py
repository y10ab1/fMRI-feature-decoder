import os
print('GPUs:', os.environ.get("CUDA_VISIBLE_DEVICES"))
import argparse
import torch
import torchaudio
import numpy as np
import time
from sklearn.metrics import f1_score
from sklearn.svm import SVR, NuSVR
from sklearn.ensemble import RandomForestRegressor

from LSARegressionTrainer import model_trainer

from statistics import mean
from datasets.LSA_brainVectors_dataset import LSA_brainVectors_dataset

from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset, random_split

from scipy import stats
from utils import generate_local_brain_dataset, scan_the_brain, Save_3DImage_to_nii, get_templateImage, plot_samplepoints,\
                    write_training_config_Csv

import pickle
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
    parser.add_argument('--MODEL_NAME', default='', type=str, help='')
    parser.add_argument('--GENRE_TYPE', default='MidiCode', type=str, choices=['Instrument', 'MidiCode', 'MainTimbre', 'SubTimbre'], help='')
    parser.add_argument('--LOCAL_BRAIN_SIDE_LENGTH', default=20, type=int, help='')
    parser.add_argument('--LOCAL_BRAIN_HOP_LENGTH', default=20, type=int, help='')
    parser.add_argument('--SAVE_DIR_PATH', default=f'Results/{time.ctime().replace(" ","-")}', type=str, help='')
    parser.add_argument('--LSA_DIR_PATH', default=f'', type=str, help='')
    parser.add_argument('--TOP_K_SHAP_VALUE', default=80, type=int, help='')
    parser.add_argument('--GPU', default=f"0", type=str, help='')

    return parser



def prepare_local_brain_LSAdataset(device, genre_type, coordinate, side_length, trainFolds=None, testFolds=None, lsa_path=None, vecotrPointsPath=None):


    test_set = LSA_brainVectors_dataset(device=device, 
                                        Foldids = testFolds,
                                        LSAPath = lsa_path,
                                        coordinate = coordinate,
                                        side_length = side_length,
                                        vecotrPointsPath = vecotrPointsPath,
                                        genre_type=genre_type)

    trainFolds, validFolds = random_split(trainFolds, (8,1))
    

    train_set = LSA_brainVectors_dataset(device=device, 
                                        Foldids = trainFolds, 
                                        genre_type=genre_type,
                                        coordinate = coordinate,
                                        side_length = side_length,
                                        vecotrPointsPath = vecotrPointsPath,
                                        LSAPath = lsa_path)
    valid_set = LSA_brainVectors_dataset(device=device, 
                                        Foldids = validFolds, 
                                        genre_type=genre_type, 
                                        coordinate = coordinate,
                                        side_length = side_length,
                                        vecotrPointsPath = vecotrPointsPath,
                                        LSAPath = lsa_path)
    
    return (train_set, valid_set, test_set)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    
    

    # if torch.cuda.is_available():
    #     device = "cuda"
    # else:
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
            train_set, valid_set, test_set = prepare_local_brain_LSAdataset(device = 'cpu',
                                                                            trainFolds = mini_trainFolds, 
                                                                            testFolds = mini_testFolds, 
                                                                            genre_type = args.GENRE_TYPE,
                                                                            coordinate = (0,0,0),
                                                                            side_length = side_length,
                                                                            lsa_path = args.LSA_DIR_PATH)
            for coordinate in coordinates:
                train_set.coordinate = coordinate
                valid_set.coordinate = coordinate
                test_set.coordinate = coordinate

                os.makedirs(f'{args.SAVE_DIR_PATH}/fold{fold}/minifold{mini_fold//3}/{coordinate}',exist_ok=True)
                print(f'{args.SAVE_DIR_PATH}/fold{fold}/minifold{mini_fold//3}/{coordinate} created.')

                
                

                # training part
                x, y = [], []
                for brainimage, label in train_set:
                    x.append((brainimage.squeeze().cpu().numpy()))
                    y.append(label.cpu().numpy().argmax(0))
                
                for brainimage, label in valid_set:
                    x.append((brainimage.squeeze().cpu().numpy()))
                    y.append(label.cpu().numpy().argmax(0))







                x_test, y_test = [], []

                for brainimage, label in test_set:
                    x_test.append((brainimage.squeeze().cpu().numpy()))
                    y_test.append(label.cpu().numpy().argmax(0))


                model = RandomForestRegressor(n_jobs=-1,verbose=1)
                model.fit(x,y)


                # Save model
                pkl_filename = f"{args.SAVE_DIR_PATH}/fold{fold}/minifold{mini_fold//3}/{coordinate}/model.pkl"
                with open(pkl_filename, 'wb') as file:
                    pickle.dump(model, file)


                
                
                # explain the model's predictions using SHAP
                # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
                explainer = shap.Explainer(model)
                shap_values = explainer(np.array(x))
                print("Shap values shape:",shap_values.shape) # (600, 8000)

                print("Shap values type:",type(shap_values.values)) # numpy.ndarray



                ############################################################################################################
                # extract top k shap values voxels value
                mean_abs_shap_values = np.mean(np.abs(np.array(shap_values.values)), axis=0)
                print("Mean abs shap values shape:",mean_abs_shap_values.shape) # (8000,)
                top_k_voxel_index = np.argsort(mean_abs_shap_values)[-args.TOP_K_SHAP_VALUE:]
                selected_voxels = np.array(x)
                print("Selected voxels shape:",selected_voxels.shape) # (600, 80)

                # save selected voxels
                pkl_filename = f"{args.SAVE_DIR_PATH}/fold{fold}/minifold{mini_fold//3}/{coordinate}/top{args.TOP_K_SHAP_VALUE}-train_voxels_labels.pkl"
                with open(pkl_filename, 'wb') as file:
                    pickle.dump((selected_voxels[:,top_k_voxel_index],np.array(y)), file)
                pkl_filename = f"{args.SAVE_DIR_PATH}/fold{fold}/minifold{mini_fold//3}/{coordinate}/top{800}-train_voxels_labels.pkl"
                with open(pkl_filename, 'wb') as file:
                    pickle.dump((selected_voxels[:,800],np.array(y)), file)
                pkl_filename = f"{args.SAVE_DIR_PATH}/fold{fold}/minifold{mini_fold//3}/{coordinate}/top{4000}-train_voxels_labels.pkl"
                with open(pkl_filename, 'wb') as file:
                    pickle.dump((selected_voxels[:,4000],np.array(y)), file)

                selected_voxels = np.array(x_test)

                pkl_filename = f"{args.SAVE_DIR_PATH}/fold{fold}/minifold{mini_fold//3}/{coordinate}/top{args.TOP_K_SHAP_VALUE}-test_voxels_labels.pkl"
                with open(pkl_filename, 'wb') as file:
                    pickle.dump((selected_voxels[:,top_k_voxel_index],np.array(y_test)), file)
                pkl_filename = f"{args.SAVE_DIR_PATH}/fold{fold}/minifold{mini_fold//3}/{coordinate}/top{800}-test_voxels_labels.pkl"
                with open(pkl_filename, 'wb') as file:
                    pickle.dump((selected_voxels[:,800],np.array(y_test)), file)
                pkl_filename = f"{args.SAVE_DIR_PATH}/fold{fold}/minifold{mini_fold//3}/{coordinate}/top{4000}-test_voxels_labels.pkl"
                with open(pkl_filename, 'wb') as file:
                    pickle.dump((selected_voxels[:,4000],np.array(y_test)), file)
                ############################################################################################################
                

                # visualize the first prediction's explanation and save plot
                os.makedirs(f'{args.SAVE_DIR_PATH}/fold{fold}/minifold{mini_fold//3}/{coordinate}/shap-summary',exist_ok=True)
                import matplotlib.pyplot as plt

                shap.summary_plot(shap_values,features=np.array(x),show=False, plot_type="bar")
                plt.savefig(f'{args.SAVE_DIR_PATH}/fold{fold}/minifold{mini_fold//3}/{coordinate}/shap-summary/summary-bar.png')
                plt.close()
                shap.summary_plot(shap_values,features=np.array(x),show=False)
                plt.savefig(f'{args.SAVE_DIR_PATH}/fold{fold}/minifold{mini_fold//3}/{coordinate}/shap-summary/summary.png')
                plt.close()


                y = model.predict(x_test)
                corr, pvalue = stats.spearmanr(y, y_test)
                pearcorr, pearpvalue = stats.pearsonr(y, y_test)
                print("Corrcoef", corr, pvalue, pearcorr, pearpvalue)
                print(np.max(y), np.min(y), np.mean(y), np.std(y))
                MAEloss = np.mean(np.abs(y-y_test))
                plot_samplepoints(y, y_test,f"{args.SAVE_DIR_PATH}/fold{fold}/minifold{mini_fold//3}/{coordinate}/random-forest{fold}.png",
                                            pearcorr, pearpvalue,
                                            corr, pvalue, 
                                            MAEloss)




                write_training_config_Csv(f"{args.SAVE_DIR_PATH}/fold{fold}/minifold{mini_fold//3}/{coordinate}/trainingConfig.csv",
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
                                            loss = MAEloss,
                                            Coordinate = coordinate,
                                            corrcoef = pearcorr,
                                            P_value = pearpvalue,
                                            spearman_p_value = pvalue,
                                            spearman_corrcoef = corr)
                

                # Highlighting local brain
                templateImage = get_templateImage()
                x,y,z = coordinate
                templateImage[x:x+side_length, y:y+side_length, z:z+side_length]=500
                
                x_length, y_length, z_length = templateImage.shape
                save_image = (torch.ones((x_length, y_length, z_length+20))).double()
                save_image[:,:,10:-10] = templateImage.double()
                Save_3DImage_to_nii(save_image, Path=f"{args.SAVE_DIR_PATH}/fold{fold}/minifold{mini_fold//3}/{coordinate}/localBrain-Highlight.nii",)


                torch.cuda.empty_cache() 

            # Remove dataset tensor to free GPU memory
            del train_set.all_BrainImages, valid_set.all_BrainImages, test_set.all_BrainImages
            del train_set.all_OneHotGenres, valid_set.all_OneHotGenres, test_set.all_OneHotGenres
            torch.cuda.empty_cache() 

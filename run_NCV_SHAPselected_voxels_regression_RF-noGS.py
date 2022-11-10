import os
print('GPUs:', os.environ.get("CUDA_VISIBLE_DEVICES"))
import argparse
import torch
import torchaudio
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


from statistics import mean
from datasets.LSA_brainVectors_dataset import LSA_brainVectors_dataset

from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset, random_split

from scipy import stats
from utils import generate_local_brain_dataset, scan_the_brain, Save_3DImage_to_nii, get_templateImage, plot_samplepoints, \
                    get_config_dict, write_training_config_Csv, get_brainImage

import shap
import pickle
import csv

# Seeds
torch.manual_seed(0)
# np.random.seed(0)

def get_parser():
    
    parser = argparse.ArgumentParser("RF", description='Hyperparameter setting.')
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
    parser.add_argument('--TARGET_SAVE_DIR_PATH', default=f'', type=str, help='')
    parser.add_argument('--LSA_DIR_PATH', default=f'', type=str, help='')
    parser.add_argument('--LOAD_DIR_PATH', default=f'', type=str, help='')
    parser.add_argument('--CB_DIR_PATH', default=f'Results/correlation-brains/NCV-RF-each-fold-correlation-brains', type=str, help='correlation-brains-path')
    parser.add_argument('--SB_DIR_PATH', default=f'Results/NCV-RF-SHAP-cubes', type=str, help='SHAP-brains-path')
    parser.add_argument('--TOP_K_SHAP_VALUE', default=80, type=int, help='')
    parser.add_argument('--GPU', default=f"0", type=str, help='')
    parser.add_argument('--TAU', default=1, type=float, help='')
    parser.add_argument('--SUBJECT', default='sub-000', type=str, help='')

    return parser


def prepare_local_brain_LSAdataset(device, genre_type, coordinate, side_length, trainFolds=None, testFolds=None, lsa_path=None, vecotrPointsPath=None):


    test_set = LSA_brainVectors_dataset(device=device, 
                                        Foldids = testFolds,
                                        LSAPath = lsa_path,
                                        coordinate = coordinate,
                                        side_length = side_length,
                                        vecotrPointsPath = vecotrPointsPath,
                                        genre_type=genre_type)    

    train_set = LSA_brainVectors_dataset(device=device, 
                                        Foldids = trainFolds, 
                                        genre_type=genre_type,
                                        coordinate = coordinate,
                                        side_length = side_length,
                                        vecotrPointsPath = vecotrPointsPath,
                                        LSAPath = lsa_path)

    
    return (train_set, test_set)

def find_target_voxels(args=None, fold:int=-1):
    # Load correlation brain
    CB_path = os.path.join(args.CB_DIR_PATH, f'fold{fold}', args.SUBJECT, f'pearson.nii')
    CB = get_brainImage(CB_path)[:,:,10:-10]

    # Get coordinates to check
    coordinates = scan_the_brain(CB)
    print(f'Cube coordinates to check: {coordinates}')

    # Find high correlation cubes' coordinates
    high_CB_coordinates = []
    for coordinate in coordinates:
        x, y, z = coordinate
        if CB[x,y,z] > args.TAU:
            print(f'High correlation cube: {coordinate}')
            high_CB_coordinates.append(coordinate)
    
    

    # Find topk SHAP value voxels' coordinates in high correlation cubes
    topk_SB_coordinates = []
    for coordinate in high_CB_coordinates:
        x,y,z = coordinate
        SB_path = os.path.join(args.SB_DIR_PATH, f'fold{fold}', args.SUBJECT,f'{coordinate}', f'shap_values.npy')
        # Load SHAP values convert to tensor
        SB_cube = np.load(SB_path)
        SB_cube = torch.from_numpy(SB_cube)
        

        _, indices = torch.topk(SB_cube.flatten(), args.TOP_K_SHAP_VALUE)
        indices = np.array(np.unravel_index(indices.numpy(), SB_cube.shape)).T
        topk_SB_coordinates += [(x+index[0], y+index[1], z+index[2]) for index in indices]
    print(f'Topk SHAP value voxels coordinates: {topk_SB_coordinates}')
    print(f'Total number of extracted voxels: {len(topk_SB_coordinates)}')

    # Save coordinates
    # TODO: save coordinates to a file

    return sorted(topk_SB_coordinates)






if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    
    

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    device = 'cpu'
    print(f"Using {device}")
        
    # Creating saving folder.
    # try:
    #     os.makedirs(f'{args.SAVE_DIR_PATH}')
    #     print(f'{args.SAVE_DIR_PATH} created.')
    # except OSError as error:
    #     print(f'{args.SAVE_DIR_PATH} existed.')




    four_folds = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]

    for fold in range(4):
        try:
            os.makedirs(f'{args.SAVE_DIR_PATH}/{args.SUBJECT}/fold{fold}')
            print(f'{args.SAVE_DIR_PATH}/{args.SUBJECT}/fold{fold} created.')
        except OSError as error:
            print(f'{args.SAVE_DIR_PATH}/{args.SUBJECT}/fold{fold} existed.')

        # Find target voxels' coordinates from high pearson correlation cubes 
        # and topk shap values.

        coordinates = find_target_voxels(args, fold)
        coordinates = np.array(coordinates)


        #print(f'Coordinates: {coordinates}',"x:",coordinates[:,0],"y:",coordinates[:,1],"z:",coordinates[:,2])
        
        # If no coordinates, then skip
        if len(coordinates) == 0:
            print("No voxel found in this fold")
            continue

        trainFolds = four_folds[(fold+1)%4] + four_folds[(fold+2)%4] + four_folds[(fold+3)%4]


        train_set,  test_set = prepare_local_brain_LSAdataset(device = device,
                                                        trainFolds = trainFolds, 
                                                        testFolds = four_folds[fold],
                                                        genre_type = args.GENRE_TYPE,
                                                        coordinate = (-1,-1,-1),
                                                        side_length = -1,
                                                        lsa_path = args.LSA_DIR_PATH)



        x, y = [], []
        for brainimage, label in train_set:
            x.append(brainimage.squeeze().cpu().numpy()[coordinates[:,0],
                                                        coordinates[:,1],
                                                        coordinates[:,2]])
            y.append(label.cpu().numpy().argmax(0))
        

        x_test, y_test = [], []

        for brainimage, label in test_set:
            x_test.append(brainimage.squeeze().cpu().numpy()[coordinates[:,0],
                                                             coordinates[:,1],
                                                             coordinates[:,2]])
            y_test.append(label.cpu().numpy().argmax(0))

        x = np.array(x)
        y = np.array(y)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        print(x.shape, y.shape, x_test.shape, y_test.shape)

        # Save data
        # np.save(f'{args.SAVE_DIR_PATH}/{args.SUBJECT}/fold{fold}/x.npy', x)
        # np.save(f'{args.SAVE_DIR_PATH}/{args.SUBJECT}/fold{fold}/y.npy', y)
        # np.save(f'{args.SAVE_DIR_PATH}/{args.SUBJECT}/fold{fold}/x_test.npy', x_test)
        # np.save(f'{args.SAVE_DIR_PATH}/{args.SUBJECT}/fold{fold}/y_test.npy', y_test)

        # Save coordinates
        # np.save(f'{args.SAVE_DIR_PATH}/{args.SUBJECT}/fold{fold}/coordinates.npy', coordinates)

        
        


        # model = RandomForestRegressor(n_jobs=-1,verbose=1)
        model = RandomForestRegressor(n_jobs=-1,verbose=1)

        model.fit(x, y)
        # explain the model's predictions using SHAP
        # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
        # explainer = shap.Explainer(model)
        # shap_values = explainer(np.array(x))
        # print(shap_values.shape)



        # visualize the first prediction's explanation and save plot
        # os.makedirs(f'{args.LOAD_DIR_PATH}/fold{fold}/shap-summary',exist_ok=True)
        # import matplotlib.pyplot as plt

        # shap.summary_plot(shap_values,features=np.array(x),show=False, plot_type="bar")
        # plt.savefig(f'{args.LOAD_DIR_PATH}/fold{fold}/shap-summary/summary-bar.png')
        # plt.close()
        # shap.summary_plot(shap_values,features=np.array(x),show=False)
        # plt.savefig(f'{args.LOAD_DIR_PATH}/fold{fold}/shap-summary/summary.png')
        # plt.close()

        

        y = model.predict(x_test)
        corr, pvalue = stats.spearmanr(y, y_test)
        pearcorr, pearpvalue = stats.pearsonr(y, y_test)
        print("Corrcoef", corr, pvalue, pearcorr, pearpvalue)
        print(np.max(y), np.min(y), np.mean(y), np.std(y))
        MAEloss = np.mean(np.abs(y-y_test))
        plot_samplepoints(y, y_test,f"{args.SAVE_DIR_PATH}/{args.SUBJECT}/fold{fold}/RF{fold}.png",
                                    pearcorr, pearpvalue,
                                    corr, pvalue, 
                                    MAEloss)
        

        # save correlation, pvalue, loss result in csv file
        write_training_config_Csv(f"{args.SAVE_DIR_PATH}/{args.SUBJECT}/fold{fold}/RF{fold}.csv",
                                    fold = fold,
                                    spearmancorr = corr,
                                    spearmanpvalue = pvalue,
                                    pearcorr = pearcorr,
                                    pearpvalue = pearpvalue,
                                    MAEloss = MAEloss)

import os
print('GPUs:', os.environ.get("CUDA_VISIBLE_DEVICES"))
import argparse
import torch
import numpy as np
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV



from statistics import mean
from datasets.LSA_brainVectors_dataset import LSA_brainVectors_dataset

from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset, random_split

from scipy import stats
from utils import generate_local_brain_dataset, scan_the_brain, Save_3DImage_to_nii, get_templateImage, plot_samplepoints,\
                    write_training_config_Csv
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
    parser.add_argument('--LOCAL_BRAIN_SIDE_LENGTH', default=10, type=int, help='')
    parser.add_argument('--LOCAL_BRAIN_HOP_LENGTH', default=10, type=int, help='')
    parser.add_argument('--SAVE_DIR_PATH', default=f'Results/{time.ctime().replace(" ","-")}', type=str, help='')
    parser.add_argument('--LSA_DIR_PATH', default=f'', type=str, help='')
    parser.add_argument('--GPU', default=f"0", type=str, help='')
    parser.add_argument('--SUBJECT', default=f"sub-000", type=str, help='')
    parser.add_argument('--TAU', default=-1, type=float, help='')

    return parser



def prepare_local_brain_LSAdataset(device, genre_type, coordinate, side_length, trainFolds=None, lsa_path=None, vecotrPointsPath=None):

    if trainFolds:
        trainFolds = trainFolds
        AllFolds = range(1,17)
        testFolds = list(set(AllFolds) - set(trainFolds))
        print('Test Fold:', testFolds)
    else:
        trainFolds = range(1, 17)
        AllFolds = range(1,17)
        testFolds = list(set(AllFolds) - set(trainFolds))


    test_set = LSA_brainVectors_dataset(device=device, 
                                        Foldids = testFolds,
                                        LSAPath = lsa_path,
                                        BrainMask = "masks/ACmask.nii",
                                        genre_type=genre_type)

    trainFolds, validFolds = random_split(trainFolds, (10,2))
    

    train_set = LSA_brainVectors_dataset(device=device, 
                                        Foldids = trainFolds, 
                                        BrainMask = "masks/ACmask.nii",
                                        genre_type=genre_type,
                                        LSAPath = lsa_path)
    valid_set = LSA_brainVectors_dataset(device=device, 
                                        Foldids = validFolds, 
                                        BrainMask = "masks/ACmask.nii",
                                        genre_type=genre_type, 
                                        LSAPath = lsa_path)
    
    return (train_set, valid_set, test_set)




if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    
    

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
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
        try:
            os.makedirs(f'{args.SAVE_DIR_PATH}/fold{fold}')
            print(f'{args.SAVE_DIR_PATH}/fold{fold} created.')
        except OSError as error:
            print(f'{args.SAVE_DIR_PATH}/fold{fold} existed.')

        
        trainFolds = four_folds[(fold+1)%4] + four_folds[(fold+2)%4] + four_folds[(fold+3)%4]
    
        # Dataset part




        train_set, valid_set, test_set = prepare_local_brain_LSAdataset(device = device,
                                                                        trainFolds = trainFolds, 
                                                                        genre_type = args.GENRE_TYPE,
                                                                        coordinate = (-1,-1,-1),
                                                                        side_length = -1,
                                                                        lsa_path = args.LSA_DIR_PATH)

                

        x, y = [], []
        for brainimage, label in train_set:
            #x.append(np.log(brainimage.squeeze().cpu().numpy()+args.EPS))
            x.append((brainimage.squeeze().cpu().numpy()))
            y.append(label.cpu().numpy().argmax(0))

        for brainimage, label in valid_set:
            #x.append(np.log(brainimage.squeeze().cpu().numpy()+args.EPS))
            x.append((brainimage.squeeze().cpu().numpy()))
            y.append(label.cpu().numpy().argmax(0))


        import matplotlib.pyplot as plt
        # plt.hist(x)
        # plt.savefig(f'hist{fold}.png')
        # plt.close()




        x_test, y_test = [], []

        for brainimage, label in test_set:
            #x_test.append(np.log(brainimage.squeeze().cpu().numpy()+args.EPS))
            x_test.append((brainimage.squeeze().cpu().numpy()))
            y_test.append(label.cpu().numpy().argmax(0))



        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        # Convert to numpy array
        x = np.array(x)
        y = np.array(y)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        print(x.shape, y.shape, x_test.shape, y_test.shape)



        model = RandomForestRegressor()

        param_grid = {
            'bootstrap': [True],
            'max_depth': [80, 90, 100, 110],
            'max_features': [2, 3],
            'min_samples_leaf': [3, 4, 5],
            'min_samples_split': [8, 10, 12],
            'n_estimators': [100, 200, 300, 1000]
        }
        grid_search = GridSearchCV(estimator = model, param_grid = param_grid, 
                          cv = 4, n_jobs = -1, verbose = 2)
        grid_search.fit(x,y)
        # Save best parameters to file
        with open(f'{args.SAVE_DIR_PATH}/fold{fold}/best_params.txt', 'w') as f:
            f.write(str(grid_search.best_params_))
            print(f'Saved to {args.SAVE_DIR_PATH}/fold{fold}/best_params.txt', 
                    grid_search.best_params_)
        model = grid_search.best_estimator_

        # explain the model's predictions using SHAP
        # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
        explainer = shap.Explainer(model)
        shap_values = explainer(np.array(x))
        print(shap_values.shape)

        # visualize the first prediction's explanation and save plot
        shap.summary_plot(shap_values,features=np.array(x),show=False, plot_type="bar")
        plt.savefig(f'shap-summary/summary{fold}-bar.png')
        plt.close()
        shap.summary_plot(shap_values,features=np.array(x),show=False)
        plt.savefig(f'shap-summary/summary{fold}.png')
        plt.close()


        y = model.predict(x_test)
        corr, pvalue = stats.spearmanr(y, y_test)
        pearcorr, pearpvalue = stats.pearsonr(y, y_test)
        print("Corrcoef", corr, pvalue, pearcorr, pearpvalue)
        print(np.max(y), np.min(y), np.mean(y), np.std(y))
        MAEloss = np.mean(np.abs(y-y_test))

        plot_samplepoints(y, y_test,f"{args.SAVE_DIR_PATH}/fold{fold}/random-forest{fold}.png",
                                    pearcorr, 
                                    pearpvalue,
                                    corr, 
                                    pvalue, 
                                    MAEloss)

        # save correlation, pvalue, loss result in csv file
        write_training_config_Csv(f"{args.SAVE_DIR_PATH}/fold{fold}/RF{fold}.csv",
                                    fold = fold,
                                    spearmancorr = corr,
                                    spearmanpvalue = pvalue,
                                    pearcorr = pearcorr,
                                    pearpvalue = pearpvalue,
                                    MAEloss = MAEloss)

        
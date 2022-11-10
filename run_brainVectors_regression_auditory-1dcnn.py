import os
print('GPUs:', os.environ.get("CUDA_VISIBLE_DEVICES"))
import argparse
import torch
import torchaudio
import numpy as np
import time
from sklearn.metrics import f1_score

from LSARegressionTrainer import model_trainer

from statistics import mean
from datasets.LSA_brainVectors_dataset import LSA_brainVectors_dataset

from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset, random_split

from scipy import stats
from utils import generate_local_brain_dataset, scan_the_brain, Save_3DImage_to_nii, get_templateImage, plot_samplepoints,\
write_training_config_Csv



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
    parser.add_argument('--LOCAL_BRAIN_SIDE_LENGTH', default=10, type=int, help='')
    parser.add_argument('--LOCAL_BRAIN_HOP_LENGTH', default=10, type=int, help='')
    parser.add_argument('--SAVE_DIR_PATH', default=f'Results/{time.ctime().replace(" ","-")}', type=str, help='')
    parser.add_argument('--LSA_DIR_PATH', default=f'', type=str, help='')
    parser.add_argument('--GPU', default=f"0", type=str, help='')

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
        
    print(f"Using {device}")
        
    # Creating saving folder.
    try:
        os.makedirs(f'{args.SAVE_DIR_PATH}')
        print(f'{args.SAVE_DIR_PATH} created.')
    except OSError as error:
        print(f'{args.SAVE_DIR_PATH} existed.')

    # Model part
    from models.model_for_brainVectors_regression_1dCNN import AAEEncoder
    

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

                

        # Training part
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
                                SAVE_DIR_PATH = f'{args.SAVE_DIR_PATH}/fold{fold}')
                                
        trainer.start_training()
        trainer.write_training_config_Csv(test_loss = trainer.test_loss,
                                            corrcoef = trainer.corrcoef,
                                            P_value = trainer.p_value,
                                            side_length = args.LOCAL_BRAIN_SIDE_LENGTH,
                                            hop_length = args.LOCAL_BRAIN_HOP_LENGTH,
                                            lsa_dir_path = args.LSA_DIR_PATH)



        # test part
        model = AAEEncoder(num_classes=1).to(device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(f"{trainer.SAVE_DIR_PATH}/{trainer.MODEL_NAME}"))
        y_test = []
        y = []

        data_loader = trainer.create_data_loader(test_set, batch_size=128)
        #data_loader = trainer.create_data_loader(train_set, batch_size=1)
        
        for _ in range(100):
            for v,t in data_loader:
                reg = model(torch.tensor(v).float().to(device))

                y.append(reg.squeeze().detach().cpu().numpy()*25)
                #y.append(cla.squeeze().argmax().detach().cpu().numpy())
                y_test.append(t.argmax(axis=1).squeeze().cpu().numpy())

        y = np.concatenate(y)
        y_test = np.concatenate(y_test)
        print(y.shape, y_test.shape)
        corr, pvalue = stats.spearmanr(y, y_test)
        pearcorr, pearpvalue = stats.pearsonr(y, y_test)
        print("Corrcoef", corr, pvalue, pearcorr, pearpvalue)
        print(np.max(y), np.min(y), np.mean(y), np.std(y))
        MAEloss = np.mean(np.abs(y-y_test))
        plot_samplepoints(y, y_test,f"{args.SAVE_DIR_PATH}/fold{fold}/CNN{fold}.png",
                                    pearcorr, 
                                    pearpvalue,
                                    corr, 
                                    pvalue, 
                                    MAEloss)

        # save correlation, pvalue, loss result in csv file
        write_training_config_Csv(f"{args.SAVE_DIR_PATH}/fold{fold}/CNN{fold}.csv",
                                    fold = fold,
                                    spearmancorr = corr,
                                    spearmanpvalue = pvalue,
                                    pearcorr = pearcorr,
                                    pearpvalue = pearpvalue,
                                    MAEloss = MAEloss)

import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import pandas as pd
import seaborn as sns

import torch
import torchaudio
import torch.nn.functional as F
from torch import nn, randn, mean, log
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset


from tqdm import tqdm
from statistics import mean
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix


from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable as Var
from scipy import stats
from utils import batch_image_masking_3D

#from medcam import medcam



# Seeds
torch.manual_seed(0)
np.random.seed(0)


class model_trainer:
    def __init__(self,
                TRAIN_SET = None,
                VALID_SET = None,
                TEST_SET = None,
                AAEEncoder = None,
                BATCH_SIZE = 4,
                EPOCHS = 20,
                LEARNING_RATE = 0.001,
                L1_WEIGHT = 0.0001,
                L2_WEIGHT = 0.0001,
                EPS = 1e-15,
                MASK = False,
                NUM_MASK = 30,
                MASK_SIZE = 10,
                DATE = '',
                FOLDID = [], # Which folds we are training when perform cross validation
                SUFFIX = "",
                MODEL_NAME = f'',
                GENRE_TYPE = "",
                COORDINATE = (-1,-1,-1),
                SAVE_DIR_PATH = None,
                LOAD_DIR_PATH = None,
                DEVICE = 'cuda'):

        self.TRAIN_SET = TRAIN_SET
        self.VALID_SET = VALID_SET
        self.TEST_SET = TEST_SET
        self.AAEEncoder = AAEEncoder
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARNING_RATE = LEARNING_RATE
        self.L1_WEIGHT = L1_WEIGHT
        self.L2_WEIGHT = L2_WEIGHT
        self.EPS = EPS
        self.MASK = MASK
        self.NUM_MASK = NUM_MASK
        self.MASK_SIZE = MASK_SIZE
        self.DATE = DATE
        self.FOLDID = FOLDID
        self.SUFFIX = SUFFIX
        self.GENRE_TYPE = GENRE_TYPE
        self.MODEL_NAME = MODEL_NAME
        self.SAVE_DIR_PATH = SAVE_DIR_PATH
        self.LOAD_DIR_PATH = LOAD_DIR_PATH
        self.COORDINATE = COORDINATE
        self.device = DEVICE
        

        self.all_pred_log = []
        self.all_truth_log = []

        self.test_loss = 0
        self.corrcoef = 0
        self.p_value = 0

        


    def create_data_loader(self, train_data, batch_size):
        # If train_data is None, return None
        if train_data is None:
            return None
        # To generate probability for each label
        def balance_prob(imbalanced_data):
            genres = []
            for _, genre in imbalanced_data:
                genres.append(genre.cpu().item())
            genreCounter = Counter(genres)
            print(genreCounter)
            balance_prob = []
            
            for _, genre in imbalanced_data:
                balance_prob.append(1/genreCounter[genre.cpu().item()])
            return balance_prob


        imbalanced_data = train_data
        print("Generate balanced dataset:")
        
        balance_prob = balance_prob(imbalanced_data)
        print(Counter(balance_prob))

        from torch.utils.data.sampler import WeightedRandomSampler
        sampler = WeightedRandomSampler(weights=balance_prob, num_samples=len(imbalanced_data), replacement=True)

        train_dataloader = DataLoader(train_data, batch_size = batch_size, sampler = sampler)
        print('------------------------------')
        return train_dataloader




    def train_single_epoch(self, model, data_loader, data_set = None, loss_fn = None, optimiser = None, device = 'cpu', mode = 'train', scheduler = []):
        
        loop = tqdm(data_loader, leave = False)

        epoch_loss = []

        
        for (BrainImage, InstrumentGenre) in loop:
        
            

            # If in training mode, we will pick a excerpt of all frames
            if mode == 'train':
                if self.MASK:
                    BrainImage = batch_image_masking_3D(BrainImage.to(device), num_mask=self.NUM_MASK, mask_size=self.MASK_SIZE)
            elif mode == 'valid':
                pass
            

            # make sure they are in cuda
            BrainImage, InstrumentGenre = BrainImage.float().to(device), InstrumentGenre.float().to(device)

            # Encoder
            Prediction = model['Encoder'](BrainImage)

            
            
            # Origianl loss
            #loss = loss_fn(Prediction.float().squeeze(), InstrumentGenre.argmax(1).float()/24)
            #losswithlog = loss_fn((1e-9 + Prediction.float().squeeze()).log(), (1e-9 + InstrumentGenre.argmax(1).float()/24).log())
            l1loss = torch.nn.L1Loss()(Prediction.float().squeeze(), InstrumentGenre.float()/24)



            # Loss with l1, l2 regularization
            try:
                l1_penalty_en = self.L1_WEIGHT * sum([p.abs().sum() for p in model['Encoder'].parameters()])
                l2_penalty_en = self.L2_WEIGHT * sum([(p**2).sum() for p in model['Encoder'].parameters()])
                #loss_with_penalty = loss + l1_penalty_en + l2_penalty_en
                loss_with_penalty = l1loss + l1_penalty_en + l2_penalty_en
            except:
                loss_with_penalty = l1loss 
                print(f"Model parameters don't exist!")
            
            
            

            
            # Backpropagate error and update weights of encoder and decoder
            if mode == 'train':
                

                # Clear gradient from previous batch
                optimiser['Encoder'].zero_grad() 
                loss_with_penalty.backward()

                # Update weights
                optimiser['Encoder'].step()

                # Show the loss(Original)
                loop.set_postfix(loss = loss_with_penalty.item())
            else:
                loop.set_postfix(val_loss = loss_with_penalty.item())

            # Decay the learning rate if loss didn't drop
            #if mode == 'valid' and len(scheduler) > 0:
            #    scheduler['Encoder'].step(loss_with_penalty)

            #print(Prediction.argmax(1), InstrumentGenre.argmax(1))
            epoch_loss.append(loss_with_penalty.item())
            
            

        
        # Record training loss and validation loss
        if mode == 'train':
            epoch_train_loss = mean(epoch_loss)
            
            print(f"loss: {epoch_train_loss}")
            return epoch_loss

        else:
            epoch_valid_loss = mean(epoch_loss)
            
            print(f"val_loss: {epoch_valid_loss}")
            return epoch_loss

        



    def train(self, model, train_data_loader, valid_data_loader = None, data_set = None, loss_fn = None, optimiser = None, device = 'cpu', epochs = 100, scheduler = []):
        loss ,val_loss = [], []
        mean_epoch_loss, mean_epoch_val_loss = float('inf'), float('inf') # loss的最大值
        min_valid_loss, min_train_loss = float('inf'), float('inf')
        
        
        for i in range(epochs):
            print(f"Epoch {i+1}")

            # training
            for m in model:
                model[m].train()
            epoch_loss = self.train_single_epoch(model = model,
                                                data_loader = train_data_loader, 
                                                data_set = data_set, 
                                                loss_fn = loss_fn, 
                                                optimiser = optimiser, 
                                                device = device,
                                                scheduler = scheduler)
            # validation
            for m in model:
                model[m].eval()
            with torch.no_grad():   
                epoch_valid_loss = self.train_single_epoch( model = model,
                                                            data_loader = valid_data_loader,
                                                            loss_fn = loss_fn, 
                                                            optimiser = optimiser, 
                                                            device = device, 
                                                            scheduler = scheduler,
                                                            mode = 'valid')



            # Save the min epoch loss of self.epochs
            mean_epoch_loss, mean_epoch_val_loss = mean(epoch_loss), mean(epoch_valid_loss)

            # Decay the learning rate if loss didn't drop
            if len(scheduler) > 0:
                scheduler['Encoder'].step(mean_epoch_val_loss)
            
            loss+=([mean_epoch_loss])
            val_loss+=([mean_epoch_val_loss])

            
            if min_valid_loss > mean_epoch_val_loss:
                min_valid_loss = mean_epoch_val_loss
                

                torch.save(model['Encoder'].state_dict(), f"{self.SAVE_DIR_PATH}/{self.MODEL_NAME}.pth")
                print('Saved best model')

            if min_train_loss > mean_epoch_loss:
                min_train_loss = mean_epoch_loss

            print("---------------------------")
            
            
        print("Finished training")
        result = {}
        result['loss'] = loss
        result['val_loss'] = val_loss
        result['min_epoch_loss'] = min_train_loss
        result['min_epoch_val_loss'] = min_valid_loss

        
        return result
            

    def plot_history(self, loss = [], min_epoch_loss = None, val_loss = [], min_epoch_val_loss = None, Name = None):

        fig, axs = plt.subplots(1)
        
        # create error sublpot
        axs.plot(loss, label="train loss")
        axs.plot(val_loss, label="val loss")   
        axs.plot([min_epoch_loss]*len(loss), color = 'k', linestyle ='--', label=f"min train epoch loss: {min_epoch_loss}")
        axs.plot([min_epoch_val_loss]*len(loss), color = 'k', linestyle ='--', label=f"min val epoch loss: {min_epoch_val_loss}")    
        
        axs.set_ylabel("loss")
        axs.set_xlabel("epoch")
        axs.legend(loc="upper right")
        axs.set_title("loss eval")



        plt.tight_layout()
        if Name:
            try:
                plt.savefig(f'{self.SAVE_DIR_PATH}/{Name}.png')
                print(f'Saved {Name}')
            except:
                print(f'Can\'t saved {Name}')

        else:
            try:
                plt.savefig(f'{self.SAVE_DIR_PATH}/analysis_{self.EPOCHS}e_valLoss={min_loss}.png')
            except:
                print(f'Can\'t saved analysis_{self.EPOCHS}e_valLoss={min_loss}.png')

        plt.close()


    def write_training_config_Csv(self, **additional_arg):
        import csv
        with open(f'{self.SAVE_DIR_PATH}/trainingConfig.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Arguments', 'Values'])
            writer.writerow(['Date', self.DATE])
            writer.writerow(['Genre type', self.GENRE_TYPE])
            writer.writerow(['TrainFolds', self.FOLDID])
            writer.writerow(['Epochs', self.EPOCHS])
            writer.writerow(['Batch size', self.BATCH_SIZE])
            writer.writerow(['Learning rate', [(self.LEARNING_RATE)]])
            writer.writerow(['L1 weight', [(self.L1_WEIGHT)]])
            writer.writerow(['L2 weight', [(self.L2_WEIGHT)]])
            writer.writerow(['Mask', self.MASK])
            writer.writerow(['Number of mask', self.NUM_MASK])
            writer.writerow(['Mask size', self.MASK_SIZE])
            writer.writerow(['Image masking', self.MASK])
            writer.writerow(['Note', self.SUFFIX])
            writer.writerow(['Device', self.device])
            writer.writerow(['Coordinate', self.COORDINATE])

            for key, val in additional_arg.items():
                writer.writerow([key, val])
    
    def testing(self, model, data_loader, loss_fn = None, device = 'cpu'):
        
        #with torch.no_grad():   
        
        loop = tqdm(data_loader, leave = False)

        epoch_loss = []
        epoch_coefficient = []

        
        for (BrainImage, InstrumentGenre) in loop:
        
            
            # make sure they are in cuda
            BrainImage, InstrumentGenre = BrainImage.float().to(device), InstrumentGenre.float().to(device)

            # Encoder
            Prediction = model['Encoder'](BrainImage)

            
            
            # Origianl loss
            loss = loss_fn(Prediction.float().squeeze(), InstrumentGenre.float()/24)

            self.all_pred_log += Prediction.float().squeeze().tolist()
            self.all_truth_log += (InstrumentGenre.float()/24).tolist()


            # Loss with l1, l2 regularization
            try:
                l1_penalty_en = self.L1_WEIGHT * sum([p.abs().sum() for p in model['Encoder'].parameters()])
                l2_penalty_en = self.L2_WEIGHT * sum([(p**2).sum() for p in model['Encoder'].parameters()])
                loss_with_penalty = loss + l1_penalty_en + l2_penalty_en
            except:
                loss_with_penalty = loss 
                print(f"Model parameters don't exist!")
        

            loop.set_postfix(test_loss = loss_with_penalty.item())

        

            epoch_loss.append(loss_with_penalty.item())
                                

        epoch_test_loss = mean(epoch_loss)
        
        return epoch_test_loss




    def start_training(self):

        # Construct model and assign it to device
        device = self.device
        encoder = self.AAEEncoder(num_classes=1).to(device)
        encoder = torch.nn.DataParallel(encoder)


        try:
            encoder.load_state_dict(torch.load(f"{self.LOAD_DIR_PATH}/{self.MODEL_NAME}.pth"))
            print('Countinue training!')
        except:
            print(f"Previous models doesn't exist!")


        
        models = {}
        models['Encoder'] = encoder
        


        # Initialise loss funtion
        loss_fn = nn.MSELoss()


        # Initialise optimiser
        optimiser_en = torch.optim.Adam(encoder.parameters(), lr = self.LEARNING_RATE)
        
        optimisers = {}
        optimisers['Encoder'] = optimiser_en
        

        # Initialise scheduler
        # Decay learning rate if loss didn't change within 5 epochs
        scheduler_en = ReduceLROnPlateau(optimiser_en, factor = 0.8, verbose = True, patience=5) 
        
        schedulers = {}
        schedulers['Encoder'] = scheduler_en
        
        train_loader = self.create_data_loader(self.TRAIN_SET, self.BATCH_SIZE)
        valid_loader = self.create_data_loader(self.VALID_SET, self.BATCH_SIZE)
        test_loader = self.create_data_loader(self.TEST_SET, self.BATCH_SIZE)

        
        
        # Train model
        result = self.train(model = models, 
                            train_data_loader = train_loader, 
                            valid_data_loader = valid_loader, 
                            loss_fn = loss_fn, 
                            optimiser = optimisers, 
                            device = device, 
                            epochs = self.EPOCHS, 
                            scheduler = schedulers)
        
        # Plot training curve
        self.plot_history(loss = result['loss'], 
                          min_epoch_loss = result['min_epoch_loss'], 
                          val_loss = result['val_loss'], 
                          min_epoch_val_loss = result['min_epoch_val_loss'],
                          Name = f'{self.MODEL_NAME}')
        
        try:
            os.remove("temp_output.pt")
        except:
            print("No temp files to remove.")

        
        # Testing
        # load best model

        # encoder.load_state_dict(torch.load(f"{self.SAVE_DIR_PATH}/{self.MODEL_NAME}/{self.MODEL_NAME}.pth"))
        # models['Encoder'] = encoder

        # print("Testing!")

        

        # for m in models:
        #     models[m].eval()
        # models['Encoder'] = medcam.inject(models['Encoder'], output_dir='attention_maps', save_maps=True)

        # total_all_pred_log = []
        # total_all_truth_log = []

        # test_loop = tqdm(range(100), leave=False)
        # for _ in test_loop:

        #     self.all_pred_log = []
        #     self.all_truth_log = []

        #     testloss = self.testing(model = models,
        #                             data_loader = test_loader,
        #                             loss_fn = loss_fn, 
        #                             device = device)

        #     coef, p_value = stats.pearsonr(self.all_pred_log, self.all_truth_log)

        #     self.test_loss += testloss
        #     self.corrcoef += coef
        #     self.p_value += p_value

        #     total_all_pred_log += self.all_pred_log
        #     total_all_truth_log += self.all_truth_log

        #     test_loop.set_postfix(test_loss = testloss, coef = coef, p_value = p_value)
        
        # self.test_loss /= 100
        # self.corrcoef /= 100
        # self.p_value /= 100
        # self.all_pred_log = total_all_pred_log
        # self.all_truth_log = total_all_truth_log
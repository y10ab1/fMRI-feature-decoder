import os
import numpy as np
import torch
from utils import get_brainImage, get_templateImage, Save_3DImage_to_nii, get_config_dict
# get each subject's fold shapbrain frompath# and merge them into one brainImage
def main(path):
    # Save path
    path_name = path.split('-')
    save_path = "-".join([path_name[0],path_name[1],'SHAP',path_name[2]])
    # get all subjects
    

    subjects = os.listdir(path)
    for subject in subjects:
        # get all folds
        folds = os.listdir(path+'/' + subject)
        if not folds:
            continue

        for fold in folds:
            minifolds = os.listdir(path+'/' + subject + '/' + fold)
            

            for minifold in minifolds:
                shap_brainImage = torch.zeros(get_brainImage().shape)

                # get all brainImages
                coordinates = os.listdir(path+'/' + subject + '/' + fold + '/' + minifold)
                # merge all brainImages
                for coordinate in coordinates:
                    # get x,y,z coordinate with type int
                    x,y,z = coordinate[1:-1].split(',')
                    x,y,z = int(x), int(y), int(z)
                    print(coordinate,x,y,z)
                    side_length = 20


                    # load shap values from numpy file
                    shap_values = np.load(path+'/' + subject + '/' + fold + '/' + minifold + '/'+ coordinate + '/shap_values.npy')
                    print(shap_values.shape)
                    # print all zero check warning
                    if np.sum(shap_values) == 0:
                        print('all zero')
                    shap_brainImage[x:x+side_length, y:y+side_length, z+10:z+10+side_length] += torch.tensor(shap_values)
                    # shap_brainImage[x:x+side_length, y:y+side_length, z:z+side_length] += torch.tensor(shap_values)

                # save merged brainImages
                # os.makedirs(f'Results/NCV-RF-SHAP-cubes/{fold}/' + subject, exist_ok=True)
                # Save_3DImage_to_nii(shap_brainImage/4, f'Results/NCV-RF-SHAP-cubes/{fold}/' + subject + '/shap_brainImage.nii')
                #os.makedirs(f'{save_path}/{fold}/' + subject, exist_ok=True)
                Save_3DImage_to_nii(shap_brainImage, f'{save_path}/{fold}/' + subject + f'/{minifold}-shap_brainImage.nii') 
                #os.rename(f'{save_path}/{fold}/' + subject + f'/mini{minifold}-shap_brainImage.nii',f'{save_path}/{fold}/' + subject + f'/{minifold}-shap_brainImage.nii')     

if __name__ == '__main__':
    for path in ['Results/NCV-3DCNN-cubes', 'Results/NCV-RF-cubes']:
        main(path)
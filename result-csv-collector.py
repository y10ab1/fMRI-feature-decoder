import csv
import os
import utils

# Collect each result file from the results directory and write it to a CSV file
# RF_RF/ RF_1dcnn/ 3dcnn_RF/ 3dcnn_1dcnn/

# path = 'Results/train-3dcnn-highcorr-cubes/RF/SHAP4000'
# path = 'Results/train-3dcnn-highcorr-cubes/RF/SHAP8000'
# path = 'Results/train-3dcnn-highcorr-cubes/1dcnn/SHAP4000'
# path = 'Results/train-RF-highcorr-cubes/RF/SHAP80'
# path = 'Results/train-RF-highcorr-cubes/1dcnn/SHAP80'
# path = 'Results/train-ACmask/1dcnn'
tau = '0.08-8'
path_list = [f'Results/NCV-RF-RF-3/{tau}/SHAP80',
             f'Results/NCV-RF-RF-3/{tau}/SHAP800',
             f'Results/NCV-RF-RF-3/{tau}/SHAP4000',
            #  f'Results/NCV-RF-CNN-2/{tau}/SHAP80',
            #  f'Results/NCV-RF-CNN-2/{tau}/SHAP800',
            #  f'Results/NCV-RF-CNN-2/{tau}/SHAP4000',
             
            #  f'Results/NCV-CNN-CNN-2/{tau}/SHAP80',
            #  f'Results/NCV-CNN-CNN-2/{tau}/SHAP800',
            #  f'Results/NCV-CNN-CNN-2/{tau}/SHAP4000'
             f'Results/NCV-CNN-RF-3/{tau}/SHAP80',
             f'Results/NCV-CNN-RF-3/{tau}/SHAP800',
             f'Results/NCV-CNN-RF-3/{tau}/SHAP4000',]

# path_list = [f'Results/NCV-AC-RF']


             



statistics_fileName = "statistics.csv"
def main(path:str):
    # Create the CSV file
    save_file_name = path.split('/')[1] + '-' + path.split('/')[3] + '.csv'
    save_dir = f'Results/statistics-3/{tau}'

    # save_file_name = 'AC_RF_gridsearch_new.csv'
    # save_dir = f'Results/statistics-3'

    # save_file_name = 'AC-CNN.csv'
    # save_dir = f'Results/statistics'

    os.makedirs(save_dir, exist_ok=True)
    with open(save_dir+"/"+save_file_name, "w") as save_f:
        writer = csv.writer(save_f)
        writer.writerow(["filePath", "subject", "fold", "pearson-corr", "perason-pvalue", "spearman-corr", "spearman-pvalue"])

        # Iterate over each result file
        for subject in os.listdir(path):
            # Read the result file
            if os.path.isdir(path+"/"+subject):
                for fold in os.listdir(path + '/' + subject):
                    for filename in os.listdir(path + '/' + subject + '/' + fold):
                        if ("CNN" in filename or "random" in filename or "RF" in filename) and filename.endswith(".csv"):
                            d = utils.get_config_dict("/".join([path, subject, fold, filename]))
                            # Write the result to the CSV file
                            pearcorr = d["pearcorr"]
                            pearpval = d["pearpvalue"]
                            spearmancorr = d["spearmancorr"]
                            spearmanpvalue = d["spearmanpvalue"]
                            foldid = fold[-1]
                            filePath = "/".join([path, subject, fold, filename])
                            writer.writerow([filePath, subject, foldid, pearcorr, pearpval, spearmancorr, spearmanpvalue])
                            print(filePath, subject, foldid, pearcorr, pearpval, spearmancorr, spearmanpvalue)




if __name__ == "__main__":
    for path in path_list:
    # for path in ['Results/NCV-AC-CNN']:
        main(path)

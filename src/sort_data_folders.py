import os

# load data
from src import _DATA_PATH
import shutil

# create new folders

path = "/u/data/s185231/ADLCV/ADLCV_project/data"

N_folder = os.path.join(path, "train", "N/")
P_folder = os.path.join(path, "train", "P/")
zero_folder = os.path.join(path, "train", "0/")

os.makedirs(N_folder, exist_ok=True)
os.makedirs(P_folder, exist_ok=True)
os.makedirs(zero_folder, exist_ok=True)

train_folder_content = os.listdir(os.path.join(path, "training", "INPUT_IMAGES"))
test_folder_content = os.listdir(os.path.join(path, "testing", "INPUT_IMAGES"))
val_folder_content = os.listdir(os.path.join(path, "validation", "INPUT_IMAGES"))

for file in train_folder_content:
    full_path = os.path.join(path, "training", "INPUT_IMAGES", file)
    filename = file.split("\\")[-1]
    if file.split('_')[-1][0] == "N":
        destination =  N_folder + filename
        # copy only files
        if os.path.isfile(full_path):
            shutil.copy(full_path, destination)

    elif file.split('_')[-1][0] == '0':
        destination =  zero_folder + filename
        # copy only files
        if os.path.isfile(full_path):
            shutil.copy(full_path, destination)

    elif file.split('_')[-1][0] == 'P':
        destination =  P_folder + filename

        # copy only files
        if os.path.isfile(full_path):
            shutil.copy(full_path, destination)
        

for file in val_folder_content:
    full_path = os.path.join(path, "validation", "INPUT_IMAGES", file)
    filename = file.split("\\")[-1]
    if file.split('_')[-1][0] == "N":
        destination =  N_folder + filename
        # copy only files
        if os.path.isfile(full_path):
            shutil.copy(full_path, destination)

    elif file.split('_')[-1][0] == '0':
        destination =  zero_folder + filename
        # copy only files
        if os.path.isfile(full_path):
            shutil.copy(full_path, destination)

    elif file.split('_')[-1][0] == 'P':
        destination =  P_folder + filename

        # copy only files
        if os.path.isfile(full_path):
            shutil.copy(full_path, destination)

for file in test_folder_content:
    full_path = os.path.join(path, "testing", "INPUT_IMAGES", file)
    filename = file.split("\\")[-1]
    if file.split('_')[-1][0] == "N":
        destination =  N_folder + filename
        # copy only files
        if os.path.isfile(full_path):
            shutil.copy(full_path, destination)

    elif file.split('_')[-1][0] == '0':
        destination =  zero_folder + filename
        # copy only files
        if os.path.isfile(full_path):
            shutil.copy(full_path, destination)

    elif file.split('_')[-1][0] == 'P':
        destination =  P_folder + filename

        # copy only files
        if os.path.isfile(full_path):
            shutil.copy(full_path, destination)
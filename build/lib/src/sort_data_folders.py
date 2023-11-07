import os

# load data
from src import _DATA_PATH
import shutil

# create new folders
def transfer_files():
    path_1 = _DATA_PATH
    path_2 = _DATA_PATH

    stage = ['training', 'validation', 'testing']
    folders = ['N/', 'P/', '0/']
    for s in stage:
        for f in folders:
                folder = os.path.join(path_2, s, f)
                os.makedirs(folder, exist_ok=True)

    train_folder_content = os.listdir(os.path.join(path_1, "training", "INPUT_IMAGES"))
    test_folder_content = os.listdir(os.path.join(path_1, "testing", "INPUT_IMAGES"))
    val_folder_content = os.listdir(os.path.join(path_1, "validation", "INPUT_IMAGES"))

    for file in train_folder_content:
        full_path = os.path.join(path_1, "training", "INPUT_IMAGES", file)
        filename = file.split("\\")[-1]
        if file.split('_')[-1][0] == "N":
            destination =  os.path.join(path_2,"training","N", filename)
            # copy only files
            if os.path.isfile(full_path):
                shutil.copy(full_path, destination)

        elif file.split('_')[-1][0] == "P":
            destination =  os.path.join(path_2,"training","P", filename)
            # copy only files
            if os.path.isfile(full_path):
                shutil.copy(full_path, destination)

        elif file.split('_')[-1][0] == "0":
            destination =  os.path.join(path_2,"training","0", filename)
            # copy only files
            if os.path.isfile(full_path):
                shutil.copy(full_path, destination)
    # test that the files are copied correctly

    # print length of folders
    print("Number of files in training folder:")
    print(len(os.listdir(os.path.join(path_2, "training", "N")))+len(os.listdir(os.path.join(path_2, "training", "P")))+len(os.listdir(os.path.join(path_2, "training", "0"))))
    print(len(train_folder_content))



    for file in val_folder_content:
        full_path = os.path.join(path_1, "validation", "INPUT_IMAGES", file)
        filename = file.split("\\")[-1]
        if file.split('_')[-1][0] == "N":
            destination =  os.path.join(path_2,"validation","N", filename)
            # copy only files
            if os.path.isfile(full_path):
                shutil.copy(full_path, destination)

        elif file.split('_')[-1][0] == "P":
            destination =  os.path.join(path_2,"validation","P", filename)
            # copy only files
            if os.path.isfile(full_path):
                shutil.copy(full_path, destination)

        elif file.split('_')[-1][0] == "0":
            destination =  os.path.join(path_2,"validation","0", filename)
            # copy only files
            if os.path.isfile(full_path):
                shutil.copy(full_path, destination)

    # print length of folders
    print("Number of files in val folder:")
    print(len(os.listdir(os.path.join(path_2, "validation", "N")))+len(os.listdir(os.path.join(path_2, "validation", "P")))+len(os.listdir(os.path.join(path_2, "validation", "0"))))
    print(len(val_folder_content))


    for file in test_folder_content:
        full_path = os.path.join(path_1, "testing", "INPUT_IMAGES", file)
        filename = file.split("\\")[-1]
        if file.split('_')[-1][0] == "N":
            destination =  os.path.join(path_2,"testing","N", filename)
            # copy only files
            if os.path.isfile(full_path):
                shutil.copy(full_path, destination)

        elif file.split('_')[-1][0] == "P":
            destination =  os.path.join(path_2,"testing","P", filename)
            # copy only files
            if os.path.isfile(full_path):
                shutil.copy(full_path, destination)

        elif file.split('_')[-1][0] == "0":
            destination =  os.path.join(path_2,"testing","0", filename)
            # copy only files
            if os.path.isfile(full_path):
                shutil.copy(full_path, destination)

    # print length of folders
    print("Number of files in test folder:")
    print(len(os.listdir(os.path.join(path_2, "testing", "N")))+len(os.listdir(os.path.join(path_2, "testing", "P")))+len(os.listdir(os.path.join(path_2, "testing", "0"))))
    print(len(test_folder_content))


def move():
    path_2 = "/u/data/s194333/ADLCV_project/data"

    stage = ['training', 'validation','testing']
    folders = ['N', 'P']
    expo = ['1', '2']

    for s in stage:
        for f in folders:
            for e in expo:
                folder = os.path.join(path_2, s, f, e)
                os.makedirs(folder, exist_ok=True)

    for s in stage:
        for f in folders:
            
            folder_content = os.listdir(os.path.join(path_2,s, f))
            print(os.path.join(path_2,s, f))

            for img in folder_content:
                full_path = os.path.join(path_2, s, f, img)
                if os.path.isfile(full_path):
                    if img.split('_')[-1][3] == "5":
                        destination =  os.path.join(path_2, s, f, "2", img)
                        shutil.move(full_path, destination)   

                    else:
                        destination =  os.path.join(path_2,s,f, "1", img)
                        shutil.move(full_path, destination)     

if __name__ == "__main__":
    transfer_files()
    move()


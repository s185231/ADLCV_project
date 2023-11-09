from PIL import Image
import os
import shutil

def resize_images(input_folder, output_folder):
    # Ensure the output folder exists, or create it if it doesn't
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files and directories in the input folder
    for item in os.listdir(input_folder):
        item_path = os.path.join(input_folder, item)
        output_item_path = os.path.join(output_folder, item)
        if os.path.isfile(item_path) and item.endswith(('.jpg', '.JPG')) and "GT_IMAGES" not in str(item_path) and "INPUT_IMAGES" not in str(item_path) and "expert" not in str(item_path):
            #print("expert" in str(item_path))
        
            # Open the image
            with Image.open(item_path) as img:
                # Resize the image to 128x128
                img = img.resize((128, 128))
                
                # Recreate the directory structure in the output folder
                os.makedirs(os.path.dirname(output_item_path), exist_ok=True)
                
                # Save the resized image to the output folder with the same directory structure
                img.save(output_item_path)

        elif os.path.isdir(item_path):
            # If it's a directory, recursively call the function
            resize_images(item_path, output_item_path)

if __name__ == "__main__":
    input_folder = '/u/data/s194333/ADLCV_project/data/'
    output_folder = '/u/data/s185231/ADLCV_project/data/'
    resize_images(input_folder, output_folder)
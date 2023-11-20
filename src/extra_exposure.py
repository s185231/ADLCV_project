import os
from PIL import Image

def exposure(img, alpha=0.05):
    """
    Function that changes the exposure of an image
    """
    # Change exposure
    img = img.point(lambda i: i * (1/alpha))

    return img

# Add if __name__ == "__main__":
if __name__ == "__main__":

    stages = ["training", "testing", "validation"]

    for s in stages:
        path_train = os.path.join("/u/data/s185231/ADLCV_project/data", s, "0")

        path_new_N = os.path.join("/u/data/s185231/ADLCV_project/data", s, "N/3")
        path_new_P = os.path.join("/u/data/s185231/ADLCV_project/data", s, "P/3")

        # Load images
        os.makedirs(path_new_N, exist_ok=True)
        os.makedirs(path_new_P, exist_ok=True)

        # For each image
        for img_name in os.listdir(path_train):
            # Load image
            print(os.path.join(path_train, img_name))
            img = Image.open(os.path.join(path_train, img_name))

            # Change exposure
            img_N = exposure(img, alpha=4)
            img_P = exposure(img, alpha=0.25)

            # Save image
            img_N.save(os.path.join(path_new_N, img_name))
            img_P.save(os.path.join(path_new_P, img_name))

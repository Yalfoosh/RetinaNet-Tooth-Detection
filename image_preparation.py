from PIL import Image
from resizeimage import resizeimage

import os

source_images_folder = os.path.abspath(os.curdir + "/data/SourceImages")
destination_images_folder = os.path.abspath(os.curdir + "/data/Images")


# Rescales all images in the source folder to have the width specified and saves them to the destination folder.
def rescale_images(width: int,
                   image_source_folder: str = source_images_folder,
                   image_destination_folder: str = destination_images_folder):
    for filename in os.listdir(image_source_folder):
        rescale_image(width, filename, image_source_folder, image_destination_folder)


# Rescales a specific image.
def rescale_image(width: int,
                  image_filename: str,
                  image_source_folder: str = source_images_folder,
                  image_destination_folder: str = destination_images_folder):
    height = width / 2
    image_path = os.path.join(image_source_folder, image_filename)
    resulting_image_path = os.path.join(image_destination_folder, image_filename)

    with open(image_path, 'r+b') as file:
        with Image.open(file) as image:
            cover = resizeimage.resize_cover(image, [width, height])

            if os.path.exists(resulting_image_path):
                os.remove(resulting_image_path)

            cover.save(resulting_image_path, image.format)

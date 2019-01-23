import xml_to_csv as xml
import image_preparation as image

import os

# The height isn't necessary since we're keeping the aspect ratio.
image_size = [3256, 1536]

# Make sure to change if the values in main.py are changed as well!
csv_folder = os.path.abspath(os.curdir + "/data/CSV")
images_folder = os.path.abspath(os.curdir + "/data/Images")
xml_folder = os.path.abspath(os.curdir + "/data/XML")

# Serves as a changelog so the image conversion can be automatic.
last_preparation_path = os.path.join(os.path.abspath(os.path.curdir), "last_preparation.txt")


# By default, scales down to 0.125 size.
def prepare_data(scale_down: float = 8):
    new_width = int(float(image_size[0]) / scale_down)

    if os.path.exists(last_preparation_path):
        read_width = int(open(last_preparation_path).read())
        if new_width == read_width:
            return

    print("Converting images to " + str(new_width) + "x" + str(int(new_width/2)) + "...")
    image.rescale_images(new_width)
    print("Generating CSV from XML files...")
    xml.convert_xml_to_csv(xml_folder, images_folder, csv_folder, scale_down)

    if os.path.exists(last_preparation_path):
        os.remove(last_preparation_path)

    f = open(last_preparation_path, "r+")
    f.write(str(new_width))

import os
import shutil

import xml_to_csv as xml
import image_preparation as image
import globals


# The height isn't necessary since we're keeping the aspect ratio.
image_size = [3256, 1536]


# Serves as a changelog so the image conversion can be automatic.
last_preparation_path = os.path.join(os.path.abspath(os.path.curdir), "last_preparation.txt")


# By default, scales down to 0.125 size.
def prepare_data(scale_down: float = 1):
    if scale_down is not 1:
        new_width = int(float(image_size[0]) / scale_down)
    else:
        new_width = image_size[0]

    if os.path.exists(last_preparation_path):
        read_width = int(open(last_preparation_path).read())
        if new_width == read_width:
            return

    if scale_down is not 1:
        print("Converting images to width: " + str(new_width) + "...")
        image.rescale_images(new_width)
    else:
        print("Copying images from " + str(globals.source_images_folder) + " to " + str(globals.images_folder))

        for file_name in os.listdir(globals.source_images_folder):
            from_file_name = os.path.join(globals.source_images_folder, file_name)
            to_file_name = os.path.join(globals.images_folder, file_name)

            shutil.copy(from_file_name, to_file_name)

    print("Generating CSV from XML files...")
    xml.convert_xml_to_csv(globals.xml_folder, globals.images_folder, globals.csv_folder, scale_down)

    if os.path.exists(last_preparation_path):
        os.remove(last_preparation_path)

    f = open(last_preparation_path, "a")
    f.write(str(new_width))

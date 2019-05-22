import os
import shutil

from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
from typing import List, Dict

import xml_to_csv as xml
import image_preparation as image
import globals


# The height isn't necessary since we're keeping the aspect ratio.
image_size = [3256, 1536]


# Serves as a changelog so the image conversion can be automatic.
last_preparation_path = os.path.join(os.path.abspath(os.path.curdir), "last_preparation.txt")


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


def get_image_names_from_xml(xml_folder_path: str):
    image_names = list()

    for xml_name in os.listdir(xml_folder_path):
        image_names.append(Path(os.path.join(xml_folder_path, xml_name)).stem + ".jpg")

    return image_names


def remove_redundant_images(image_file_names: List[str], image_folder_path: str):
    image_file_names = set(image_file_names)

    for image_path in os.listdir(image_folder_path):
        image_name = Path(image_path).stem + ".jpg"

        if image_name not in image_file_names:
            print("{} is not found".format(image_name))
            # os.remove(os.path.join(image_folder_path, image_path))


def copy_images_to_new_location(image_folder_path: str, image_file_names: List[str], new_image_folder_path: str):
    if image_folder_path == new_image_folder_path:
        return

    if not os.path.exists(new_image_folder_path):
        os.mkdir(new_image_folder_path)

    for image_file_name in image_file_names:
        full_path = os.path.join(image_folder_path, image_file_name)

        if os.path.exists(full_path):
            shutil.copy(full_path, os.path.join(new_image_folder_path, image_file_name))


def get_image_path_to_xml_path_dict(xml_folder_path: str, image_folder_path: str):
    images_to_xml = dict()

    images = map(lambda x: Path(x).stem, os.listdir(image_folder_path))

    for image_name in images:
        supposed_xml = os.path.join(os.path.join(os.curdir, xml_folder_path), image_name + ".xml")

        images_to_xml[os.path.join(os.path.join(os.curdir, image_folder_path), image_name + ".jpg")] = supposed_xml

    return images_to_xml


def get_image_path_to_xml_implicit(image_paths, xml_folder_path: str):
    image_to_xml = dict()

    for image_path in image_paths:
        image_to_xml[image_path] = os.path.join(xml_folder_path, Path(image_path).stem + ".xml")

    return image_to_xml


def create_blurred(image_to_xml: Dict[str, str], new_images_folder: str, csv_path, blur_radius: int):
    for image_path in image_to_xml:
        image_name = Path(image_path).stem
        new_image_path = os.path.join(new_images_folder, image_name + " blur " + str(blur_radius) + ".jpg")

        Image.open(image_path) \
            .filter(ImageFilter.GaussianBlur(radius=blur_radius)) \
            .save(new_image_path)

        return new_image_path


def create_brightness(image_to_xml: Dict[str, str],
                      new_images_folder: str,
                      csv_path: str,
                      lightness_percentage: float = 0):
    lightness_float = 1.0 + lightness_percentage / 100.

    for image_path in image_to_xml:
        image_name = Path(image_path).stem
        new_image_name = image_name + " "

        if lightness_percentage > 0:
            new_image_name += "+"

        new_image_name += str(lightness_percentage) + "% bright"

        ImageEnhance.Brightness(Image.open(image_path))\
                    .enhance(lightness_float)\
                    .save(os.path.join(new_images_folder, new_image_name + ".jpg"))


def seclude_xml(xml_folder_path: str, image_folder_path: str, csv_folder_path: str):

    if not os.path.exists(csv_folder_path):
        os.mkdir(csv_folder_path)

    xml.convert_xml_to_csv(xml_folder=xml_folder_path,
                           images_folder=image_folder_path,
                           csv_destination_folder=csv_folder_path,
                           divide_by=1.0)


def get_normalized_csv_string(csv_folder_path: str):
    new_string = ""

    with open(os.path.join(csv_folder_path, "data.csv")) as f:
        for line in f.readlines():
            split_lines = line.split(",")
            split_lines[5] = split_lines[5].replace("\n", "")

            if split_lines[5] not in globals.xml_labels:
                if split_lines[5] != "IMPLANTAT":
                    if split_lines[5].endswith("I"):
                        split_lines[5] = "IMPLANTAT"
                    else:
                        split_lines[5] = split_lines[5][:2]

            string = ""

            for element in split_lines:
                string += str(element) + ","

            string = string[:-1]

            new_string += string + "\n"

    return new_string[:-1]


def override_normalized_csv(csv_folder_path: str):
    csv_name = os.path.join(csv_folder_path, "data.csv")
    fixed = get_normalized_csv_string(csv_folder_path)

    with open(csv_name, "w+") as f:
        f.write(fixed)


def get_weird(csv_path: str):
    weird_dict = dict()

    with open(csv_path) as file:
        for line in file.readlines():
            split_lines = line.split(",")
            if len(split_lines) < 6:
                continue

            split_lines[5] = split_lines[5].replace("\n", "")

            if split_lines[5] not in globals.xml_labels:
                if split_lines[0] not in weird_dict:
                    weird_dict[split_lines[0]] = dict()

                weird_dict[split_lines[0]][split_lines[5]] = True

    weird_list = set()

    for key in weird_dict:
        weird_list.add(key)

    return weird_list


def get_fixed_csv(csv_path: str, entries):
    new_csv = ""

    with open(csv_path) as f:
        for line in f.readlines():
            split_lines = line.split(",")

            if split_lines[0] in entries:
                continue
            else:
                string = ""

                for data in split_lines:
                    string += data + ","

                string = string[:-1]

                new_csv += string

    return new_csv


def override_fixed_csv(csv_path: str):
    fixed = get_fixed_csv(csv_path, get_weird(csv_path))

    with open(csv_path, "w+") as f:
        f.write(fixed)


def select_images_from_xml(csv_path: str):
    images = set()

    with open(csv_path) as f:
        for line in f.readlines():
            split_lines = line.split(",")

            if len(split_lines) < 6:
                continue

            images.add(split_lines[0])
            
    images = list(images)
    images.sort()
    
    return images


def fix_coordinates(csv_path: str):
    new_string = ""

    with open(csv_path) as f:
        for line in f.readlines():
            split_lines = line.split(",")

            min_x = int(split_lines[1])
            min_y = int(split_lines[2])
            max_x = int(split_lines[3])
            max_y = int(split_lines[4])

            if min_x >= max_x:
                max_x = min_x + 1

            if min_y >= max_y:
                max_y = min_y + 1

            string = "{},{},{},{},{},{}".format(split_lines[0],
                                                str(min_x), str(min_y), str(max_x), str(max_y),
                                                split_lines[5])

            new_string += string

    new_string = new_string.strip()

    with open(csv_path, "w") as f:
        f.write(new_string)


def filter_data():
    xml_folder = os.path.join(os.curdir, "new data/xml")
    image_folder = os.path.join(os.curdir, "new data/images")
    new_image_folder = os.path.join(os.curdir, "new data/new images")
    csv_folder = os.path.join(os.curdir, "new data/temp csv")
    csv_double_path = os.path.join(os.curdir, "new data/temp csv/data.csv")

    seclude_xml(xml_folder_path=xml_folder, image_folder_path=image_folder, csv_folder_path=csv_folder)
    override_normalized_csv(csv_folder)
    override_fixed_csv(csv_double_path)
    fix_coordinates(csv_double_path)

    """selected_images = select_images_from_xml(csv_double_path)
    image_to_xml = get_image_path_to_xml_implicit(selected_images, xml_folder)

    create_blurred(image_to_xml=image_to_xml,
                   new_images_folder=new_image_folder,
                   csv_path=csv_double_path,
                   blur_radius=3)"""


# filter_data()

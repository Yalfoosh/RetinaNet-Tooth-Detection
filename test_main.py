import os
from pathlib import Path
from copy import deepcopy

from data_preparation import get_image_names_from_xml, remove_redundant_images, copy_images_to_new_location,\
                             get_image_path_to_xml_path_dict, create_blurred, create_with_brightness


def get_classes(class_path: str):
    to_return = set()

    with open(class_path) as f:
        lines = f.readlines()

        for line in lines:
            to_return.add(line.split(",")[5])

    return to_return


def find_ineligible(class_path: str):
    with open(class_path) as f:
        lines = f.readlines()

        for line in lines:
            line = Path(line.split(",")[0]).stem

            if len(line) > 5 and line[6] == "(":
                print(line)


def fix_min_max(class_path: str):
    new_file = open(class_path + "_new", mode="w+")
    new_lines = list()

    with open(class_path) as f:
        lines = f.readlines()

        for line in lines:
            line = line.split(",")
            new_line = deepcopy(line)

            if int(line[1]) >= int(line[3]):
                new_line[3] = str(int(line[1]) + 1)

            if int(line[2]) >= int(line[4]):
                new_line[4] = str(int(line[2]) + 1)

            some_new_line = ""

            for l in new_line:
                some_new_line += l + ","

            new_lines.append(some_new_line[:-1])

        new_file.writelines(new_lines)
        new_file.close()


def transform_images():
    xml_path = os.path.join(os.curdir, "new data/xml")
    image_path = os.path.join(os.curdir, "new data/images")
    new_image_path = os.path.join(os.curdir, "new data/new images")

    image_file_names = get_image_names_from_xml(xml_folder_path=xml_path)

    # copy_images_to_new_location(image_path, image_file_names, new_image_path)

    remove_redundant_images(image_file_names=image_file_names, image_folder_path=new_image_path)

    images_to_xml = get_image_path_to_xml_path_dict(xml_folder_path=xml_path, image_folder_path=new_image_path)
    create_blurred(images_to_xml, new_images_folder=new_image_path, new_xml_folder=xml_path, blur_radius=3)

    create_with_brightness(images_to_xml, new_images_folder=new_image_path, new_xml_folder=xml_path,
                           lightness_percentage=-25)
    create_with_brightness(images_to_xml, new_images_folder=new_image_path, new_xml_folder=xml_path,
                           lightness_percentage=-50)
    create_with_brightness(images_to_xml, new_images_folder=new_image_path, new_xml_folder=xml_path,
                           lightness_percentage=25)
    create_with_brightness(images_to_xml, new_images_folder=new_image_path, new_xml_folder=xml_path,
                           lightness_percentage=50)


def main():
    find_ineligible(os.path.join(os.curdir, "new data/csv/data.csv"))
    find_ineligible(os.path.join(os.curdir, "new data/validation csv/data.csv"))


main()

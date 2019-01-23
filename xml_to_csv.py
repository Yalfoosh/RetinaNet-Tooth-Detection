import xml.etree.ElementTree as XMLParser
import os
import sys


# Assuming you have 1 xml per picture, converts them all to a single .csv file. Divide_by specifies how many times the
# the image was scaled down. By default it isn't scaled at all.
def convert_xml_to_csv(xml_folder: str, images_folder: str, csv_destination_folder: str, divide_by: float = 1):
    csv_string = ""

    if os.path.exists(xml_folder):
        if not os.path.exists(csv_destination_folder):
            print(csv_destination_folder + " does not exist, creating it...", file=sys.stderr)
            os.makedirs(csv_destination_folder)

        for filename in os.listdir(xml_folder):
            if filename.endswith(".xml"):
                csv_string += xml_to_csv_string(xml_folder, filename, images_folder, divide_by) + "\n"

        path_to_write = os.path.join(csv_destination_folder, "data.csv")

        if os.path.exists(path_to_write):
            os.remove(path_to_write)

        file = open(os.path.join(csv_destination_folder, "data.csv"), "w+")
        file.write(csv_string.strip())
        file.close()
    else:
        raise IOError(xml_folder + " does not exist!")


# Converts a given xml to a .csv string.
def xml_to_csv_string(xml_folder: str, xml_name: str, images_folder: str, divide_by: float = 1) -> str:
    xml_path = os.path.join(xml_folder, xml_name)
    if os.path.exists(xml_path):
        to_return = ""
        parsed_tree = XMLParser.parse(xml_path).getroot()

        for obj in parsed_tree.findall("object"):
            to_return += os.path.abspath(os.path.join(images_folder, xml_name.replace(".xml", ".jpg")))

            # Automatically rescaled the bounding boxes as well.
            for bounding_box in obj.findall("bndbox"):
                for xmin in bounding_box.findall("xmin"):
                    to_return += ","

                    if divide_by is not 1:
                        to_return += str(int(float(xmin.text) / divide_by))
                    else:
                        to_return += xmin.text
                    break

                for ymin in bounding_box.findall("ymin"):
                    to_return += ","

                    if divide_by is not 1:
                        to_return += str(int(float(ymin.text) / divide_by))
                    else:
                        to_return += ymin.text
                    break

                for xmax in bounding_box.findall("xmax"):
                    to_return += ","

                    if divide_by is not 1:
                        to_return += str(int(float(xmax.text) / divide_by))
                    else:
                        to_return += xmax.text
                    break

                for ymax in bounding_box.findall("ymax"):
                    to_return += ","

                    if divide_by is not 1:
                        to_return += str(int(float(ymax.text) / divide_by))
                    else:
                        to_return += ymax.text
                    break

                break

            for name in obj.findall("name"):
                to_return += "," + name.text
                break

            to_return += "\n"

        return to_return.strip()
    else:
        raise IOError(xml_path + " does not exist!")

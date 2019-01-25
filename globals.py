import os
import datetime

# If you're changing these, be sure to change them in data_preparation.py as well.
csv_folder = os.path.abspath(os.path.join(os.curdir, "data/CSV"))
images_folder = os.path.abspath(os.path.join(os.curdir, "data/Images"))
source_images_folder = os.path.abspath(os.path.join(os.curdir, "data/SourceImages"))
xml_folder = os.path.abspath(os.path.join(os.curdir, "data/XML"))

model_save_folder = os.path.abspath(os.curdir + "/save")

evaluation_images_folder = os.path.abspath(os.path.join(os.curdir, "[Evaluation Images]"))
resulting_images_folder = os.path.abspath(os.path.join(os.curdir, "[Inference Results]"))

# 32 + IMPLANTATI
number_of_teeth_types = 33
number_of_files = 3995

# This one was set as default although other backbones can be used.
default_backbone = "resnet50"

data_file = os.path.abspath(os.path.join(csv_folder, "data.csv"))
class_file = os.path.abspath(os.path.join(csv_folder, "classes.csv"))

file_value_separator = " - "

labels_to_names =\
    {
        0: "1-1",   1: "1-2",   2: "1-3",   3: "1-4",   4: "1-5",   5: "1-6",   6: "1-7",   7: "1-8",
        8: "2-1",   9: "2-2",   10: "2-3",  11: "2-4",  12: "2-5",  13: "2-6",  14: "2-7",  15: "2-8",
        16: "3-1",  17: "3-2",  18: "3-3",  19: "3-4",  20: "3-5",  21: "3-6",  22: "3-7",  23: "3-8",
        24: "4-1",  25: "4-2",  26: "4-3",  27: "4-4",  28: "4-5",  29: "4-6",  30: "4-7",  31: "4-8",
        32: "Implantat"
    }

iou_threshold = 0.5

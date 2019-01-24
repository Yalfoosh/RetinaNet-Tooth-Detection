# Note:
#
# boxes, scores, labels = model.predict_on_batch(input)

import data_preparation as preparation
import save_load

from keras_retinanet import models as retinanet_models
from keras_retinanet import losses as retinanet_losses
from keras_retinanet.preprocessing import csv_generator as csv
import keras
import os
import datetime

#
# Methods
#


def prepare_model(save_folder: str):
    if os.path.exists(save_folder) and len(os.listdir(save_folder)) is not 0:
        file_name = os.listdir(save_folder)[-1]
        loaded_model = save_load.load_model(save_folder, file_name)

        if loaded_model is not None:
            return loaded_model
    else:
        return retinanet_models.backbone(default_backbone).retinanet(num_classes=number_of_teeth_types), 0


# The reason this is a method is so when a OOM error occurs, it tries again.
def train(epochs: int):
    loss = {"regression": retinanet_losses.smooth_l1(), "classification": retinanet_losses.focal()}
    generator = csv.CSVGenerator(csv_data_file=data_file, csv_class_file=class_file)

    i = last_epoch
    while i < epochs:
        try:
            # I seem to be getting way faster convergence with these values than the flat 1e-5 learning rate.
            if i < 2:
                optimizer = keras.optimizers.adam(lr=1e-3, clipnorm=0.001)
            elif i < 10:
                optimizer = keras.optimizers.adam(lr=1e-4, clipnorm=0.001)
            elif i < 70:
                optimizer = keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
            else:
                optimizer = keras.optimizers.adam(lr=1e-6, clipnorm=0.001)

            model.compile(loss=loss, optimizer=optimizer)
            history = model.fit_generator(generator=generator, verbose=1, workers=4, max_queue_size=32,
                                          steps_per_epoch=number_of_files)

            model_save_file_name = save_load.generate_save_string(datetime.datetime.now(),
                                                                  i + 1,
                                                                  history.history["regression_loss"][0],
                                                                  history.history["classification_loss"][0])

            model.save(os.path.join(model_save_folder, model_save_file_name), overwrite=True)
            i += 1
        except Exception:
            continue

#
# Variables
#


# If you're changing these, be sure to change them in data_preparation.py as well.
csv_folder = os.path.abspath(os.curdir + "/data/CSV")
images_folder = os.path.abspath(os.curdir + "/data/Images")
xml_folder = os.path.abspath(os.curdir + "/data/XML")

model_save_folder = os.path.abspath(os.curdir + "/save")
model_path = os.path.join(model_save_folder, str(datetime.datetime.now()).replace(":", "-") + ".h5")

# 32 + IMPLANTATI
number_of_teeth_types = 33
number_of_files = 3995

# This one was set as default although other backbones can be used.
default_backbone = "resnet50"

data_file = os.path.abspath(os.path.join(csv_folder, "data.csv"))
class_file = os.path.abspath(os.path.join(csv_folder, "classes.csv"))

#
# Code
#

preparation.prepare_data()
model, last_epoch = prepare_model(model_save_folder)
train(100)

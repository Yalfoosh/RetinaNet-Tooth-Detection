# Note:
#
# boxes, scores, labels = model.predict_on_batch(input)

import data_preparation as preparation

from keras_retinanet import models as retinanet_models
from keras_retinanet import losses as retinanet_losses
from keras_retinanet.preprocessing import csv_generator as csv
import keras
import os
import datetime

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
epochs = 11

# This one was set as default although other backbones can be used.
default_backbone = "resnet50"

# Make sure to change accordingly based on csv_folder.
data_file = os.path.abspath(os.curdir + "/data/CSV/data.csv")
class_file = os.path.abspath(os.curdir + "/data/CSV/classes.csv")

loaded = False

#
# Code
#

preparation.prepare_data()

# This automatically loads the last save.
if os.path.exists(model_save_folder) and len(os.listdir(model_save_folder)) is not 0 and False:
    model = keras.models.load_model(filepath=os.path.join(model_save_folder, os.listdir(model_save_folder)[-1]))

    loaded = True
else:
    model = retinanet_models.backbone(default_backbone)\
                            .retinanet(num_classes=number_of_teeth_types)

loss = {"regression": retinanet_losses.smooth_l1(), "classification": retinanet_losses.focal()}
generator = csv.CSVGenerator(csv_data_file=data_file, csv_class_file=class_file)

# Slows down learning rates.
if loaded is True:
    start_epochs = 1
else:
    start_epochs = 0

# I seem to be getting way faster convergence with these values that the flat 1e-5 learning rate.
for i in range(start_epochs, epochs):
    if i == 0:
        optimizer = keras.optimizers.adam(lr=1e-3, clipnorm=0.001)
    elif i < 5:
        optimizer = keras.optimizers.adam(lr=1e-4, clipnorm=0.001)
    elif i < 10:
        optimizer = keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    else:
        optimizer = keras.optimizers.adam(lr=1e-6, clipnorm=0.001)

    model.compile(loss=loss, optimizer=optimizer)
    model.fit_generator(generator=generator, verbose=1, workers=4, max_queue_size=32, steps_per_epoch=3995)

    # Every 5 epochs it creates a new save.
    if i % 5 == 0:
        model_path = os.path.join(model_save_folder, str(datetime.datetime.now()).replace(":", "-") + ".h5")

    model.save(model_path, overwrite=True)

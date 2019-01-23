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
last_epoch = 0
epochs = 100

# This one was set as default although other backbones can be used.
default_backbone = "resnet50"

# Make sure to change accordingly based on csv_folder.
data_file = os.path.abspath(os.path.join(os.curdir, "data/CSV/data.csv"))
class_file = os.path.abspath(os.path.join(os.curdir, "data/CSV/classes.csv"))

#
# Code
#

preparation.prepare_data()

# This automatically loads the last save.
if os.path.exists(model_save_folder) and len(os.listdir(model_save_folder)) is not 0:
    model_file_name = os.listdir(model_save_folder)[-1]
    result = save_load.load_model(model_save_folder, model_file_name)

    if result is not None:
        model, last_epoch = result
    else:
        model = retinanet_models.backbone(default_backbone) \
                                .retinanet(num_classes=number_of_teeth_types)
else:
    model = retinanet_models.backbone(default_backbone)\
                            .retinanet(num_classes=number_of_teeth_types)

loss = {"regression": retinanet_losses.smooth_l1(), "classification": retinanet_losses.focal()}
generator = csv.CSVGenerator(csv_data_file=data_file, csv_class_file=class_file)

# I seem to be getting way faster convergence with these values that the flat 1e-5 learning rate.
for i in range(last_epoch, epochs):
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

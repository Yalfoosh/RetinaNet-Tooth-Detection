# Note:
#
# boxes, scores, labels = model.predict_on_batch(input)

import data_preparation as preparation
import save_load
import globals

from keras_retinanet import models as retinanet_models
from keras_retinanet import losses as retinanet_losses
from keras_retinanet.preprocessing import csv_generator as csv
import keras
import os
import datetime
import gc
import sys

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
        return retinanet_models.backbone(globals.default_backbone)\
                               .retinanet(num_classes=globals.number_of_teeth_types), 0


def train(training_model, last_epoch: int, epochs: int):
    loss = {"regression": retinanet_losses.smooth_l1(), "classification": retinanet_losses.focal()}
    generator = csv.CSVGenerator(csv_data_file=globals.data_file, csv_class_file=globals.class_file)

    i = last_epoch
    while i < epochs:
        try:
            # I seem to be getting way faster convergence with these values than the flat 1e-5 learning rate.
            if i < 5:
                optimizer = keras.optimizers.adam(lr=1e-3, clipnorm=0.001)
            elif i < 10:
                optimizer = keras.optimizers.adam(lr=1e-4, clipnorm=0.001)
            elif i < 70:
                optimizer = keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
            else:
                optimizer = keras.optimizers.adam(lr=1e-6, clipnorm=0.001)

            training_model.compile(loss=loss, optimizer=optimizer)
            history = training_model.fit_generator(generator=generator, verbose=1)

            i += 1

            model_save_file_name = save_load.generate_save_string(datetime.datetime.now(),
                                                                  i,
                                                                  history.history["regression_loss"][0],
                                                                  history.history["classification_loss"][0])

            training_model.save(os.path.join(globals.model_save_folder, model_save_file_name), overwrite=True)

            # Keras has some memory leaks, this should fix it.
            del training_model
            del history
            gc.collect()

            training_model = prepare_model(globals.model_save_folder)[0]
        except Exception as e:
            print("\n")
            print(e, file=sys.stderr)

            i += 1
            model_save_file_name = save_load.generate_save_string(datetime.datetime.now(), i, 1, 1)
            training_model.save(os.path.join(globals.model_save_folder, model_save_file_name), overwrite=True)

            del training_model
            gc.collect()

            training_model = prepare_model(globals.model_save_folder)[0]
            continue


def main():
    preparation.prepare_data()
    model, last_epoch = prepare_model(globals.model_save_folder)
    train(model, last_epoch, 100)


#
# Code
#

main()

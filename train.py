# Note:
#
# boxes, scores, labels = model.predict_on_batch(input)

import xml_to_csv as xml
import data_preparation as preparation
import save_load
import globals

from keras_retinanet import models as retinanet_models
from keras_retinanet import losses as retinanet_losses
from keras_retinanet.preprocessing import csv_generator as csv
from keras import backend as K
import keras
import os
import datetime
import gc

images_folder = os.path.join(os.curdir, "new data/new images")
validation_folder = os.path.join(os.curdir, "new data/validation")

xml_folder = os.path.join(os.curdir, "new data/xml")
validation_xml_folder = os.path.join(os.curdir, "new data/validation xml")

csv_data = os.path.join(os.curdir, "new data/csv")
csv_validation_data = os.path.join(os.curdir, "new data/validation csv")
csv_classes = os.path.join(os.curdir, "new data/classes.csv")

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
    optimizer = keras.optimizers.adam(lr=3e-4, clipnorm=0.001)

    training_model.compile(loss=loss, optimizer=optimizer)

    generator = csv.CSVGenerator(csv_data_file=os.path.join(csv_data, "data.csv"), csv_class_file=csv_classes)
    """validation_generator = csv.CSVGenerator(csv_data_file=os.path.join(csv_validation_data, "data.csv"),
                                            csv_class_file=csv_classes)"""

    history = None

    i = last_epoch
    while i < epochs:
        try:
            history = training_model.fit_generator(generator=generator, verbose=1, epochs=5)

            i += 1

            model_save_file_name = save_load.generate_save_string(datetime.datetime.now(),
                                                                  i,
                                                                  history.history["regression_loss"][0],
                                                                  history.history["classification_loss"][0])

            training_model.save(os.path.join(globals.model_save_folder, model_save_file_name), overwrite=True)

            # Keras has some memory leaks, this should fix it.
            K.clear_session()
            del training_model
            del history
            gc.collect()

            training_model = prepare_model(globals.model_save_folder)[0]
        except Exception as e:
            print(e)
            exit()


def main():
    """xml.convert_xml_to_csv(xml_folder=xml_folder,
                           images_folder=images_folder,
                           csv_destination_folder=csv_data,
                           divide_by=1.0)

    xml.convert_xml_to_csv(xml_folder=validation_xml_folder,
                           images_folder=validation_folder,
                           csv_destination_folder=csv_validation_data,
                           divide_by=1.0)"""

    # print("AAAAAAA")

    model, last_epoch = prepare_model(globals.model_save_folder)
    train(model, last_epoch, 1000)


#
# Code
#

main()

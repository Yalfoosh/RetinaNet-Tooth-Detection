import datetime
import keras
import os
from keras_retinanet.models import resnet

separator = " - "


def generate_save_string(first_saved: datetime.datetime,
                         epochs: int, regression_loss: float, classification_loss: float,
                         extension: str = ".h5"):
    to_return = ""

    to_return += str(first_saved.year).zfill(4) + str(first_saved.month).zfill(2) + str(first_saved.day).zfill(2) + "_"
    to_return += str(first_saved.hour).zfill(2) + str(first_saved.minute).zfill(2) + str(first_saved.second).zfill(2)

    to_return += separator + str(epochs)

    to_return += separator + str(round(regression_loss, 4))
    to_return += separator + str(round(classification_loss, 4))
    to_return += separator + extension

    return to_return


def read_save_string(save_name: str) -> (str, int, int, int, str) or None:
    units = save_name.split(separator)

    if len(units) is not 5:
        return None

    datetime_string = units[0]
    epochs = int(units[1])
    regression_loss = float(units[2])
    class_loss = float(units[3])
    extension = units[4][1:]

    return datetime_string, epochs, regression_loss, class_loss, extension


def load_model(folder_path: str, file_name: str):
    load_path = os.path.join(folder_path, file_name)
    date, epochs, reg_loss, class_loss, extension = read_save_string(file_name)

    print("Loading model from " + load_path +
          "\nEpochs ran: " + str(epochs) +
          "\nRegression loss: " + str(reg_loss) + ", Classification loss: " + str(class_loss))

    custom_objects = resnet.ResNetBackbone("resnet50").custom_objects
    model = keras.models.load_model(filepath=load_path, custom_objects=custom_objects)

    return model, epochs

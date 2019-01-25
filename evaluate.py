# Note:
#
# boxes, scores, labels = model.predict_on_batch(input)

import save_load
import globals

from keras_retinanet.utils.image import read_image_bgr, preprocess_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

import cv2
import datetime
import matplotlib.pyplot as plt
import numpy
import os
import scipy.misc

model = None


def get_model(folder_path: str, file_name: str):
    return save_load.load_model(folder_path, file_name)[0]


def get_latest_model(folder_path: str):
    return save_load.load_latest_model(folder_path)


def get_inference_model(training_model):
    return save_load.get_inference_model(training_model)


def mark_and_return_image(inference_model=model, image_path: str = ""):
    if os.path.exists(image_path) and inference_model is not None:
        original_image = read_image_bgr(image_path)

        resulting_image = original_image.copy()
        resulting_image = cv2.cvtColor(resulting_image, cv2.COLOR_BGR2RGB)

        original_image = preprocess_image(original_image)

        boxes, scores, labels = inference_model.predict_on_batch(numpy.expand_dims(original_image, axis=0))

        # visualize detections
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # Scores are sorted so you can break the first time you encounter <0.5
            if score < globals.iou_threshold:
                break

            color = label_color(label)

            b = box.astype(int)
            draw_box(resulting_image, b, color=color)

            # caption = "{} {:.3f}".format(globals.labels_to_names[label], score)
            caption = "{}".format(globals.labels_to_names[label])
            draw_caption(resulting_image, b, caption)

        return resulting_image


def show_image(image):
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(image)
    plt.show()


def infer_and_save_result(inference_model: str = model,
                          image_path: str = "",
                          destination_path: str = "",
                          new_file_name: str = None):
    if not os.path.exists(destination_path):
        os.mkdir(destination_path)

    if os.path.exists(destination_path) and inference_model is not None:
        resulting_image = mark_and_return_image(inference_model, image_path)

        if new_file_name is None:
            new_file_name = os.path.basename(image_path)

        new_image_path = os.path.join(destination_path, new_file_name) + ".jpg"

        if resulting_image is not None:
            scipy.misc.toimage(resulting_image, cmin=0.0, cmax=...).save(new_image_path)
        else:
            raise IOError("Image not in correct format!")
    else:
        raise IOError("Either the model is missing, or the result folder!")


def auto_infer():
    if os.path.exists(globals.evaluation_images_folder):
        file_names = os.listdir(globals.evaluation_images_folder)

        for file_name in file_names:
            print("Evaluating " + str(file_name))
            current_image_path = os.path.join(globals.evaluation_images_folder, file_name)

            infer_and_save_result(model, current_image_path, globals.resulting_images_folder)


model = get_inference_model(get_latest_model(globals.model_save_folder)[0])

auto_infer()

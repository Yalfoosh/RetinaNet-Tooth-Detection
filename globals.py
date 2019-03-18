import os
import numpy as np

from typing import List, Dict

from typing import Tuple

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


def get_the_most_likely_detected_objects(detected_objects: List["DetectedObject"],
                                         minimum_score: float = 0.5,
                                         suppress_iou_higher_than: float = 0.5):
    likely_candidates = get_detected_objects_above_score(detected_objects=detected_objects, minimum_score=minimum_score)
    likely_candidates =\
        get_detected_objects_after_non_maximum_suppression(detected_objects=likely_candidates,
                                                           intersection_over_union_minimum=suppress_iou_higher_than)

    return likely_candidates


def get_detected_objects_after_non_maximum_suppression(detected_objects: List["DetectedObject"],
                                                       intersection_over_union_minimum: float = 0.5):
    objects_to_ignore = set()
    objects_to_consider = list()
    currently_overlapping_objects = list()

    for detected_object in detected_objects:
        if detected_object in objects_to_ignore:
            continue

        currently_overlapping_objects.clear()
        currently_overlapping_objects.append(detected_object)
        objects_to_ignore.add(detected_object)

        for detected_object_doppelganger in detected_objects:
            if detected_object_doppelganger in objects_to_ignore:
                continue

            if detected_object.intersection_over_union(detected_object_doppelganger) >= intersection_over_union_minimum:
                currently_overlapping_objects.append(detected_object_doppelganger)

        currently_overlapping_objects.sort(key=lambda x: x.score, reverse=True)
        objects_to_consider.append(currently_overlapping_objects[0])

        for i in range(1, len(currently_overlapping_objects)):
            objects_to_ignore.add(currently_overlapping_objects[i])

    return objects_to_consider


def get_detected_objects_with_highest_score_per_class(detected_objects: List["DetectedObject"]) \
        -> List["DetectedObject"]:
    label_to_detected_objects = dict()

    for detected_object in detected_objects:
        if detected_object.label not in label_to_detected_objects:
            label_to_detected_objects[detected_object.label] = list()

        label_to_detected_objects[detected_object.label].append(detected_object)

    for label in label_to_detected_objects:
        label_to_detected_objects[label].sort(key=lambda x: x.score, reverse=True)
        label_to_detected_objects[label] = label_to_detected_objects[label][0]

    return list(label_to_detected_objects.values())


def get_detected_objects_above_score(detected_objects: List["DetectedObject"],
                                     minimum_score: float = 0.5) -> List["DetectedObject"]:
    satisfactory = list()

    for detected_object in detected_objects:
        if detected_object.score >= minimum_score:
            satisfactory.append(detected_object)

    return satisfactory


class DetectedObject:
    def __init__(self, box: List[float or int], score: float, label: str):
        self.box = box
        self.score = score
        self.label = label

    @staticmethod
    def from_pairs(boxes, scores, labels) -> List["DetectedObject"]:
        object_list = list()

        for (box, score, label) in zip(boxes, scores, labels):
            object_list.append(DetectedObject(box=box, score=score, label=label))

        return object_list

    def is_at_same_place(self, other):
        return self.box[0] - other.box[0] < 0.0001 and self.box[1] - other.box[1] < 0.0001 \
               and self.box[2] - other.box[2] < 0.0001 and self.box[3] - other.box[3] < 0.0001

    def intersection_over_union(self, other: "DetectedObject") -> float:
        x1 = max(self.box[0], other.box[0])
        y1 = max(self.box[1], other.box[1])
        x2 = min(self.box[2], other.box[2])
        y2 = min(self.box[3], other.box[3])

        overlap = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

        first_box_area = (self.box[2] - self.box[0] + 1) * (self.box[3] - self.box[1] + 1)
        second_box_area = (other.box[2] - other.box[0] + 1) * (other.box[3] - other.box[1] + 1)

        return overlap / float(first_box_area + second_box_area - overlap)

    def __eq__(self, other):
        if isinstance(other, DetectedObject) and self.label == other.label and self.score - other.score < 0.0001:
            return self.is_at_same_place(other)

    def decompose(self):
        return self.box, self.score, self.label

    @staticmethod
    def decompose_list(detected_objects: List["DetectedObject"]):
        return_boxes = list()
        return_scores = list()
        return_labels = list()

        for detected_object in detected_objects:
            return_boxes.append(detected_object.box)
            return_scores.append(detected_object.score)
            return_labels.append(detected_object.label)

        return return_boxes, return_scores, return_labels

    def __hash__(self):
        some_hash = hash(self.label) ^ hash(self.score)

        for entry in self.box:
            some_hash ^= hash(entry)

        return some_hash

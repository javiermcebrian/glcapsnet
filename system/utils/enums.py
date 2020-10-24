import sys
import os
sys.path.append(os.path.abspath('../'))

from experiment.config import path_rgb, path_of, path_segmentation
from enum import Enum


class Mode(Enum):
    train = 'train'
    test = 'test'
    predict = 'predict'
    def __str__(self):
        return str(self.value)

class Dataset(Enum):
    train = 'train'
    val = 'val'
    test = 'test'
    predict = 'predict'
    def __str__(self):
        return str(self.value)

class Feature(Enum):
    rgb = path_rgb
    of = path_of
    segmentation_probabilities = path_segmentation
    all = 'all'
    def __str__(self):
        return str(self.value)

class Condition(Enum):
    daytime = 'daytime'
    weather = 'weather'
    landscape = 'landscape'
    def __str__(self):
        return str(self.value)
    @classmethod
    def to_list(cls):
        return list(map(lambda x: x.value, cls))

class Daytime(Enum):
    morning = 'morning'
    evening = 'evening'
    night = 'night'
    def __str__(self):
        return str(self.value)
    @classmethod
    def to_list(cls):
        return list(map(lambda x: x.value, cls))

class Weather(Enum):
    sunny = 'sunny'
    cloudy = 'cloudy'
    rainy = 'rainy'
    def __str__(self):
        return str(self.value)
    @classmethod
    def to_list(cls):
        return list(map(lambda x: x.value, cls))

class Landscape(Enum):
    downtown = 'downtown'
    countryside = 'countryside'
    highway = 'highway'
    def __str__(self):
        return str(self.value)
    @classmethod
    def to_list(cls):
        return list(map(lambda x: x.value, cls))

class GT(Enum):
    saliency = 'saliency'
    def __str__(self):
        return str(self.value)


ConditionMapper = {Condition.daytime: Daytime, Condition.weather: Weather, Condition.landscape: Landscape}

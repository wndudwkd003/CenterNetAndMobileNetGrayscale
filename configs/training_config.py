import easydict
import os

config = easydict.EasyDict()

config.DATASET = easydict.EasyDict()

config.DATASET.BASE_PATH = rf"D:\Datasets\result(gpt)\convert_result"
config.DATASET.TRAIN = "train"
config.DATASET.VALID = "valid"
config.DATASET.TEST = "test"
config.DATASET.ANNOTATION = "_export_annotations.coco.json"


config.DATASET.TRAIN_PATH = os.path.join(config.DATASET.BASE_PATH, config.DATASET.TRAIN)
config.DATASET.VALID_PATH = os.path.join(config.DATASET.BASE_PATH, config.DATASET.VALID)
config.DATASET.TEST_PATH = os.path.join(config.DATASET.BASE_PATH, config.DATASET.TEST)

config.TRAIN = easydict.EasyDict()
config.TRAIN.BATCH_SIZE = 32
config.TRAIN.LR = 0.001


config.DEBUG = easydict.EasyDict()
config.DEBUG.DEBUG_PRINT = False






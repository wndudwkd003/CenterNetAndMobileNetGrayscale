import easydict
import os

from .mobilenet_config import config as mobilenet_config

config = easydict.EasyDict()

##############################################################################################
# 학습 종류
config.TYPE = easydict.EasyDict()
TRAIN_TYPE = ["CLASSIFICATION", "OBJECT_DETECTION"]
TYPE = 1
config.TYPE.TRAIN_TYPE = TRAIN_TYPE[TYPE]

##############################################################################################
# 데이터셋 관련 설정
config.DATASET = easydict.EasyDict()
config.DATASET.BASE_PATH = rf"D:\Datasets\result(gpt)\convert_result"
config.DATASET.TRAIN = "train"
config.DATASET.VALID = "valid"
config.DATASET.TEST = "test"
config.DATASET.ANNOTATION = "_export_annotations.coco.json"
config.DATASET.NUM_CLASSES = 12
config.DATASET.TRAIN_PATH = os.path.join(config.DATASET.BASE_PATH, config.DATASET.TRAIN)
config.DATASET.VALID_PATH = os.path.join(config.DATASET.BASE_PATH, config.DATASET.VALID)
config.DATASET.TEST_PATH = os.path.join(config.DATASET.BASE_PATH, config.DATASET.TEST)

##############################################################################################
# 데이터셋 전처리 관련 설정
config.TRANSFORM = easydict.EasyDict()
config.TRANSFORM.RESIZE = 224
config.TRANSFORM.MEAN = 0.5
config.TRANSFORM.STD = 0.5

##############################################################################################
# 학습 관련 설정
config.TRAIN = easydict.EasyDict()
config.TRAIN.BATCH_SIZE = 32
config.TRAIN.LR = 0.001
config.TRAIN.EPOCH = 2

##############################################################################################
# 디버깅 관련 설정
config.DEBUG = easydict.EasyDict()
config.DEBUG.DEBUG_PRINT = False

##############################################################################################
# 가중치 파일 관련 설정
config.WEIGHT = easydict.EasyDict()
config.WEIGHT.SAVE = True
config.WEIGHT.FILE_NAME = r"model_weights"
config.WEIGHT.EXTENSION = r".pth"

if config.TYPE.TRAIN_TYPE == TRAIN_TYPE[0]:
    config.WEIGHT.BASE_PATH = rf"D:\Development\Pycharm Projects\CenterNetGrayscale\models\classification"
else:
    config.WEIGHT.BASE_PATH = rf"D:\Development\Pycharm Projects\CenterNetGrayscale\models\object_detection"

if mobilenet_config.BACKBONE.LIGHTWEIGHT:
    FILE_NAME = "lightweight_"
else:
    FILE_NAME = "original_"

config.WEIGHT.TOTAL_FILE_NAME = FILE_NAME+config.WEIGHT.FILE_NAME+config.WEIGHT.EXTENSION
config.WEIGHT.SAVE_PATH = os.path.join(config.WEIGHT.BASE_PATH, config.WEIGHT.TOTAL_FILE_NAME)

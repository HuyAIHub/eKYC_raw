from ultralytics import YOLO
from model.PaddleOCR.paddleocr import PaddleOCR,draw_ocr
from vietocr.tool.predictor import Predictor

class PretrainedModel:
    _instance = None
    def __new__(cls, cfg=None, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(PretrainedModel, cls).__new__(cls, *args, **kwargs)

            cls.segmentYolov8 = YOLO(cfg['segment_yolov8']['weight'])

            cls.classifyCardSideModel = YOLO(cfg['classify_card_side']['weight'])

            cls.detectTextModel = YOLO(cfg['detect_text_yolov8']['weight'])

            net_config = cfg['recognize_face_text']['net']
            # net_config.update(cfg['seq2seg'])
            net_config.update(cfg['transformer'])
            cls.recognizeFaceTextModel = Predictor(net_config)

            # cls.recognizeTextModel_CCCD_chip = PaddleOCR(det_db_thresh = 0.3, det_db_box_thresh = 0.3, 
            #                                              use_gpu=True,gpu_mem=500,lang="en")



        return cls._instance
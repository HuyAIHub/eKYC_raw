import cv2
import math
import numpy as np
from fastapi.responses import JSONResponse as handle_response
from utils_ocr.utils import handle_error
import config_app.constants as const
from process.PretrainedModel import PretrainedModel
from process.ocr.detect_recognize_face_text.detect_recognize_face_text_cccd_chip import detect_recognize_face_text_cccd_chip
from process.ocr.detect_recognize_face_text.detect_recognize_face_text_cccd import detect_recognize_face_text_cccd
from process.ocr.detect_recognize_face_text.detect_recognize_face_text_cmnd import detect_recognize_face_text_cmnd

from config_app.config import get_config
config_app, config_model = get_config()
models = PretrainedModel(config_model['ocr_model'])
model_detect_text = models.detectTextModel
model_recognition = models.recognizeFaceTextModel

def detect_recognize_face_text(image_crop_front, image_crop_back, card_side_predict, card_type_image, path_folder, log_obj):
    if card_type_image[0] == "cccd-chip":
        result_ocr,im_show_front, im_show_back, image_face, confidence = detect_recognize_face_text_cccd_chip(image_crop_front, 
                                                                                                                image_crop_back, 
                                                                                                                card_side_predict,
                                                                                                                model_detect_text, 
                                                                                                                model_recognition)
        message = "Bóc tách thành công"
        DetailCode = 201
    elif card_type_image[0] == "cmnd-cccd-2016":
        result_ocr,im_show_front, im_show_back, image_face, confidence = detect_recognize_face_text_cccd(image_crop_front, 
                                                                                                            image_crop_back, 
                                                                                                            card_side_predict,
                                                                                                            model_detect_text, 
                                                                                                            model_recognition)
    else:
        # cmnd-cu
        result_ocr,im_show_front, im_show_back, image_face, confidence = detect_recognize_face_text_cmnd(image_crop_front, 
                                                                                                            image_crop_back, 
                                                                                                            card_side_predict,
                                                                                                            model_detect_text, 
                                                                                                            model_recognition)
    if config_app['parameter']['save_img']:
        cv2.imwrite(path_folder + "/box_front.jpg", im_show_front) 
        cv2.imwrite(path_folder + "/box_back.jpg", im_show_back)

    print(result_ocr)
    log_obj.info("result_ocr:" + " " + str(result_ocr))
    result_ocr['confidence'] = confidence
    print("result_ocr", result_ocr)
    if not result_ocr["identCardNumber"].isnumeric():
        result_ocr = {}
    if not bool(result_ocr):
        return handle_error(const.ERROR['INVALID_ID_CARD'], const.ERROR['INVALID_MISSING_CODE'], 
                            400, const.DETAILCODE['INVALID_MISSING_CODE'])
    result = {
        "code": 200,
        "message": "Bóc tách thành công",
        "data": result_ocr,
        "DetailCode": 201
    }
    return handle_response(result)
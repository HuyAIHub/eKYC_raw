import time
import os
import cv2
from utils_ocr.utils import convert_pil_to_np, handle_error
from config_app.config import get_config
import config_app.constants as const

from process.ocr.detect_color.detect_color import detect_color
from utils_ocr.check_blur import check_blur
from process.ocr.segment_crop_rotate_card_yolov8.segment_crop_rotate_card_yolov8 import segment_crop_rotate_card_yolov8
from process.ocr.detect_recognize_face_text.detect_recognize_face_text import detect_recognize_face_text

config_app, config_model = get_config()

def warmup_api(log_obj):
    front = './data/cccd_chip_front.jpeg'
    back = './data/cccd_chip_back.jpeg'
    id_request = "010101"
    user = "warmup"
    ocr(front, back, id_request, user, log_obj)


def ocr(front, back, id_request, user, log_obj):
    user = str(user)
    print("----------------NEW_SESSION--------------")
    print("GuildID  = ", id_request)

    input1 = convert_pil_to_np(front)
    input2 = convert_pil_to_np(back)

    print("DETECT CROP CARD USING YOLOV8 SEGMENTATION")
    log_obj.info("DETECT CROP CARD USING YOLOV8 SEGMENTATION")
    d = time.time()
    image_crop, card_side_predict, card_type_image, ratio_area, count_side_overlap = segment_crop_rotate_card_yolov8([input1, input2])
    print("Time Yolov8 Segment & Rotate", time.time()-d)
    print(card_side_predict)
    print(card_type_image) 
    log_obj.info("Time Segmentation & Classification Yolov8:" + " " + str(time.time()-d))
    log_obj.info("card_side_predict:" + " " + str(card_side_predict))
    log_obj.info("card_type_image:" + " " + str(card_type_image))

    if card_side_predict[0] == "front":
        image_crop_front = image_crop[0]
        image_crop_back = image_crop[1]
        image_front = input1
        image_back = input2
    else:
        image_crop_front = image_crop[1]
        image_crop_back = image_crop[0]
        image_front = input2
        image_back = input1
    
    path_folder = './results/' + user + "/" + str(id_request)
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)
    if config_app['parameter']['save_img']:
        cv2.imwrite(path_folder + "/front.jpg", image_front) 
        cv2.imwrite(path_folder + "/back.jpg", image_back)
        if image_crop_front != '': cv2.imwrite(path_folder + "/front_crop.jpg", image_crop_front)
        if image_crop_back != '': cv2.imwrite(path_folder + "/back_crop.jpg", image_crop_back)

    if card_type_image[0] == "NO_MASK" or card_type_image[1] == "NO_MASK":
        return handle_error(const.ERROR['INVALID_ID_CARD'], const.ERROR['INVALID_MASK'], 
                            400, const.DETAILCODE['INVALID_MASK'])
    if card_type_image[0] == "TOO_MANY_MASK" or card_type_image[0] == "TOO_MANY_MASK":
        return handle_error(const.ERROR['INVALID_ID_CARD'], const.ERROR['INVALID_ID_CARD_NUMBER'], 
                            400, const.DETAILCODE['INVALID_ID_CARD_NUMBER'])
    if card_type_image[0] == "MISSING_CORNER" or card_type_image[1] == "MISSING_CORNER":
        return handle_error(const.ERROR['INVALID_ID_CARD'], const.ERROR['INVALID_MISSING_CORNER'], 
                            400, const.DETAILCODE['INVALID_MISSING_CORNER'])
    if card_type_image[0] != card_type_image[1]:
        return handle_error(const.ERROR['INVALID_ID_CARD'], const.ERROR['INVALID_DIFFERENT_TYPE'],
                             400, const.DETAILCODE['INVALID_DIFFERENT_TYPE'])
    if card_side_predict[0] == card_side_predict[1]:
        return handle_error(const.ERROR['INVALID_ID_CARD'], const.ERROR['INVALID_SAME_SIDE'],
                             400, const.DETAILCODE['INVALID_SAME_SIDE'])
    if ratio_area[0] < 0.15 or ratio_area[1] < 0.15:
        return handle_error(const.ERROR['INVALID_ID_CARD'], const.ERROR['INVALID_SIZE'],
                             400, const.DETAILCODE['INVALID_SIZE'])
    
    # detect color of image
    log_obj.info("DETECT COLOR")
    if card_side_predict[0] == "front":
        check_gray_image1 = detect_color(image_crop_front, card_side_predict[0])
        check_gray_image2 = detect_color(image_crop_back, card_side_predict[1])
    else:
        check_gray_image1 = detect_color(image_crop_front, card_side_predict[1])
        check_gray_image2 = detect_color(image_crop_back, card_side_predict[0])
    # check blur
    log_obj.info("CHECK QUALITY IMAGE CARD")
    check_blur_image1 = check_blur(image_crop_front)
    check_blur_image2 = check_blur(image_crop_back)
    if check_blur_image1 or check_blur_image2:
        return handle_error(const.ERROR['INVALID_ID_CARD'], const.ERROR['WRONG_QUALITY'], 
                            400, const.DETAILCODE['INVALID_WRONG_QUALITY'])
    if check_gray_image1 == "gray" or check_gray_image2 == "gray":
        return handle_error(const.ERROR['INVALID_ID_CARD'], const.ERROR['INVALID_WRONG_QUALITY_GRAY'], 
                            400, const.DETAILCODE['INVALID_WRONG_QUALITY_GRAY'])
    

    log_obj.info("DETECT & RECOGNIZED TEXT")
    d = time.time()
    result_ocr = detect_recognize_face_text(image_crop_front, image_crop_back, card_side_predict, card_type_image, path_folder, log_obj)
    print("Time Detect & Recogization Text", time.time()-d)
    log_obj.info("Time Detect & Recogization Text:" + " " + str(time.time()-d))
    return result_ocr


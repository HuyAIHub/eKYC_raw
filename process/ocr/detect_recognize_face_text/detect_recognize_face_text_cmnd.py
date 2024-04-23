import cv2
from utils_ocr.utils import convert_np_to_pillow
from process.ocr.detect_recognize_face_text.utils_text import fix_surname, fix_datetime, fix_name_address_2
from utils_ocr.utils import sort_words


from config_app.config import get_config
config_app, config_model = get_config()

def detect_recognize_face_text_cmnd(image_crop_front, image_crop_back, card_side_predict, model_detect_text, model_recognition):
    image_front = image_crop_front.copy()
    image_back = image_crop_back.copy()
    confidence_list = {}
    result = {}
    result["identCardType"] = "CHỨNG MINH NHÂN DÂN"
    result['identCardNumber'] = ""
    result['identCardName'] = ""
    result['identCardBirthDate'] = ""
    result['identCardNation'] = "Việt Nam"
    result['identCardEthnic'] = ""
    result['identCardGender'] = ""
    result['identCardCountry'] = ""
    result["identCardCountryCity"] = ""
    result["identCardCountryDistrict"] = ""
    result["identCardCountryWards"] = ""
    result["identCardCountryStreet"] = ""
    result['identCardAdrResidence'] = ""
    result["identCardAdrResidenceCity"] = ""
    result["identCardAdrResidenceDistrict"] = ""
    result["identCardAdrResidenceWards"] = ""
    result["identCardAdrResidenceStreet"] = ""
    result['identCardIssueDate'] = ""
    result["identCardExpireDate"] = ""
    result["identCardIssuePlace"] = ""

    detect_text = model_detect_text.predict(source=[image_front, image_back], save=False, 
                                        save_txt=False, conf=config_app['parameter']['conf_detect_text'])
    detect_text_front = detect_text[0]
    detect_text_back = detect_text[1]
    im_show_front, im_show_back = detect_text_front.plot(), detect_text_back.plot()
    box_text_fronts = detect_text_front.boxes.data.numpy()
    box_text_backs = detect_text_back.boxes.data.numpy()
    box_text_fronts = sort_words(box_text_fronts)

    # =========== Xử lý mặt trước CMND ========================
    num_type_face = config_app['class_names']["Anh"]
    for i, box_text_front in enumerate(box_text_fronts):
        if int(num_type_face) == int(box_text_front[5]):
            img_face = image_crop_front[int(box_text_front[1])-2:int(box_text_front[3]+2), int(box_text_front[0])-2:int(box_text_front[2])+2]
            break

    type_texts = ["Maso", "Hoten", "Namsinh", "Quequan", "Noithuongtru"]
    result_text, result_confidences = [], []
    for i, type_text in enumerate(type_texts):
        num_type_text = config_app['class_names'][type_text]
        img_box_text, result_confidence = [], []
        for j, box_text_front in enumerate(box_text_fronts):
            if int(num_type_text) == int(box_text_front[5]):
                img_box = image_crop_front[int(box_text_front[1])-2:int(box_text_front[3]+2), int(box_text_front[0])-2:int(box_text_front[2])+2]
                text, confidence_id = model_recognition.predict(convert_np_to_pillow(img_box),return_prob=True)
                img_box_text.append(text)
                result_confidence.append(confidence_id)
        result_text.append(img_box_text)
        result_confidences.append(result_confidence)

    # Ma So
    id_card_text = result_text[0][0]
    result['identCardNumber'] = id_card_text
    confidence_list['identCardNumber'] = result_confidences[0][0]
    # Ho_Ten
    name_card_text = ''
    for i, text in enumerate(result_text[1]):
        if i == 0:
            name_card_text = text
        else:
            name_card_text = name_card_text + " " + text
    name_card_text = fix_surname(name_card_text)
    result['identCardName'] = name_card_text
    confidence_list['identCardName'] = result_confidences[1][0]
    # Nam sinh
    birth_card_text = result_text[2][0]
    birth_card_text = fix_datetime(birth_card_text)
    result['identCardBirthDate'] = birth_card_text
    confidence_list['identCardBirthDate'] = result_confidences[2][0]
    # Quequan
    text_nguyenquan = ''
    for i, text in enumerate(result_text[3]):
        if i == 0:
            text_nguyenquan = text
        else:
            text_nguyenquan = text_nguyenquan + " " + text
    text_nguyenquan, components_nguyenquan = fix_name_address_2(text_nguyenquan)
    result['identCardCountry'] = text_nguyenquan
    result['identCardCountryCity'] = components_nguyenquan[0]
    result['identCardCountryDistrict'] = components_nguyenquan[1]
    result['identCardCountryWards'] = components_nguyenquan[2]
    result['identCardCountryStreet'] = components_nguyenquan[3]
    confidence_list['identCardCountry'] = result_confidences[3][0]
    # Thuong tru
    text_dkhk = ''
    for i, text in enumerate(result_text[4]):
        if i == 0:
            text_dkhk = text
        else:
            text_dkhk = text_dkhk + " " + text
    # text_dkhk = text_dkhk.replace("Thị Trấn,", "Thị Trấn")
    text_dkhk, components_dkhk = fix_name_address_2(text_dkhk)
    text_dkhk = text_dkhk.replace(" Thị Trấn,", ", Thị Trấn")
    text_dkhk = text_dkhk.replace("Âp", "Ấp")
    text_dkhk = text_dkhk.replace("Nội Nội", "Nội")
    text_dkhk = text_dkhk.replace("Tphồ", "TP.Hồ")
    text_dkhk = text_dkhk.replace("Hồ Chí Minh", "TP.Hồ Chí Minh")
    text_dkhk = text_dkhk.replace("Tdp", "TDP")
    text_dkhk = text_dkhk.replace("Qt, Hà Đông", "Q.Hà Đông")

    result['identCardAdrResidence'] = text_dkhk
    result['identCardAdrResidenceCity'] = components_dkhk[0]
    result['identCardAdrResidenceDistrict'] = components_dkhk[1]
    result['identCardAdrResidenceWards'] = components_dkhk[2]
    result['identCardAdrResidenceStreet'] = components_dkhk[3].replace("Thị Trấn", "")
    confidence_list['identCardAdrResidence'] = result_confidences[4][0]

    # =========== Xử lý cho CMND mặt sau ========================
    # type_texts = ["NhanDang", "NgayCap", "TroTrai", "TroPhai", "SoDinhDanh"]
    type_texts = ["NgayCap", "dantoc"]
    result_text, result_confidences = [], []
    for i, type_text in enumerate(type_texts):
        num_type_text = config_app['class_names'][type_text]
        img_box_text, result_confidence = [], []
        for j, box_text_back in enumerate(box_text_backs):
            if int(num_type_text) == int(box_text_back[5]):
                img_box = image_crop_back[int(box_text_back[1])-2:int(box_text_back[3]+2), int(box_text_back[0])-2:int(box_text_back[2])+2]
                text, confidence_id = model_recognition.predict(convert_np_to_pillow(img_box),return_prob=True)
                img_box_text.append(text)
                result_confidence.append(confidence_id)
        result_text.append(img_box_text)
        result_confidences.append(result_confidence)
    # Ngay cap
    IssueDate_card_text = result_text[0][0]
    IssueDate_card_text = fix_datetime(IssueDate_card_text)
    result['identCardIssueDate'] = IssueDate_card_text
    confidence_list['identCardIssueDate'] = result_confidences[0][0]
    # Dan toc
    id_card_ethnic = result_text[1][0]
    result['identCardEthnic'] = id_card_ethnic
    confidence_list['identCardEthnic'] = result_confidences[1][0]

    return result, im_show_front, im_show_back, img_face, confidence_list
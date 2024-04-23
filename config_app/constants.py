import os

FJoin = os.path.join

# so luong classes dung de xay dung nen mo hinh SSD.
NUM_CLASSES = 21  # Don't change this.

# so luong "anh gia" tao ra tu anh goc (chua tinh anh goc).
NUMBER_AUGMENTATIONS = 1

# cac phuong thuc tao "anh gia" (khong lam thay doi noi dung cua anh goc).
AUGMENTATION_METHODS = ['color-1', 'color-2', 'contrast-1', 'contrast-2', 'brightness-1', 'brightness-2', 'sharpness-1', 'sharpness-2']
# AUGMENTATION_METHODS = ['color', 'contrast', 'brightness', 'sharpness']

# dinh nghia step cho training
DETECTION_SIDE_ROTATION = 'side-rotation'
DETECTION_FACE_TEXT = 'face-text'

# cac loai giay to
IDENTITY_CARD_OLD = "cmnd-cu"
IDENTITY_CARD_2014 = "cmnd-2014"
IDENTITY_CARD_2016 = "cccd-2016"

# so luong chu so trong tung loai CMND
ID_NUMBER_QUANTITY = {
    IDENTITY_CARD_OLD: 9,
    IDENTITY_CARD_2014: 12,
    IDENTITY_CARD_2016: 12
}

# duong dan mac dinh den file cau hinh
CONFIG_PATH = 'config_ocr/config.yml'

# duong dan mac dinh den input data
DATA_SET_SIDE_ROTATION_PATH = "dataset"
DATA_SET_FACE_TEXT_PATH = "labeled_data_step_02"

# cac che do cua ung dung
APP_MODE_FILTER_INPUT_DATA = 'filter-input-data'  # tien xu li (convert file PDF, loc anh kem chat luong) truoc khi gan nhan bang tay
APP_MODE_HANDLE_HYPER_AI = 'handle-hyper-ai'  # lay ket qua gan nhan tren HyperAI web app, roi chuyen thanh file *.xml
APP_MODE_SIDE_ROTATE = 'side-rotate'  # xac dinh mat/chieu va xoay lai anh, dung file gan nhan bang tay - phuc vu cho viec training tiep step sau
APP_MODE_GENERATE_DATASET = 'generate-dataset'  # chuyen doi du lieu xml da dc gan nhan Buoc 1/Buoc 2 sang dinh dang dau vao cua cac mo hinh
APP_MODE_CREATE_SAMPLE_BGRD = 'create-sample-bgrd'  # tao them du lieu gia cho Buoc 1 bang cach chen anh goc vao trong anh background khac
APP_MODE_CREATE_SAMPLE_FLIP = 'create-sample-flip'  # tao them du lieu gia cho Buoc 1 bang cach lat anh theo chieu ngang / doc
APP_MODE_CREATE_SAMPLE_MASK = 'create-sample-mask'  # tao them du lieu gia cho Buoc 1 bang phep chieu affin
APP_MODE_GENERATE_PRE_PROCESS = 'generate-pre-process'

# dinh nghia cac loai target
TARGET_STEP01_MASK = 'step01-mask'
TARGET_STEP01_CHECKLEGAL = 'step01-check-legal'
TARGET_STEP02_SSD = 'step02-ssd'
TARGET_STEP02_SSD_EMBLEMSIGN = 'step02-ssd-emblemsign'
TARGET_STEP02_ICDAR = 'step02-icdar'
TARGET_STEP02_TINY = 'step02-tiny-yolo-v3'
TARGET_STEP03_TESSERACT = 'step03-tesseract'
TARGET_STEP03_XCEPTION = 'step03-xception'

# Dinh nghia cac nhan cho step: side-rotation
PICTURE_CORNER = 'corner'
PICTURE_EMBLEM = 'national emblem'
PICTURE_SIGN = 'red sign'
PICTURE_FACE = 'face'
PICTURE_FINGERPRINT = 'fingerprint'
PICTURE_BARCODE = 'barcode'

# Dinh nghia cac nhan cho step: face-text
PICTURE = 'picture'
LABEL = 'label'
CONTENT = 'content'
NUMBER = 'number'
DATETIME = 'datetime'
LABELS_FACE_TEXT = [PICTURE, LABEL, CONTENT]

# Dinh nghia cac gia tri cho ket qua cua step 1: xac dinh mat-chieu
SIDE_FRONT = "front"  # mat truoc
SIDE_BACK = "back"  # mat sau

ROTATION_CORRECT = "chup thang"
ROTATION_LEFT = "xoay trai"
ROTATION_RIGHT = "xoay phai"
ROTATION_REVERSE = "chup nguoc"

# Dinh nghia che do anh
IMG_COLOR = 'color'
IMG_GRAY_BINARY = 'gray_binary'

# dinh nghia che do duyet thu muc file dau vao
EXPLORE_CONVERT = 'convert'  # (tien xu li) chuyen moi file dinh dang PDF thanh 1 folder anh
EXPLORE_FILTER = 'filter'  # (tien xu li) loc anh xam/den trang
EXPLORE_BROWSE = 'browse'  # lay ra danh sach file anh (chi o thu muc hien tai, khong lay trong thu muc con)

# cac duoi mo rong pho bien cua dinh dang anh
IMAGE_EXTENSION_LIST = {
    'bmp', 'gif', 'jpg', 'png', 'jpeg', 'tif', 'tiff'
}


# cac loi cua api
ERROR = {
    'INVALID_ID_CARD': "Giấy tờ không hợp lệ",
    'INVALID_DIFFERENT_TYPE': "Hai ảnh truyền khác loại giấy tờ",
    'INVALID_SAME_SIDE': "Giấy tờ truyền cùng mặt",
    'INVALID_SIZE': "Giấy tờ chụp quá xa",
    'INVALID_RECOGNITION_ID_CARD': "Giấy tờ thiếu dấu hiệu nhân dạng",
    'INVALID_RECOGNITION_ID_CARD2': "Giấy tờ thiếu dấu hiệu nhân dạng - quốc huy",
    'WRONG_QUALITY' : "Giấy tờ không hợp lệ, bị mờ/bóng",
    'INVALID_WRONG_QUALITY_GRAY' : "Giấy tờ không hợp lệ, cần truyền ảnh màu",
    'WRONG_FACE'    : "Ảnh khuôn mặt trên giấy tờ không hợp lệ",
    'INVALID_MASK': "Vui lòng chụp rõ nét, đầy đủ thông tin",
    'INVALID_MISSING_INFO': "Vui lòng chụp rõ nét, đầy đủ thông tin",
    'INVALID_ID_CARD_NUMBER': "Ảnh có nhiều giấy tờ",
    'INVALID_MISSING_CORNER': "Giấy tờ bị che hoặc mất góc",
    'INVALID_MISSING_CODE': "Số định danh bị mờ, hoặc không rõ nét",
}
DETAILCODE = {
    'INVALID_ID_CARD': 405,
    'INVALID_DIFFERENT_TYPE': 406,
    'INVALID_SAME_SIDE': 407,
    'INVALID_SIZE': 408,
    'INVALID_ID_CARD_NUMBER': 409,
    'INVALID_RECOGNITION_ID_CARD':202,
    'INVALID_RECOGNITION_ID_CARD2' : 203,
    'INVALID_MISSING_CORNER':410,
    'INVALID_WRONG_QUALITY':411,
    'INVALID_WRONG_FACE':412,
    'INVALID_MASK':413,
    'INVALID_WRONG_QUALITY_GRAY': 414,
    'INVALID_MISSING_CODE': 415,
}


def pr(base_name, text):
    """ print text with additional info (filename)
    Args:
        base_name: name of caller file
        text: the content need to output
    Returns:
    """
    print("\n{}  {}".format(os.path.basename(base_name).upper(), text))
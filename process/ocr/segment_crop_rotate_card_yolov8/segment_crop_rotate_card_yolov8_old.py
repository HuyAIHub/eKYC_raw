import cv2
import math
import numpy as np
import config_app.constants as const
from process.PretrainedModel import PretrainedModel

from config_app.config import get_config
config_app, config_model = get_config()
models = PretrainedModel(config_model['ocr_model'])
model_segment = models.segmentYolov8
# model_classify = models.classifyCardSideModel

def angle(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle_rad = np.arccos(dot_product)
    return angle_rad


def distance(vector_1, vector_2):
    distance_res = np.linalg.norm(vector_2 - vector_1)
    return distance_res


def perp(a):
    b = np.zeros_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


def intersect(a1, a2, b1, b2):
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom.astype(float)) * db + b1

def create_side_matrix(size):
    """ create 2-D array with the value 1 in the "side" and 0 in "inside"
    Example with (6, 4)-size array: [[1 1 1 1] [1 0 0 1] [1 0 0 1] [1 0 0 1] [1 0 0 1] [1 1 1 1]]
    :param size: (height, width) of array
    :return: the side matrix
    """
    side_matrix = np.zeros(size, dtype=np.int)
    (h, w) = size[:]
    for i in range(w):
        side_matrix[0, i] = 1
        side_matrix[h - 1, i] = 1

    for j in range(h):
        side_matrix[j, 0] = 1
        side_matrix[j, w - 1] = 1

    return side_matrix


def segment_crop_rotate_card_yolov8(image):
    # set
    img = image.copy()
    card_type = "error"
    corners = []
    all_distances = []
    distances = []
    rect = []
    min_rect_vec = []
    edges = []
    start = 0
    new_rect = [0, 0, 0, 0]
    new_rect_vec = []
    angle_threshold = 3
    num_edge = 0

    results = model_segment.predict(source=img, save=False, save_txt=False)  # save predictions as labels
    for result in results:
        boxes = result.boxes
        msk = result.masks
        
        if msk is None:
            return '', '', 'NO_MASK', '', ''
        else:
            msk = msk.data.cpu().numpy()  # Masks object for segmenation masks outputs
            classes = boxes.cls.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
        
        if len(scores) > 0 and scores[0] >= 0.5:
            if classes[0] == 2 or classes[0] == 4:
                card_type = "cmnd-cccd-2016"
            elif classes[0] == 0 or classes[0] == 1:
                card_type = "cccd-chip"
            elif classes[0] == 3 or classes[0] == 5:
                card_type = "cmnd-cu"
        if len(msk) == 0:
            return '', '', 'NO_MASK', '', ''
        if len(msk) > 1:
            return '', '', 'TOO_MANY_MASK', '', '' 
        
        msk = msk[0]
        msk = cv2.resize(msk, (image.shape[1], image.shape[0]))
        msk[msk >= 0.5] = 1
        msk[msk < 0.5] = 0

        side_matrix = create_side_matrix(msk.shape)
        side_binary = np.bitwise_and(side_matrix, msk.astype(np.int))
        count_side_overlap = np.sum(side_binary)

        # Tìm contour cho mask
        contours, hierarchy = cv2.findContours(msk.astype('uint8')*255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Tìm tỷ lệ mask so với ảnh
        ratio_area = np.sum(msk.astype('uint8')) / (img.shape[0]*img.shape[1])
        # Tìm contour lớn nhất
        max_contour = max(contours, key=cv2.contourArea)
        # Tìm convex hull
        convex_hull = cv2.convexHull(max_contour)
        # Tìm hình chữ nhật bao bọc convex nhỏ nhất
        min_rect = cv2.minAreaRect(convex_hull)
        min_rect = np.int0(cv2.boxPoints(min_rect))
        min_width = min(distance(min_rect[0], min_rect[1]), distance(min_rect[1], min_rect[2]))
        for i in range(4):
            min_rect_vec.append(min_rect[i] - min_rect[(i+1) % 4])
        # Loại bỏ các điểm có góc > 170 độ của convex (điểm nằm trên đường thẳng)
        for i, point in enumerate(convex_hull):
            if angle(convex_hull[i - 1][0] - point[0],
                    convex_hull[(i + 1) % len(convex_hull)][0] - point[0]) < angle_threshold:
                corners.append(point)
        # Lọc và Loại bỏ các cạnh vát góc (các cạnh mà tạo 1 góc lớn so với các cạnh của hình chữ nhật bao nhỏ nhất hoặc cả 2 đầu
        # mút của cạnh cách xa các điểm của hình chữ nhật bao nhỏ nhất) và tính độ lớn cho các cạnh thỏa mãn
        for i, point in enumerate(corners):
            all_distances.append(distance(point[0], corners[(i+1) % len(corners)][0]))
            min_angle = np.pi / 2
            min_dis_to_rect = img.shape[0]
            current_vec = point[0] - corners[(i + 1) % len(corners)][0]
            for k in range(4):
                ang = angle(current_vec, min_rect_vec[k])
                if ang < min_angle or (ang > np.pi/2 and ang - np.pi/2 < min_angle):
                    min_angle = ang if ang < np.pi / 2 else ang - np.pi / 2
                dis1 = distance(point[0], min_rect[k])
                dis2 = distance(corners[(i + 1) % len(corners)][0], min_rect[k])
                min_dis_to_rect = dis1 if min_dis_to_rect > dis1 else min_dis_to_rect
                min_dis_to_rect = dis2 if min_dis_to_rect > dis2 else min_dis_to_rect
            if min_angle < np.pi * (10 / 180) or min_dis_to_rect <= min_width / 12:
                edges.append(i)
                distances.append(distance(point[0], corners[(i + 1) % len(corners)][0]))
        # Sắp xếp các cạnh theo độ lớn
        edges = np.array(edges)
        distances = np.array(distances)
        all_distances = np.array(all_distances)
        max_dis_arg = all_distances.argsort()[-1]
        max_args = distances.argsort()
        # Tính 4 cạnh lớn hẳn (lớn hơn gấp 10 lần các cạnh bé)
        for i in range(len(max_args)):
            dis_ratio = distances[max_args[len(max_args) - i - 1]] / all_distances[max_dis_arg]
            if dis_ratio < 1 / 10:
                num_edge = i
                break
        # cv2.imwrite('/root/AI/ocr_v2/data/crop.png', msk*255)
        if num_edge < 4:
            edges = np.array(range(len(corners)))
            max_args = all_distances.argsort()
        max_args = np.sort(max_args[-4:])
        # Tìm 4 giao điểm bởi các cạnh đó
        for i in range(4):
            if edges[max_args[(i+1) % 4]] == max_dis_arg:
                start = i
            point = intersect(corners[edges[max_args[i]]][0],
                            corners[(edges[max_args[i]] + 1) % len(corners)][0],
                            corners[edges[max_args[(i+1) % 4]]][0],
                            corners[(edges[max_args[(i+1) % 4]] + 1) % len(corners)][0])
            rect.append(point.astype(float))
        # Sắp xếp lại các giao điểm cho đúng thứ tự để co giãn đúng (chiều dài, chiều rộng)
        for i in range(4):
            new_rect[(4 - start + i) % 4] = rect[i]
        for i in range(4):
            new_rect_vec.append(new_rect[i] - new_rect[(i + 1) % 4])
            # Tìm cạnh bị vát góc
        for i, point in enumerate(corners):
            min_angle = np.pi / 2
            min_dis_to_rect = img.shape[0]
            current_vec = point[0] - corners[(i + 1) % len(corners)][0]
            for k in range(4):
                ang = angle(current_vec, new_rect_vec[k])
                if ang < min_angle or (ang > np.pi / 2 and ang - np.pi / 2 < min_angle):
                    min_angle = ang if ang < np.pi / 2 else ang - np.pi / 2
                dis1 = distance(point[0], new_rect[k])
                dis2 = distance(corners[(i + 1) % len(corners)][0], new_rect[k])
                min_dis_to_rect = dis1 if min_dis_to_rect > dis1 else min_dis_to_rect
                min_dis_to_rect = dis2 if min_dis_to_rect > dis2 else min_dis_to_rect
            if min_angle < np.pi * (10 / 180) or min_dis_to_rect <= min_width / 15:
                continue
            else:
                return '', '', 'MISSING_CORNER', '', ''
        # Tạo ma trận biến đổi tuyến tính theo 4 giao điểm và 4 điểm cố định kích thước 1026x640
        dst = np.array([[0, 0], [1026 - 1, 0], [1026 - 1, 640 - 1], [0, 640 - 1]], dtype="float32")
        m = cv2.getPerspectiveTransform(np.array(new_rect, dtype="float32"), dst)
        warped = cv2.warpPerspective(image, m, (1026, 640))
        warped = warped.astype('uint8')

        # Dự đoán chiều của thẻ và xoay
        # if classes[0] in [0,2,3,4,5]:
        dst_copy = warped.copy()
        dst_copy = cv2.resize(dst_copy, (320, 320))
        dst_copy = np.expand_dims(dst_copy, 0)
        y_pred = model_classify.predict(dst_copy)
        y_pred = np.argmax(y_pred, axis=1)
        y_pred = y_pred.astype('int')
        if y_pred in [1,3,5,7,9,11]:
            warped = warped[::-1, ::-1]
        if y_pred in [2,3,6,7,10,11]:
            return warped, const.SIDE_FRONT, card_type, ratio_area, count_side_overlap
        if y_pred in [0,1,4,5,8,9]:
            return warped, const.SIDE_BACK, card_type, ratio_area, count_side_overlap
        if y_pred == 12:
            card_type = "NO_MASK"
            return warped, const.SIDE_BACK, card_type, ratio_area, count_side_overlap
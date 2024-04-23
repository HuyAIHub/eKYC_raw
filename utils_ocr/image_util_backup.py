#
#  Copyright by Hyperlogy Corporation, 2020
#  Smart eKYC project.
#
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.spatial.distance as dis

from PIL import Image, ImageEnhance, ImageOps, ImageDraw
from ocr_v2.config import constants as const
from ocr_v2.config.constants import FJoin
from ocr_v2.model.detect_text.evaluate import get_annotation
from ocr_v2.utils.dir_util import dir_explore

MAXIMUM_ALLOWED_RESOLUTION = 89478485  # maximum allowed pixels


class ImageVisualizer(object):
    """ Class for visualizing image
    Attributes:
        idx_to_name: list to convert integer to string ocr_v2.label
        class_colors: colors for drawing boxes and labels
        save_dir: directory to store images
    """

    def __init__(self, idx_to_name, class_colors=None, save_dir=None):
        self.idx_to_name = idx_to_name
        if class_colors is None or len(class_colors) != len(self.idx_to_name):
            self.class_colors = [[0, 255, 0]] * len(self.idx_to_name)
        else:
            self.class_colors = class_colors

        if save_dir is None:
            self.save_dir = '../'
        else:
            self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)

    def save_image(self, img, boxes, labels, scores, name):
        """ Method to draw boxes and labels
            then save to dir
        Args:
            img: numpy array (width, height, 3)
            boxes: numpy array (num_boxes, 4)
            labels: numpy array (num_labels)
            scores: numpy array (num_scores)
            name: name of image to be saved
        """
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        save_path = os.path.join(self.save_dir, name)

        for i, box in enumerate(boxes):
            idx = labels[i] - 1
            cls_name = self.idx_to_name[idx]
            score = round(scores[i], 2)  # only show 2 digit numbers
            ax.add_patch(patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0], box[3] - box[1],
                linewidth=1, edgecolor=(0., 1., 0.),
                facecolor="none"))
            visual_text = cls_name[0]
            plt.text(
                box[0],
                box[1],
                s=visual_text,
                color="red",
                verticalalignment="top",
                bbox={"color": (0., 1., 0.), "pad": 0},
            )
        plt.axis("off")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0)
        plt.close('all')


def remove_alpha(img):
    """ Function to convert 4-channel image (RGBA) to 3-channel image (RGB)

    Args:
        img: the original PIL Image

    Returns:
        convert_img: the 3-channel image (RGB)
    """
    img_depth = len(img.mode)
    # print("remove_alpha() The image has {} channels".format(img_depth))

    convert_img = img
    if img_depth == 4:
        convert_img = img.convert('RGB')
    return convert_img


def filter_high_resolution(input_folder):
    """ filter if image has large resolution
    Args:
        input_folder: input image folder
    Returns:
        images after resize
    """
    file_list = dir_explore(input_folder, const.EXPLORE_BROWSE)
    for file in file_list:
        image = Image.open(FJoin(input_folder, file[0] + file[1]))
        (w, h) = image.size
        res = w * h
        if res >= MAXIMUM_ALLOWED_RESOLUTION:
            print("\n Detect image {} with high resolution as {}".format(file[0], res))
            #ratio = np.sqrt(res / MAXIMUM_ALLOWED_RESOLUTION)
            #new_w = int(w / (2*ratio))
            #new_h = int(h / (2*ratio))
            #new_img = image.resize((new_w, new_h))
            #new_img.save(FJoin(input_folder, file[0] + '_resize' + file[1]))


def rotate(img, rotation):
    """ Function to rotate the image with provided rotation
    Args:
        img: original image
        rotation: the rotation which is define in constant.ROTATION_xyz
    Returns:
        the image after do rotation
    """
    angle = 0  # goc xoay, mac dinh '0' la anh ban dau
    # angle: Goc (tinh theo do) la cung chieu kim dong ho.
    if rotation == const.ROTATION_LEFT:
        angle = Image.ROTATE_270
    elif rotation == const.ROTATION_REVERSE:
        angle = Image.ROTATE_180
    elif rotation == const.ROTATION_RIGHT:
        angle = Image.ROTATE_90
    print("angle = ", angle)
    # chu y: goi ham transpose(0) thi tham so 0 nghia la FLIP_LEFT_RIGHT
    rotated_img = img if rotation == const.ROTATION_CORRECT else img.transpose(angle)
    return rotated_img


def augmentation(filename, img, aug_method):
    """ Function to do augmentation with provided image in 3 ways : 'color adjust', 'noise addition', 'contrast adjust'

    Args:
        filename: name of provided image
        img: the original PIL Image
        aug_method: augmentation method (xyz-1 will decrease, xyz-2 will increase the property of original img)

    Returns:
        new_name: the name of augmentation image
        aug_img: image after do augmentation
    """
    new_name = filename + "_" + aug_method
    # print("|| HYPERLOGY augmentation() new_name = {}, type(img) = {}, aug_method = {}".format(new_name, type(img), aug_method))
    aug_img = None
    if aug_method == const.AUGMENTATION_METHODS[0]:
        aug_img = ImageEnhance.Color(img).enhance(factor=0.8)
    elif aug_method == const.AUGMENTATION_METHODS[1]:
        aug_img = ImageEnhance.Color(img).enhance(factor=1.5)

    elif aug_method == const.AUGMENTATION_METHODS[2]:
        aug_img = ImageEnhance.Contrast(img).enhance(factor=0.8)
    elif aug_method == const.AUGMENTATION_METHODS[3]:
        aug_img = ImageEnhance.Contrast(img).enhance(factor=1.5)

    elif aug_method == const.AUGMENTATION_METHODS[4]:
        aug_img = ImageEnhance.Brightness(img).enhance(factor=0.8)
    elif aug_method == const.AUGMENTATION_METHODS[5]:
        aug_img = ImageEnhance.Brightness(img).enhance(factor=1.5)

    elif aug_method == const.AUGMENTATION_METHODS[6]:
        aug_img = ImageEnhance.Sharpness(img).enhance(factor=0.8)
    elif aug_method == const.AUGMENTATION_METHODS[7]:
        aug_img = ImageEnhance.Sharpness(img).enhance(factor=1.5)

    # print("|| HYPERLOGY augmentation() type(aug_img) = {}".format(type(aug_img)))
    return new_name, aug_img


# gia tri hieu chinh de lay ra cac diem goc "hop li" trong moi goc cua phan CMND trong anh chup
ADJUST_VALUE = 5  # thu nghiem voi 5 pixels cho thay ket qua co cai thien.

# Dinh nghia chieu hieu chinh cho 4 goc
DIRECTION_INCREASE = 'increase'
DIRECTION_DECREASE = 'decrease'


def adjust_corner(x, y, direction, adj_val=ADJUST_VALUE):
    """ adjust the corner points such that approximate quite good the real bounding of ID card.
    Args:
        x: 1st corner
        y: 2nd corner
        direction: increment or decrement
        adj_val: the adjustment value
    Returns:
        new corner points after do adjustment
    """
    new_x, new_y = (x, y)
    if direction == DIRECTION_INCREASE:
        if new_x > new_y:
            new_x = new_x + adj_val
        elif new_y > new_x:
            new_y = new_y + adj_val
    elif direction == DIRECTION_DECREASE:
        if new_x < new_y:
            new_x = new_x - adj_val
        elif new_y < new_x:
            new_y = new_y - adj_val
    return new_x, new_y


def choose_corner_points(corner_boxes):
    """ choose the corner points from corner boxes
    Args:
        corner_boxes: the boxes of corners
    Returns:
        the proper points within corner boxes
    """
    sort_corners_index = np.argsort([box[0] + box[1] for box in corner_boxes])  # sap xep cac box theo 'xmin + ymin'
    tl_box = corner_boxes[sort_corners_index][0]  # top-left box co gia tri 'xmin + ymin' nho nhat
    br_box = corner_boxes[sort_corners_index][3]  # bottom-right box co gia tri 'xmin + ymin' lon nhat
    tr_box = corner_boxes[sort_corners_index][1]
    bl_box = corner_boxes[sort_corners_index][2]
    if tr_box[2] / tr_box[3] < bl_box[2] / bl_box[3]:  # so sanh cac gia tri 'xmax/ymax'
        # top-right box co ti le 'xmax/ymax' lon hon bottom-left box.
        tr_box = corner_boxes[sort_corners_index][2]
        bl_box = corner_boxes[sort_corners_index][1]
    # xac dinh 4 diem thich hop o trong 4 goc
    tl_point = [tl_box[0], tl_box[1]]  # xmin, ymin
    tr_point = [tr_box[2], tr_box[1]]  # xmax, ymin
    bl_point = [bl_box[0], bl_box[3]]  # xmin, ymax
    br_point = [br_box[2], br_box[3]]  # xmax, ymax
    print("tl_point = {}, tr_point = {}, bl_point = {}, br_point = {}".format(tl_point, tr_point, bl_point, br_point))
    # hieu chinh 4 diem goc dua vao tuong quan hoanh do (x), tung do (y) giua chung
    tl_point[0], bl_point[0] = adjust_corner(tl_point[0], bl_point[0], DIRECTION_INCREASE)
    tr_point[0], br_point[0] = adjust_corner(tr_point[0], br_point[0], DIRECTION_DECREASE)
    tl_point[1], tr_point[1] = adjust_corner(tl_point[1], tr_point[1], DIRECTION_INCREASE)
    bl_point[1], br_point[1] = adjust_corner(bl_point[1], br_point[1], DIRECTION_DECREASE)
    print("tl_point = {}, tr_point = {}, bl_point = {}, br_point = {}".format(tl_point, tr_point, bl_point, br_point))
    return tl_point, tr_point, bl_point, br_point


def choose_perspective_size(corner_points):
    """ choose the proper size for new images after applying perspective projection
    Args:
        corner_points: the four points corresponding to four corners
    Returns:
        the proper size (i.e, width and height) of new image
    """
    tl_point, tr_point, bl_point, br_point = (corner_points[:])
    w_up = dis.euclidean(tl_point, tr_point)
    w_down = dis.euclidean(bl_point, br_point)
    h_left = dis.euclidean(tl_point, bl_point)
    h_right = dis.euclidean(tr_point, br_point)
    # print("w_up = {}, w_down = {}, h_left = {}, h_right = {}".format(w_up, w_down, h_left, h_right))
    proper_width = int((w_up + w_down) / 2)
    proper_height = int((h_left + h_right) / 2)
    print("proper_width = {}, proper_height = {}".format(proper_width, proper_height))
    return proper_width, proper_height


def create_sample_from_background(input_folder, bgrd_folder, additional_quantity):
    """ create more sample images from original images, using predefined background images
    Args:
        input_folder: folder contains original images
        bgrd_folder: folder contains background images
        additional_quantity: the quantity of additional images for each original image
    Returns:
        - the image after added into different random-choice background
        - the corresponding mask image to each output image (to use as input data for SegNet model - Object Detection step)
    """
    image_folder = FJoin(input_folder, 'img_idcard')  # thu muc chua anh goc
    os.makedirs(image_folder, exist_ok=True)
    mask_folder = FJoin(input_folder, 'mask_idcard')  # thu muc chua anh den trang tuong ung
    os.makedirs(mask_folder, exist_ok=True)
    bgrd_list = dir_explore(bgrd_folder, const.EXPLORE_BROWSE)
    file_list = dir_explore(input_folder, const.EXPLORE_BROWSE)
    for bgrd in bgrd_list:
        bgrd_image = Image.open(FJoin(bgrd_folder, bgrd[0] + bgrd[1]))
        (W, H) = bgrd_image.size
        print("bgrd_image.size = {}".format((W, H)))
        for i in enumerate(range(additional_quantity)):
            rand_index = np.random.choice(range(len(file_list)))
            rand_file = file_list[rand_index]
            print("rand_index = {}, rand_file = {}".format(rand_index, rand_file))
            rand_image = Image.open(FJoin(input_folder, rand_file[0] + rand_file[1]))
            (w, h) = rand_image.size
            print("rand_image.size = {}".format((w, h)))
            rand_ratio = np.random.choice(range(2, 8))
            ratio = np.minimum(W/w, H/h) * rand_ratio / 10.0
            print("rand_ratio = {}, ratio = {}".format(rand_ratio, ratio))
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            """
            tao ra du lieu anh moi bang cach resize anh goc va dua vao trong anh background
            """
            new_image = rand_image.resize((new_w, new_h))
            rand_w = np.random.choice(range(8-rand_ratio)) + 1.0
            rand_h = np.random.choice(range(8-rand_ratio)) + 1.0
            padding_w = int(W * rand_w / 10.0)
            padding_h = int(H * rand_h / 10.0)
            bgrd_img = bgrd_image.copy()
            bgrd_img.paste(new_image, (padding_w, padding_h))
            """
            tao ra du lieu anh 'mask' den trang theo cach tuong tu
            """
            background = Image.new('RGBA', (W, H), (0, 0, 0, 255))
            mask = Image.new('RGBA', (new_w, new_h), (255, 255, 255, 255))
            background.paste(mask, (padding_w, padding_h))
            if W > H:
                bgrd_img.save(fp=FJoin(image_folder, rand_file[0] + "_" + str(i) + rand_file[1]))
                background.save(fp=FJoin(mask_folder, "mask_" + rand_file[0] + "_" + str(i) + ".png"))
            else:
                rotated_img = rotate(bgrd_img, const.ROTATION_RIGHT)
                rotated_img.save(fp=FJoin(image_folder, rand_file[0] + "_" + str(i) + rand_file[1]))
                rotated_bgrd = rotate(background, const.ROTATION_RIGHT)
                rotated_bgrd.save(fp=FJoin(mask_folder, "mask_" + rand_file[0] + "_" + str(i) + ".png"))


def create_sample_mask(input_folder):
    """
    create sample for input of SegNet model
    :param input_folder:
    :return:
    """
    res_folder = FJoin(input_folder, 'results')
    os.makedirs(res_folder, exist_ok=True)
    file_list = dir_explore(input_folder, const.EXPLORE_BROWSE)
    for file in file_list:
        anno_path = FJoin(input_folder, 'Annotation', '{}.xml').format(file[0])
        if os.path.exists(anno_path):
            labeled_objs = get_annotation(anno_path)
            print("\nHYPERLOGY - anno_path = {}\nlabeled_objects = {}".format(anno_path, labeled_objs))
            corner_objects = [obj for obj in labeled_objs if obj['name'] == const.PICTURE_CORNER]
            corner_bboxes = np.array([x['bbox'] for x in corner_objects])  # [xmin, ymin, xmax, ymax]
            print("corner_boxes = {}".format(corner_bboxes))
            corner_number = len(corner_bboxes)
            print("corner_number = {}".format(corner_number))
            if corner_number == 4:
                # Bat dau xac dinh 4 diem input, 4 diem output de thuc hien phep chieu
                dst_pts = choose_corner_points(corner_bboxes)

                new_width, new_height = choose_perspective_size(dst_pts)
                print("new_width = {}, new_height = {}".format(new_width, new_height))

                src_pts = [(0, 0), (new_width, 0), (0, new_height), (new_width, new_height)]
                coeffs = find_coefficients(src_pts, dst_pts)  # coeffs = [a,b,c,d,e,f,g,h]

                orig_img = Image.open(FJoin(input_folder, file[0] + file[1]))
                print("orig_img.size = {}".format(orig_img.size))
                trans_img = orig_img.transform((new_width, new_height), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
                trans_img.save(fp=FJoin(res_folder, file[0] + "_trans" + file[1]))
                """
                tao dau vao cho qua trinh train bang segnet tren Google Colab
                """
                image_folder = FJoin(res_folder, 'img_idcard')  # thu muc chua anh goc
                os.makedirs(image_folder, exist_ok=True)
                mask_folder = FJoin(res_folder, 'mask_idcard')  # thu muc chua anh den trang tuong ung
                os.makedirs(mask_folder, exist_ok=True)
                # tao ra anh den trang lam 'mask'
                (width, height) = orig_img.size
                background = Image.new('RGBA', (width, height), (0, 0, 0, 255))
                mask = Image.new('RGBA', (width, height))
                pdraw = ImageDraw.Draw(mask)
                tl_point, tr_point, bl_point, br_point = dst_pts[:]
                pdraw.polygon([tuple(tl_point), tuple(tr_point), tuple(br_point), tuple(bl_point)],
                              fill=(255, 255, 255, 255), outline=(255, 255, 255, 255))
                background.alpha_composite(mask)
                # bgrd1 = orig_img.convert('RGBA')
                # bgrd1.alpha_composite(mask)
                if width > height:
                    orig_img.save(fp=FJoin(image_folder, file[0] + file[1]))
                    background.save(fp=FJoin(mask_folder, "mask_" + file[0] + ".png"))  # bgrd1
                else:
                    rotated_img = rotate(orig_img, const.ROTATION_RIGHT)
                    rotated_img.save(fp=FJoin(image_folder, file[0] + file[1]))
                    rotated_bgrd = rotate(background, const.ROTATION_RIGHT)  # bgrd1
                    rotated_bgrd.save(fp=FJoin(mask_folder, "mask_" + file[0] + ".png"))


def create_sample_by_flipping(input_folder):
    """ create more sample images from original images, by flipping horizontally / vertically
    Args:
        input_folder: folder contains original images
    Returns:
        - the image after flipping
    """
    image_list = dir_explore(input_folder, const.EXPLORE_BROWSE)  # thu muc chua anh goc
    output_flip_folder = FJoin(input_folder, "flip")
    output_mirror_folder = FJoin(input_folder, "mirror")
    os.makedirs(output_flip_folder, exist_ok=True)
    os.makedirs(output_mirror_folder, exist_ok=True)
    for file in image_list:
        img = Image.open(FJoin(input_folder, file[0] + file[1]))
        if img is not None:
            hor_img = ImageOps.mirror(img)
            hor_img.save(fp=FJoin(output_mirror_folder, file[0] + "_hor_img" + file[1]))
            ver_img = ImageOps.flip(img)
            ver_img.save(fp=FJoin(output_flip_folder, file[0] + "_ver_img" + file[1]))


def find_coefficients(src, dst):
    """ This method calculate for perspective transform 8-tuple (a,b,c,d,e,f,g,h).
    Use to transform an image:
    X = (a x + b y + c)/(g x + h y + 1)
    Y = (d x + e y + f)/(g x + h y + 1)
    Args:
        src: the four points of source images [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        dst: the four points of destination images [(X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4)]
    Returns:
        the 8-tuple coefficients (a,b,c,d,e,f,g,h) of perspective transform
    """
    a_matrix = []
    if len(src) == 4 and len(dst) == 4:
        for (x, y), (X, Y) in zip(src, dst):
            a_matrix.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
            a_matrix.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
        a_matrix = np.array(a_matrix)
    else:
        print("HYPERLOGY - Do not have enough points to transform image.")
    b_matrix = np.array(dst).reshape(8)
    # print("\na_matrix = {} \nb_matrix = {}".format(a_matrix, b_matrix))
    return np.linalg.solve(a_matrix, b_matrix)


def find_reverse_point(coeffs, point):
    """ find the source point which will become provided point through "perspective transform"
    Args:
        coeffs: the 8-tuple coefficients (a,b,c,d,e,f,g,h)
        point: the destination point (X, Y)
    Returns:
        the source point (x, y) such that below equations will adapt:
        X = (a x + b y + c)/(g x + h y + 1)
        Y = (d x + e y + f)/(g x + h y + 1)
    """
    (a, b, c, d, e, f, g, h) = (coeffs[:])  # dat lai ten cac he so cho de xem
    (X, Y) = (point[:])
    a_matrix = np.array([[a-X*g, b-X*h], [d-Y*g, e-Y*h]])
    b_matrix = np.array([X-c, Y-f])
    return np.linalg.solve(a_matrix, b_matrix)


def transform_point(coeffs, point):
    """ transform the point through "perspective transform" using the 8-tuple 'coeffs' as coefficients
    Args:
        coeffs: the 8-tuple coefficients (a,b,c,d,e,f,g,h)
        point: the source point (x, y)
    Returns:
        the destination point (X, Y)
    """
    (a, b, c, d, e, f, g, h) = (coeffs[:])  # dat lai ten cac he so cho de xem
    (x, y) = (point[:])
    x_trans = (a*x + b*y + c) / (g*x + h*y + 1)
    y_trans = (d*x + e*y + f) / (g*x + h*y + 1)
    return x_trans, y_trans


def check_color_image(img):
    """ check if an input image is color image or gray / binary image by two steps:
        - check bit depth: color image has at least 24 bit
        - check different between color channels: gray/binary image has this different nearly 0
    Args:
        img: the input image
    Returns:
        the type of image: 'color' / 'gray' / 'binary'
    """
    penalty_threshold = 50  # nguong de phan loai cac anh cong chung (co nhieu vung mau do)
    filter_threshold = 5.0  # nguong chap nhan / loai bo anh; tim ra bang thuc nghiem
    img_depth = len(img.mode)  # image depth, in bytes
    # print("img_depth = ", img_depth)
    if img_depth < 3:
        result = const.IMG_GRAY_BINARY
    else:
        pixels = np.asarray(img)
        #print("width = {}, height = {}".format(width, height))
        #print("pixels = ", pixels)
        diff_channel = 0.0  # do sai khac trung binh cua cac kenh mau trong toan bo anh

        def diff(pixel):  # tinh do sai khac trung binh cua cac kenh mau o moi diem anh
            diff_gb = np.abs(int(pixel[1]) - int(pixel[2]))
            diff_rg = np.abs(int(pixel[0]) - int(pixel[1]))
            diff_rb = np.abs(int(pixel[0]) - int(pixel[2]))
            if int(pixel[0]) > 255 - penalty_threshold and int(pixel[1]) + int(pixel[2]) < 2*penalty_threshold:
                diffs = diff_gb - (diff_rg + diff_rb)*penalty_threshold
            else:
                diffs = diff_gb + diff_rg + diff_rb
            return diffs

        width, height = img.size
        for h in range(height):
            for w in range(width):
                diff_channel += diff(pixels[h, w])
        diff_channel = diff_channel / (height*width)
        print("diff_channel = ", diff_channel)
        result = const.IMG_GRAY_BINARY if diff_channel <= filter_threshold else const.IMG_COLOR
    # print("result = ", result)
    return result

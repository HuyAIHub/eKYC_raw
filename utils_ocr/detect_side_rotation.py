#
#  Copyright by Hyperlogy Corporation, 2020
#  Smart eKYC project.
#
"""
Buoc 1: Phan biet mat truoc/sau va xac dinh chieu cua buc anh
"""
import numpy as np
import os
import ocr_v2.config.constants as const

from PIL import Image, ImageDraw
from ocr_v2.config.constants import FJoin
from ocr_v2.utils.image_util import find_coefficients, rotate, find_reverse_point, choose_perspective_size, choose_corner_points


def detect_rotation(side, size, es_point):
    """ Detect the rotation of image based on the relative position of emblem_sign point within the image
    Args:
        side: the image is front side or back side
        size: width and height of image
        es_point: emblem_sign point within the image
    Returns:
        ROTATION_CORRECT / ROTATION_LEFT / ROTATION_RIGHT / ROTATION_REVERSE
    """
    (width, height) = size[:]
    if side == const.SIDE_FRONT:  # mat truoc
        if es_point[0] < width / 2:
            if es_point[1] < height / 2:  # quoc huy nam o goc tren, ben trai
                rotation = const.ROTATION_CORRECT
            else:  # quoc huy nam o goc duoi, ben trai
                rotation = const.ROTATION_LEFT
        else:
            if es_point[1] < height / 2:  # quoc huy nam o goc tren, ben phai
                rotation = const.ROTATION_RIGHT
            else:  # quoc huy nam o goc duoi, ben phai
                rotation = const.ROTATION_REVERSE
    else:  # mat sau
        if width > height:  # anh chup thang hoac chup nguoc
            if es_point[1] > height / 2:  # dau do nam phia duoi
                rotation = const.ROTATION_CORRECT
            else:  # dau do nam phia tren
                rotation = const.ROTATION_REVERSE
        else:  # anh chup xoay trai hoac xoay phai
            if es_point[0] > width / 2:  # dau do nam ben phai
                rotation = const.ROTATION_LEFT
            else:  # dau do nam ben trai
                rotation = const.ROTATION_RIGHT
    return rotation


def detect_side_rotation(img_folder, filename_full, labeled_objects, result_step_folder):  # card_type,
    """ detect the side and rotation of provided image, based on bounding box (i.e, location) of its corners and emblem/sign.
    Args:
        img_folder: the folder contains original image
        filename_full: name of original image, including name and extension as a list: [name, extension]
        labeled_objects: the list of its corners and emblem/sign, including object name and bounding box.
        result_step_folder: the expected place to store the results.
    Returns:
        - the side and rotation of original image (save to an external file)
        - the new image and file_name after applying rotating to correct view.
    """
    print("\nfilename_full = ", filename_full)
    result_file = FJoin(result_step_folder, '{}_rotate_result.txt'.format(filename_full[0]))

    emblem_sign_objects = [obj for obj in labeled_objects
                           if obj['name'] in [const.PICTURE_EMBLEM, const.PICTURE_SIGN]]
    print("emblem_sign_objects = {}".format(emblem_sign_objects))
    if len(emblem_sign_objects) != 1:
        print("HYPERLOGY - found none or too many emblems/signs ??")
        with open(result_file, 'a') as f:
            f.write('Phat hien khong thay hoac qua nhieu : quoc huy / dau do.|NG')
            f.close()
    else:
        """
        Phan biet mat truoc/sau dua vao dau hieu: quoc huy / dau do.
        """
        emblem_sign_object = emblem_sign_objects[0]  # chi co 1 object trong list
        side = const.SIDE_FRONT if emblem_sign_object['name'] == const.PICTURE_EMBLEM else const.SIDE_BACK
        with open(result_file, 'a') as f:
            f.write('Phat hien duoc mat cua anh dau vao.|{}'.format(side))
            f.close()

        es_box = emblem_sign_object['bbox']  # Lay ra vi tri hinh dai dien (xmin, ymin, xmax, ymax)
        emblem_sign_point = ((es_box[0] + es_box[2]) / 2, (es_box[1] + es_box[3]) / 2)  # Lay ra tam cua hinh dai dien.
        print("emblem_sign_point = {}".format(emblem_sign_point))

        corner_objects = [obj for obj in labeled_objects if obj['name'] == const.PICTURE_CORNER]
        corner_bboxes = np.array([x['bbox'] for x in corner_objects])  # [xmin, ymin, xmax, ymax]
        # print("corner_boxes = {}".format(corner_bboxes))
        corner_number = len(corner_bboxes)
        """
        truong hop chup sat (toan man hinh): chi can xoay anh luon
        """
        if corner_number <= 1:
            orig_img = Image.open(FJoin(img_folder, filename_full[0] + filename_full[1]))
            (width, height) = orig_img.size
            # Xac dinh chieu xoay cua anh dua vao vi tri cua hinh dai dien (quoc huy / dau do) so voi 4 goc
            rotation = detect_rotation(side, (width, height), emblem_sign_point)
            # Ghi lai ket qua va Xoay lai anh
            with open(result_file, 'a') as f:
                f.write('Phat hien chieu cua anh dau vao.|{}'.format(rotation))
                f.close()
            rotated_img = rotate(orig_img, rotation)
            rotated_img.save(fp=FJoin(result_step_folder, filename_full[0] + "_rotate" + filename_full[1]))
            """
        truong hop chuan
        """
        elif corner_number == 4:
            # Bat dau xac dinh 4 diem input, 4 diem output de thuc hien phep chieu
            dst_pts = choose_corner_points(corner_bboxes)
            new_width, new_height = choose_perspective_size(dst_pts)

            src_pts = [(0, 0), (new_width, 0), (0, new_height), (new_width, new_height)]
            coeffs = find_coefficients(src_pts, dst_pts)  # coeffs = [a,b,c,d,e,f,g,h]
            new_es_point = find_reverse_point(coeffs, emblem_sign_point)
            # print("\ncoefficients = {}\nnew_es_point = {}".format(coeffs, new_es_point))

            orig_img = Image.open(FJoin(img_folder, filename_full[0] + filename_full[1]))
            trans_img = orig_img.transform((new_width, new_height), Image.PERSPECTIVE, coeffs, Image.BICUBIC)

            # Xac dinh chieu xoay cua anh dua vao vi tri cua hinh dai dien (quoc huy / dau do) so voi 4 goc
            rotation = detect_rotation(side, (new_width, new_height), new_es_point)
            # Ghi lai ket qua va Xoay lai anh
            with open(result_file, 'a') as f:
                f.write('Phat hien chieu cua anh dau vao.|{}'.format(rotation))
                f.close()
            rotated_img = rotate(trans_img, rotation)
            rotated_img.save(fp=FJoin(result_step_folder, filename_full[0] + "_rotate" + filename_full[1]))

            """
            tao dau vao cho qua trinh train bang segnet tren Google Colab
            """
            image_folder = FJoin(result_step_folder, 'img_idcard')  # thu muc chua anh goc
            os.makedirs(image_folder, exist_ok=True)
            mask_folder = FJoin(result_step_folder, 'mask_idcard')  # thu muc chua anh den trang tuong ung
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
            if width > height:
                orig_img.save(fp=FJoin(image_folder, filename_full[0] + filename_full[1]))
                background.save(fp=FJoin(mask_folder, "mask_" + filename_full[0] + ".png"))
            else:
                rotated_img = rotate(orig_img, const.ROTATION_RIGHT)
                rotated_img.save(fp=FJoin(image_folder, filename_full[0] + filename_full[1]))
                rotated_bgrd = rotate(background, const.ROTATION_RIGHT)
                rotated_bgrd.save(fp=FJoin(mask_folder, "mask_" + filename_full[0] + ".png"))

            """
        truong hop can thuat toan noi suy
        """
        elif corner_number == 3:
            print("HYPERLOGY - truong hop can thuat toan noi suy ?!")

        else:
            print("HYPERLOGY - found Not enough or Redundant corners ??")
            with open(result_file, 'a') as f:
                f.write('Phat hien khong du hoac qua nhieu goc cua CMND.|NG')
                f.close()

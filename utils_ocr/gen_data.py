#
#  Copyright by Hyperlogy Corporation, 2020
#  Smart eKYC project.
#
import codecs
import cv2
import json
import numpy as np
import os
import tensorflow as tf
import xml.dom.minidom as minidom
import xml.etree.ElementTree as et

from functools import partial
from PIL import Image, ImageDraw
from shutil import copyfile
from ocr_v2.config import constants as const
from ocr_v2.config.constants import FJoin
from ocr_v2.model.crop.detect_card import dist, sort_poly
from ocr_v2.model.detect_text.evaluate import get_annotation
from ocr_v2.utils.box_util import compute_target
from ocr_v2.utils.dir_util import dir_explore
from ocr_v2.utils.image_util import find_coefficients, remove_alpha, augmentation


class GenDataset:
    """ Class for Generate Dataset

    Attributes:
        root_dir: dataset root dir (ex: ./data/image)
        num_examples: number of examples to be used
                      (in case one wants to overfit small data)
    """

    def __init__(self, root_dir, default_boxes, new_size, num_examples):
        super(GenDataset, self).__init__()
        self.idx_to_name = const.LABELS_FACE_TEXT
        print("self.idx_to_name = ", self.idx_to_name)
        self.name_to_idx = dict([(v, k)
                                 for k, v in enumerate(self.idx_to_name)])
        self.data_dir = root_dir
        self.image_dir = self.data_dir  # FJoin(self.data_dir, '_RawImages')
        print("self.image_dir = ", self.image_dir)
        self.anno_dir = FJoin(self.data_dir, 'Annotation')
        self.ids = dir_explore(self.image_dir, explore_mode=const.EXPLORE_BROWSE)
        # print("HYPERLOGY self.ids =  ", self.ids)
        self.default_boxes = default_boxes
        self.new_size = new_size

        if num_examples != -1:
            self.ids = self.ids[:num_examples]

        self.train_ids = self.ids[:int(len(self.ids) * 0.75)]
        self.val_ids = self.ids[int(len(self.ids) * 0.75):]

        self.augmentation_methods = const.AUGMENTATION_METHODS + ['original']
        print("HYPERLOGY self.augmentation_methods = ", self.augmentation_methods)

    def __len__(self):
        return len(self.ids)

    def _get_image(self, index):
        """ Method to read image from file
            then resize to (300, 300)
            then subtract by ImageNet's mean
            then convert to Tensor
        Args:
            index: the index to get filename from self.ids
        Returns:
            img: tensor of shape (3, 300, 300)
        """
        filename = self.ids[index][0]
        fileextension = self.ids[index][1]
        img_path = FJoin(self.image_dir, filename + fileextension)
        # print("\n|| HYPERLOGY _get_image() img_path = ", img_path)
        img = remove_alpha(Image.open(img_path))  # remove alpha of image if necessary
        # print("|| HYPERLOGY _get_image() type(img) = {}".format(type(img)))
        return img

    def _get_annotation(self, index, orig_shape):  # only for APP_MODE_TRAIN and APP_MODE_PREDICT
        """ Method to read annotation from file
            Boxes are normalized to image size
            Integer labels are increased by 1
        Args:
            index: the index to get filename from self.ids
            orig_shape: image's original shape
        Returns:
            boxes: numpy array of shape (num_gt, 4)
            labels: numpy array of shape (num_gt,)
        """
        h, w = orig_shape
        filename = self.ids[index][0]
        anno_path = FJoin(self.anno_dir, filename + '.xml')
        # print("anno_path = ", anno_path)
        objects = et.parse(anno_path).findall('object')
        boxes = []
        labels = []

        for obj in objects:
            name = obj.find('name').text.lower().strip()
            # print("name = ", name)
            bndbox = obj.find('bndbox')
            xmin = (float(bndbox.find('xmin').text) - 1) / w
            ymin = (float(bndbox.find('ymin').text) - 1) / h
            xmax = (float(bndbox.find('xmax').text) - 1) / w
            ymax = (float(bndbox.find('ymax').text) - 1) / h
            boxes.append([xmin, ymin, xmax, ymax])

            labels.append(self.name_to_idx[name] + 1)

        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    def generate(self, subset=None):
        """ The __getitem__ method
            so that the object can be iterable
        Args:
            subset: the sub set of input data
        Returns:
            img: tensor of shape (300, 300, 3)
            boxes: tensor of shape (num_gt, 4)
            labels: tensor of shape (num_gt,)
        """
        indices = self.ids
        if subset == 'train':
            indices = self.train_ids
        elif subset == 'val':
            indices = self.val_ids

        for index in range(len(indices)):
            orig_filename = indices[index][0]
            fileextension = indices[index][1]
            orig_img = self._get_image(index)
            w, h = orig_img.size
            boxes, labels = self._get_annotation(index, (h, w))
            boxes = tf.constant(boxes, dtype=tf.float32)
            labels = tf.constant(labels, dtype=tf.int64)
            #print("\n index = {}, orig_filename = {}, boxes = {}, labels = {}".format(index, orig_filename, boxes, labels))
            # print("|| HYPERLOGY generate() index = {}, orig_filename = {}".format(index, orig_filename))

            # includes original and augmented images.
            number_augmentation = const.NUMBER_AUGMENTATIONS + 1
            for times in range(number_augmentation):
                augmentation_method = np.random.choice(self.augmentation_methods)
                filename = orig_filename
                img = orig_img
                if augmentation_method != 'original':
                    filename, img = augmentation(filename, img, augmentation_method)  # thuc hien augmentation method
                # print("|| HYPERLOGY generate() augmentation_methods = {}, filename = {}, type(img) = {}".format(augmentation_methods,filename,type(img)))
                img = np.array(img.resize(
                    (self.new_size, self.new_size)), dtype=np.float32)
                img = (img / 127.0) - 1.0
                img = tf.constant(img, dtype=tf.float32)

                gt_confs, gt_locs = compute_target(self.default_boxes, boxes, labels)
                # print("\n gt_confs = {}, gt_locs = {}".format(gt_confs, gt_locs))
                yield filename, fileextension, img, gt_confs, gt_locs


def create_batch_generator(root_dir, default_boxes,
                           new_size, batch_size, num_batches):
    # Giu nguyen num_batches va num_examples khi truy cap thu muc anh goc.
    # num_examples = batch_size * num_batches if num_batches > 0 else -1
    data = GenDataset(root_dir, default_boxes, new_size, batch_size * num_batches)

    info = {
        'idx_to_name': data.idx_to_name,
        'name_to_idx': data.name_to_idx,
        'length': len(data),
        'image_dir': data.image_dir,
        'anno_dir': data.anno_dir
    }

    num_batches *= (const.NUMBER_AUGMENTATIONS + 1) if num_batches > 0 else -1
    num_examples = batch_size * num_batches if num_batches > 0 else -1

    train_gen = partial(data.generate, subset='train')
    train_dataset = tf.data.Dataset.from_generator(train_gen, (tf.string, tf.string, tf.float32, tf.int64, tf.float32))
    train_dataset = train_dataset.shuffle(buffer_size=num_examples).batch(batch_size)
    #train_dataset = train_dataset.batch(batch_size)
    #print("    Call SHUFFLE Call SHUFFLE Call SHUFFLE")

    val_gen = partial(data.generate, subset='val')
    val_dataset = tf.data.Dataset.from_generator(val_gen, (tf.string, tf.string, tf.float32, tf.int64, tf.float32))
    val_dataset = val_dataset.batch(1)  # validate for each image

    return train_dataset.take(-1), val_dataset.take(-1), info


def generate_dataset_step01_mask(input_img_folder, input_mask_folder):
    """ generate the 'combined' image from original image and mask image (put mask above the original)
    :param input_img_folder:
    :param input_mask_folder:
    :return:
    """
    img_paste_mask_folder = FJoin(input_img_folder, 'paste_mask')
    # img_paste_mask_folder = FJoin(input_mask_folder, 'paste_mask')
    os.makedirs(img_paste_mask_folder, exist_ok=True)
    img_list = dir_explore(input_img_folder, const.EXPLORE_BROWSE)
    for img_file in img_list:
        mask_file = 'mask_' + img_file[0] + '.png'
        mask_path = FJoin(input_mask_folder, mask_file)
        print("\n\nmask_file = {}".format(mask_file))
        if os.path.exists(mask_path):
            image = Image.open(fp=FJoin(input_img_folder, img_file[0] + img_file[1]))
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print("Error case with mask file: '{}'".format(mask_file))
            else:
                areas = []
                vertices = None

                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    areas.append(area)
                idx = np.argmax(areas)
                cnt = contours[idx]
                if areas[int(idx)] > 100:
                    approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
                    # Checking if the no. of sides of the selected region is 4.
                    if len(approx) == 4:
                        vertices = approx

                box = sort_poly(vertices.astype(np.int32).reshape((4, 2)))
                x, y, z, t = box[1], box[0], box[3], box[2]

                if dist(x, y) < dist(x, t):
                    r = dist(x, y)
                    d = dist(x, t)
                    style = 0
                else:
                    d = dist(x, y)
                    r = dist(x, t)
                    style = 1
                r = int(r)
                d = int(d)
                # print("r = {}, d = {}".format(r, d))

                pts1 = np.float32([x, y, z, t])
                if style == 0:
                    pts2 = np.float32([[0, 0], [0, r], [d, r], [d, 0]])
                else:  # style == 1:
                    pts2 = np.float32([[d, 0], [0, 0], [0, r], [d, r]])
                coeffs = find_coefficients(pts2, pts1)  # lay ti le phep chieu
                reversed_coeffs = find_coefficients(pts1, pts2)
                dst = image.transform((d, r), Image.PERSPECTIVE, coeffs,
                                      Image.BICUBIC)  # keo dan anh ra 4 goc bang phep chieu
                # print("image.size = {}".format(image.size))
                dst = dst.convert('RGBA')

                # paste vao anh ban dau
                # reversed_img = dst.transform(image.size, Image.PERSPECTIVE, reversed_coeffs, Image.BICUBIC)
                # print("reversed_img.size = {}".format(reversed_img.size))
                mask_img = Image.new('RGBA', (d, r), (255, 255, 255, 255))
                reversed_mask = mask_img.transform(image.size, Image.PERSPECTIVE, reversed_coeffs, Image.BICUBIC)
                image.paste(reversed_mask, (0, 0), mask=reversed_mask)
                image.save(fp=FJoin(img_paste_mask_folder, img_file[0] + img_file[1]))
        else:
            print("\n Do not find mask image for '{}'".format(img_file[0] + img_file[1]))


def generate_dataset_step01_check_legal(input_img_mask_folder, input_background_folder):
    """ generate the input data for classification
    :param input_img_mask_folder:
    :param input_background_folder:
    :return:
    """
    input_img_corner_cut_path = FJoin(input_img_mask_folder, 'idcard_corner_cut')
    input_img_corner_cut_results = FJoin(input_img_corner_cut_path, 'results')
    os.makedirs(input_img_corner_cut_results, exist_ok=True)
    img_list_corner_cut = dir_explore(input_img_corner_cut_path, const.EXPLORE_BROWSE)

    input_img_circle_cut_path = FJoin(input_img_mask_folder, 'idcard_circle_cut')
    input_img_circle_cut_results = FJoin(input_img_circle_cut_path, 'results')
    os.makedirs(input_img_circle_cut_results, exist_ok=True)
    img_list_circle_cut = dir_explore(input_img_circle_cut_path, const.EXPLORE_BROWSE)

    input_mask_path = FJoin(input_img_mask_folder, 'mask')
    background_list = dir_explore(input_background_folder, const.EXPLORE_BROWSE)
    background_list = [file[0] + file[1] for file in background_list]
    # for img_file in img_list_corner_cut:
    #     image = Image.open(fp=FJoin(input_img_corner_cut_path, img_file[0]+img_file[1]))
    #     mask_file = 'mask_' + img_file[0] + '.png'
    #     print("\n\nmask_file = {}".format(mask_file))
    #     mask = cv2.imread(FJoin(input_mask_path, mask_file), cv2.IMREAD_GRAYSCALE)
    #     if mask is None:
    #         print("Error case with img_file '{}'".format(img_file))
    #     else:
    #         areas = []
    #         vertices = None
    #
    #         contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #         for cnt in contours:
    #             area = cv2.contourArea(cnt)
    #             areas.append(area)
    #         idx = np.argmax(areas)
    #         cnt = contours[idx]
    #         if areas[int(idx)] > 100:
    #             approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
    #             # Checking if the no. of sides of the selected region is 4.
    #             if len(approx) == 4:
    #                 vertices = approx
    #
    #         box = sort_poly(vertices.astype(np.int32).reshape((4, 2)))
    #         x, y, z, t = box[1], box[0], box[3], box[2]
    #
    #         if dist(x, y) < dist(x, t):
    #             r = dist(x, y)
    #             d = dist(x, t)
    #             style = 0
    #         else:
    #             d = dist(x, y)
    #             r = dist(x, t)
    #             style = 1
    #         r = int(r)
    #         d = int(d)
    #         # print("r = {}, d = {}".format(r, d))
    #
    #         pts1 = np.float32([x, y, z, t])
    #         if style == 0:
    #             pts2 = np.float32([[0, 0], [0, r], [d, r], [d, 0]])
    #         else:  # style == 1:
    #             pts2 = np.float32([[d, 0], [0, 0], [0, r], [d, r]])
    #         coeffs = find_coefficients(pts2, pts1)  # lay ti le phep chieu
    #         reversed_coeffs = find_coefficients(pts1, pts2)
    #         dst = image.transform((d, r), Image.PERSPECTIVE, coeffs, Image.BICUBIC)  # keo dan anh ra 4 goc bang phep chieu
    #         # print("image.size = {}".format(image.size))
    #         dst = dst.convert('RGBA')
    #
    #         """
    #         sinh anh 'cat goc'
    #         """
    #         # chon ngau nhien ti le cat o hai canh (trong khoang 15-30%)
    #         ratio_cut_r = np.random.choice(np.arange(15, 30))
    #         ratio_cut_d = np.random.choice(np.arange(15, 30))
    #         # print("ratio_cut_r = {}, ratio_cut_d = {}".format(ratio_cut_r, ratio_cut_d))
    #         dimen_r = int(ratio_cut_r * r / 100)
    #         dimen_d = int(ratio_cut_d * d / 100)
    #         # print("dimen_r = {}, dimen_d = {}".format(dimen_r, dimen_d))
    #
    #         # sinh ngau nhien 1 vung cat voi kich thuoc da chon tu anh background
    #         background_file = np.random.choice(background_list)
    #         print("background_file = {}".format(background_file))
    #         background_img = cv2.imread(FJoin(input_background_folder, background_file))
    #         background_size = background_img.shape
    #         # print("background_size = {}".format(background_size))
    #         cut_xmin = (background_size[0] - dimen_d) // 2
    #         cut_xmax = (background_size[0] + dimen_d) // 2
    #         cut_ymin = (background_size[1] - dimen_r) // 2
    #         cut_ymax = (background_size[1] + dimen_r) // 2
    #         background_cut_img = background_img[cut_ymin:cut_ymax, cut_xmin:cut_xmax]
    #         # print("background_cut_img.shape = {}".format(background_cut_img.shape))
    #         cv2.imwrite(FJoin(input_background_folder, 'temp', "temp.png"), background_cut_img)
    #
    #         # chon ngau nhien 1 trong 4 goc cat: 0 la trai tren, 1 la phai tren, 2 la trai duoi, 3 la phai duoi
    #         cut_points = [(0, 0), (dimen_d - 1, 0), (0, dimen_r - 1), (dimen_d - 1, dimen_r - 1)]
    #         base_points = [(0, 0), (d - dimen_d - 1, 0), (0, r - dimen_r - 1), (d - dimen_d - 1, r - dimen_r - 1)]
    #         # print("cut_points = {}".format(cut_points))
    #         ratio_cut_corner = np.random.choice(np.arange(4))
    #         # print("ratio_cut_corner = {}".format(ratio_cut_corner))
    #         paste_position = base_points[ratio_cut_corner]
    #         # print("paste_position = {}".format(paste_position))
    #         cut_points.remove(cut_points[3 - ratio_cut_corner])
    #
    #         # paste vao anh ban dau
    #         paste_img = Image.open(FJoin(input_background_folder, 'temp', "temp.png"))
    #         paste_size = paste_img.size
    #         mask = Image.new('L', paste_size, 0)
    #         pdraw = ImageDraw.Draw(mask)
    #         pdraw.polygon(cut_points, fill=255)  # , outline=(255, 0, 0, 255)
    #         dst.paste(paste_img, paste_position, mask=mask)
    #         reversed_img = dst.transform(image.size, Image.PERSPECTIVE, reversed_coeffs, Image.BICUBIC)
    #         # print("reversed_img.size = {}".format(reversed_img.size))
    #         mask_img = Image.new('RGBA', (d, r), (255, 255, 255, 255))
    #         reversed_mask = mask_img.transform(image.size, Image.PERSPECTIVE, reversed_coeffs, Image.BICUBIC)
    #         image.paste(reversed_img, (0, 0), mask=reversed_mask)
    #         image.save(fp=FJoin(input_img_corner_cut_results, img_file[0] + img_file[1]))

    for img_file in img_list_circle_cut:
        image = Image.open(fp=FJoin(input_img_circle_cut_path, img_file[0]+img_file[1]))
        mask_file = 'mask_' + img_file[0] + '.png'
        print("\n\nmask_file = {}".format(mask_file))
        mask = cv2.imread(FJoin(input_mask_path, mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print("Error case with img_file '{}'".format(img_file))
        else:
            areas = []
            vertices = None

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                areas.append(area)
            idx = np.argmax(areas)
            cnt = contours[idx]
            if areas[int(idx)] > 100:
                approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
                # Checking if the no. of sides of the selected region is 4.
                if len(approx) == 4:
                    vertices = approx

            box = sort_poly(vertices.astype(np.int32).reshape((4, 2)))
            x, y, z, t = box[1], box[0], box[3], box[2]

            if dist(x, y) < dist(x, t):
                r = dist(x, y)
                d = dist(x, t)
                style = 0
            else:
                d = dist(x, y)
                r = dist(x, t)
                style = 1
            r = int(r)
            d = int(d)
            # print("r = {}, d = {}".format(r, d))

            pts1 = np.float32([x, y, z, t])
            if style == 0:
                pts2 = np.float32([[0, 0], [0, r], [d, r], [d, 0]])
            else:  # style == 1:
                pts2 = np.float32([[d, 0], [0, 0], [0, r], [d, r]])
            coeffs = find_coefficients(pts2, pts1)  # lay ti le phep chieu
            reversed_coeffs = find_coefficients(pts1, pts2)
            dst = image.transform((d, r), Image.PERSPECTIVE, coeffs, Image.BICUBIC)  # keo dan anh ra 4 goc bang phep chieu
            # print("image.size = {}".format(image.size))
            dst = dst.convert('RGBA')

            """
            sinh anh 'duc lo'
            """
            # chon ngau nhien ti le cat o hai canh (trong khoang 10-15%)
            ratio_cut_r = np.random.choice(np.arange(10, 15))
            ratio_cut_d = np.random.choice(np.arange(10, 15))
            # print("ratio_cut_r = {}, ratio_cut_d = {}".format(ratio_cut_r, ratio_cut_d))
            dimen_r = int(ratio_cut_r * r / 100)
            dimen_d = int(ratio_cut_d * d / 100)
            # print("dimen_r = {}, dimen_d = {}".format(dimen_r, dimen_d))

            # sinh ngau nhien 1 vung cat voi kich thuoc da chon tu anh background
            background_file = np.random.choice(background_list)
            print("background_file = {}".format(background_file))
            background_img = cv2.imread(FJoin(input_background_folder, background_file))
            background_size = background_img.shape
            # print("background_size = {}".format(background_size))
            cut_xmin = (background_size[0] - dimen_d) // 2
            cut_xmax = (background_size[0] + dimen_d) // 2
            cut_ymin = (background_size[1] - dimen_r) // 2
            cut_ymax = (background_size[1] + dimen_r) // 2
            background_cut_img = background_img[cut_ymin:cut_ymax, cut_xmin:cut_xmax]
            # print("background_cut_img.shape = {}".format(background_cut_img.shape))
            cv2.imwrite(FJoin(input_background_folder, 'temp', "temp.png"), background_cut_img)

            # chon ngau nhien 1 trong 4 goc cat: 0 la trai tren, 1 la phai tren, 2 la trai duoi, 3 la phai duoi
            # cut_points = [(0, 0), (dimen_d - 1, 0), (0, dimen_r - 1), (dimen_d - 1, dimen_r - 1)]
            base_points = [(dimen_d//2, dimen_r//2), (d - (dimen_d*3)//2 - 1, dimen_r//2), (dimen_d//2, r - (dimen_r*3)//2 - 1), (d - (dimen_d*3)//2 - 1, r - (dimen_r*3)//2 - 1)]
            # print("cut_points = {}".format(cut_points))
            ratio_cut_corner = np.random.choice(np.arange(4))
            # print("ratio_cut_corner = {}".format(ratio_cut_corner))
            paste_position = base_points[ratio_cut_corner]
            # print("paste_position = {}".format(paste_position))
            # cut_points.remove(cut_points[3 - ratio_cut_corner])

            # paste vao anh ban dau
            paste_img = Image.open(FJoin(input_background_folder, 'temp', "temp.png"))
            paste_size = paste_img.size
            mask = Image.new('L', paste_size, 0)
            pdraw = ImageDraw.Draw(mask)
            dimen = int(min(dimen_d, dimen_r)*0.8)  # duong kinh duong tron
            pdraw.ellipse(((dimen_d-dimen)//2, (dimen_r-dimen)//2, (dimen_d+dimen)//2, (dimen_r+dimen)//2), fill=255)
            dst.paste(paste_img, paste_position, mask=mask)
            reversed_img = dst.transform(image.size, Image.PERSPECTIVE, reversed_coeffs, Image.BICUBIC)
            # print("reversed_img.size = {}".format(reversed_img.size))
            mask_img = Image.new('RGBA', (d, r), (255, 255, 255, 255))
            reversed_mask = mask_img.transform(image.size, Image.PERSPECTIVE, reversed_coeffs, Image.BICUBIC)
            image.paste(reversed_img, (0, 0), mask=reversed_mask)
            image.save(fp=FJoin(input_img_circle_cut_results, img_file[0] + img_file[1]))


def generate_dataset_step02_ssd(input_folder):
    """ generate the SSD input annotations from labeled data of step_02 (got at HyperAI web app)
    Args:
        input_folder: the folder contains labeled data
    Returns:
        the corresponding ssd input annotations for each input image
    """
    file_list = dir_explore(input_folder, const.EXPLORE_BROWSE)
    output_front_anno = FJoin(input_folder, const.SIDE_FRONT, 'Annotation')  # contain front images and corresponding Annotation
    output_back_anno = FJoin(input_folder, const.SIDE_BACK, 'Annotation')  # contain back images and corresponding Annotation
    os.makedirs(output_front_anno, exist_ok=True)
    os.makedirs(output_back_anno, exist_ok=True)
    for file in file_list:
        anno_path = FJoin(input_folder, 'Annotation', '{}.xml'.format(file[0]))
        print("anno_path = {}".format(anno_path))
        if os.path.exists(anno_path):
            labeled_objs = get_annotation(anno_path)
            #print("labeled_objs = {}".format(labeled_objs))
            picture_objects = [obj for obj in labeled_objs if obj['type'] == const.PICTURE]
            side = const.SIDE_FRONT if len(picture_objects) == 1 else const.SIDE_BACK
            image = Image.open(FJoin(input_folder, file[0] + file[1]))
            image.save(FJoin(input_folder, side, file[0] + file[1]))
            anno_side_result = FJoin(input_folder, side, 'Annotation', '{}.xml'.format(file[0]))
            annotation = et.Element("annotation")
            for obj in labeled_objs:
                obj_type = obj["type"]
                object_name = obj_type if obj_type in [const.PICTURE, const.LABEL] else const.CONTENT
                object_tag = et.SubElement(annotation, "object")
                et.SubElement(object_tag, "name").text = object_name
                et.SubElement(object_tag, "difficult").text = '0'

                obj_bbox = obj['bbox']  # xmin, ymin, xmax, ymax
                bndbox = et.SubElement(object_tag, "bndbox")
                et.SubElement(bndbox, "xmin").text = str(obj_bbox[0])
                et.SubElement(bndbox, "ymin").text = str(obj_bbox[1])
                et.SubElement(bndbox, "xmax").text = str(obj_bbox[2])
                et.SubElement(bndbox, "ymax").text = str(obj_bbox[3])

            rough_string = et.tostring(annotation, 'utf-8')
            reparsed = minidom.parseString(rough_string)
            reparsed = reparsed.toprettyxml(indent="    ")
            with open(anno_side_result, 'w') as f:
                f.write(reparsed)
                f.close()


def generate_dataset_step02_ssd_add_emblemsign(input_folder):
    """ generate the SSD input annotations from previous generated annotations and manual labeled data
    Args:
        input_folder: the folder contains previous generated annotations and labeled data
    Returns:
        the corresponding ssd input annotations for each input image
    """
    file_list = dir_explore(input_folder, const.EXPLORE_BROWSE)
    output_front_anno = FJoin(input_folder, const.SIDE_FRONT, 'Annotation')  # contain front images and corresponding Annotation
    output_back_anno = FJoin(input_folder, const.SIDE_BACK, 'Annotation')  # contain back images and corresponding Annotation
    os.makedirs(output_front_anno, exist_ok=True)
    os.makedirs(output_back_anno, exist_ok=True)
    for file in file_list:
        anno_path = FJoin(input_folder, 'Annotation', '{}.xml'.format(file[0]))
        print("anno_path = {}".format(anno_path))
        anno_path_emblemsign = FJoin(input_folder, 'Annotation_Emblem_Sign', '{}.xml'.format(file[0]))
        if os.path.exists(anno_path) and os.path.exists(anno_path_emblemsign):
            labeled_objs = get_annotation(anno_path)
            picture_objects = [obj for obj in labeled_objs if obj['type'] == const.PICTURE]
            side = const.SIDE_FRONT if len(picture_objects) == 1 else const.SIDE_BACK
            img_src_path = FJoin(input_folder, file[0] + file[1])
            img_dst_path = FJoin(input_folder, side, file[0] + file[1])
            copyfile(img_src_path, img_dst_path)
            anno_side_result = FJoin(input_folder, side, 'Annotation', '{}.xml'.format(file[0]))
            annotation = et.Element("annotation")
            # lay lai cac nhan da gan, xuat ra dau vao cho SSD
            for obj in labeled_objs:
                object_tag = et.SubElement(annotation, "object")
                obj_type = obj["type"]
                object_name = obj_type if obj_type in [const.PICTURE, const.LABEL] else const.CONTENT
                et.SubElement(object_tag, "name").text = object_name
                et.SubElement(object_tag, "difficult").text = '0'
                obj_bbox = obj['bbox']  # xmin, ymin, xmax, ymax
                bndbox = et.SubElement(object_tag, "bndbox")
                et.SubElement(bndbox, "xmin").text = str(obj_bbox[0])
                et.SubElement(bndbox, "ymin").text = str(obj_bbox[1])
                et.SubElement(bndbox, "xmax").text = str(obj_bbox[2])
                et.SubElement(bndbox, "ymax").text = str(obj_bbox[3])

            # lay them nhan moi (gan bo sung)
            labeled_objs_emblemsign = get_annotation(anno_path_emblemsign)
            emblemsign_objects = [obj for obj in labeled_objs_emblemsign if obj['name'] in [const.PICTURE_EMBLEM, const.PICTURE_SIGN]]
            for obj in emblemsign_objects:
                object_tag = et.SubElement(annotation, "object")
                et.SubElement(object_tag, "name").text = const.PICTURE
                et.SubElement(object_tag, "difficult").text = '0'
                obj_bbox = obj['bbox']  # xmin, ymin, xmax, ymax
                bndbox = et.SubElement(object_tag, "bndbox")
                et.SubElement(bndbox, "xmin").text = str(obj_bbox[0])
                et.SubElement(bndbox, "ymin").text = str(obj_bbox[1])
                et.SubElement(bndbox, "xmax").text = str(obj_bbox[2])
                et.SubElement(bndbox, "ymax").text = str(obj_bbox[3])

            rough_string = et.tostring(annotation, 'utf-8')
            reparsed = minidom.parseString(rough_string)
            reparsed = reparsed.toprettyxml(indent="    ")
            with open(anno_side_result, 'w', encoding='utf-8') as f:
                f.write(reparsed)
                f.close()


def generate_dataset_step02_icdar(input_folder):
    """ generate the ICDAR_2015 input from labeled data of step_02 (got at HyperAI web app)
    Args:
        input_folder: the folder contains labeled data
    Returns:
        the corresponding EAST input for each input image
    """
    file_list = dir_explore(input_folder, const.EXPLORE_BROWSE)
    output_folder = FJoin(input_folder, 'ICDAR_2015')
    output_front_img = FJoin(output_folder, const.SIDE_FRONT, 'train_images')  # contain front images
    output_front_anno = FJoin(output_folder, const.SIDE_FRONT, 'train_gt')  # contain corresponding Annotation with front images
    output_back_img = FJoin(output_folder, const.SIDE_BACK, 'train_images')  # contain back images
    output_back_anno = FJoin(output_folder, const.SIDE_BACK, 'train_gt')  # contain back images
    output_both_anno = FJoin(output_folder, 'both_sides', 'train_gt')  # contain corresponding Annotation with back images
    os.makedirs(output_front_img, exist_ok=True)
    os.makedirs(output_front_anno, exist_ok=True)
    os.makedirs(output_back_img, exist_ok=True)
    os.makedirs(output_back_anno, exist_ok=True)
    os.makedirs(output_both_anno, exist_ok=True)
    for file in file_list:
        anno_path = FJoin(input_folder, 'Annotation', '{}.xml'.format(file[0]))
        print("anno_path = {}".format(anno_path))
        if os.path.exists(anno_path):
            labeled_objs = get_annotation(anno_path)
            #print("labeled_objs = {}".format(labeled_objs))
            picture_objects = [obj for obj in labeled_objs if obj['type'] == const.PICTURE]
            side = const.SIDE_FRONT if len(picture_objects) == 1 else const.SIDE_BACK
            image = Image.open(FJoin(input_folder, file[0] + file[1]))
            image.save(FJoin(output_folder, side, "train_images", file[0] + file[1]))
            anno_side_result = FJoin(output_folder, side, 'train_gt', 'gt_{}.txt'.format(file[0]))
            anno_both_result = FJoin(output_both_anno, '{}.txt'.format(file[0]))
            for obj in labeled_objs:
                obj_type = obj["type"]
                if obj_type != const.PICTURE:
                    obj_name = obj['name']
                    obj_bbox = obj['bbox']  # xmin, ymin, xmax, ymax
                    xmin, ymin, xmax, ymax = obj_bbox[:]  # [str(coord) for coord in obj_bbox]
                    with codecs.open(anno_side_result, 'a', encoding='utf-8') as f:
                        f.write("{},{},{},{},{},{},{},{},{}\n".format(xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax, obj_name))
                        f.close()
                    with codecs.open(anno_both_result, 'a', encoding='utf-8') as f:
                        f.write("{},{},{},{},{},{},{},{},{}\n".format(xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax, obj_name))
                        f.close()


def generate_dataset_step02_tiny_yolov3(input_folder):
    """ generate the Tiny YOLO v3 input from labeled data of step_02 (got at HyperAI web app)
    Args:
        input_folder: the folder contains labeled data
    Returns:
        the corresponding EAST input for each input image
    """
    file_list = dir_explore(input_folder, const.EXPLORE_BROWSE)
    output_folder = FJoin(input_folder, 'TinyYoloV3')  # chuyen thanh dau vao cho Tiny Yolo v3
    output_front_folder = FJoin(output_folder, const.SIDE_FRONT)  # contain front images
    output_back_folder = FJoin(output_folder, const.SIDE_BACK)  # contain back images
    os.makedirs(output_front_folder, exist_ok=True)
    os.makedirs(output_back_folder, exist_ok=True)
    for file in file_list:
        anno_path = FJoin(input_folder, 'Annotation', '{}.xml'.format(file[0]))
        print("anno_path = {}".format(anno_path))
        if os.path.exists(anno_path):
            labeled_objs = get_annotation(anno_path)
            #print("labeled_objs = {}".format(labeled_objs))
            picture_objects = [obj for obj in labeled_objs if obj['type'] == const.PICTURE]
            side = const.SIDE_FRONT if len(picture_objects) == 1 else const.SIDE_BACK
            output_side_folder = FJoin(output_folder, side)
            image = Image.open(FJoin(input_folder, file[0] + file[1]))
            (img_width, img_height) = image.size
            anno_side_result = FJoin(output_side_folder, '{}.txt'.format(file[0]))

            def xywh(obj):
                ob_bbox = obj['bbox']  # xmin, ymin, xmax, ymax
                xmi, ymi, xma, yma = ob_bbox[:]
                cente_x = (xmi + xma) / (2 * img_width)
                cente_y = (ymi + yma) / (2 * img_height)
                box_wi = (xma - xmi) / img_width
                box_he = (yma - ymi) / img_height
                return cente_x, cente_y, box_wi, box_he

            if side == const.SIDE_FRONT:
                image.save(FJoin(output_side_folder, file[0] + file[1]))

                numbers = [obj for obj in labeled_objs if obj['type'] == const.NUMBER]
                if len(numbers) > 0:
                    number = numbers[0]
                    center_x, center_y, box_width, box_height = xywh(number)
                    with codecs.open(anno_side_result, 'a', encoding='utf-8') as f:
                        f.write("{} {} {} {} {}\n".format('0', center_x, center_y, box_width, box_height))
                        f.close()

                datetimes = [obj for obj in labeled_objs if obj['type'] == const.DATETIME]
                if len(datetimes) > 0:
                    datetime = datetimes[0]
                    center_x, center_y, box_width, box_height = xywh(datetime)
                    with codecs.open(anno_side_result, 'a', encoding='utf-8') as f:
                        f.write("{} {} {} {} {}\n".format('1', center_x, center_y, box_width, box_height))
                        f.close()

                contents = [obj for obj in labeled_objs if obj['type'] == const.CONTENT]
                sorted_ymin = [contents[x] for x in np.argsort([obj['bbox'][1] for obj in contents])]  # sap xep theo ymin
                for i, obj in enumerate(sorted_ymin):
                    index = 1
                    # obj_bbox = obj['bbox']  # xmin, ymin, xmax, ymax
                    # xmin, ymin, xmax, ymax = obj_bbox[:]
                    # if i == 0 and ymin < img_height / 2:  # truong Ho ten luon nam o nua tren CMND
                    #     index = 1
                    center_x, center_y, box_w, box_h = xywh(obj)
                    with codecs.open(anno_side_result, 'a', encoding='utf-8') as f:
                        f.write("{} {} {} {} {}\n".format(str(index), center_x, center_y, box_w, box_h))
                        f.close()
            elif side == const.SIDE_BACK:
                image.save(FJoin(output_side_folder, file[0] + file[1]))

                numbers = [obj for obj in labeled_objs if obj['type'] == const.NUMBER]
                for number in numbers:
                    center_x, center_y, box_width, box_height = xywh(number)
                    with codecs.open(anno_side_result, 'a', encoding='utf-8') as f:
                        f.write("{} {} {} {} {}\n".format('0', center_x, center_y, box_width, box_height))
                        f.close()

                contents = [obj for obj in labeled_objs if obj['type'] == const.CONTENT]
                sorted_ymin = [contents[x] for x in np.argsort([obj['bbox'][1] for obj in contents])]  # sap xep theo ymin
                for obj in sorted_ymin:
                    index = 1
                    center_x, center_y, box_w, box_h = xywh(obj)
                    with codecs.open(anno_side_result, 'a', encoding='utf-8') as f:
                        f.write("{} {} {} {} {}\n".format(str(index), center_x, center_y, box_w, box_h))
                        f.close()
        else:
            print("Except image is {}".format(file[0]+file[1]))


def generate_dataset_step03_tesseract(input_folder):
    """ generate the Tesseract input from labeled data of step_02 (got at HyperAI web app)
    Args:
        input_folder: the folder contains labeled data
    Returns:
        the corresponding Tesseract input for each input image
    """
    # cat cac anh chua text de lam dau vao cua buoc 3 (recognition)
    file_list = dir_explore(input_folder, const.EXPLORE_BROWSE)
    output_folder = FJoin(input_folder, 'Tesseract')
    front_id_folder = FJoin(output_folder, 'front_id')
    front_text_folder = FJoin(output_folder, 'front_text')
    back_folder = FJoin(output_folder, 'back')
    temp_folder = FJoin(output_folder, 'temp')  # store the temporary images before copying to save_folder
    os.makedirs(front_id_folder, exist_ok=True)
    os.makedirs(front_text_folder, exist_ok=True)
    os.makedirs(back_folder, exist_ok=True)
    os.makedirs(temp_folder, exist_ok=True)

    for file in file_list:
        anno_path = FJoin(input_folder, 'Annotation', '{}.xml'.format(file[0]))
        print("anno_path = {}".format(anno_path))
        if os.path.exists(anno_path):
            labeled_objs = get_annotation(anno_path)
            #print("labeled_objs = {}".format(labeled_objs))
            picture_objects = [obj for obj in labeled_objs if obj['type'] == const.PICTURE]
            side = const.SIDE_FRONT if len(picture_objects) == 1 else const.SIDE_BACK
            image = Image.open(FJoin(input_folder, file[0] + file[1]))
            for i, obj in enumerate(labeled_objs):
                obj_type = obj["type"]
                if obj_type != const.PICTURE:
                    obj_name = obj['name']
                    box_name = file[0] + '_' + str(i) + ".tif"
                    gt_name = file[0] + '_' + str(i) + ".gt.txt"
                    obj_bbox = obj['bbox']  # xmin, ymin, xmax, ymax
                    xmin, ymin, xmax, ymax = obj_bbox[:]  # [str(coord) for coord in obj_bbox]
                    crop_img = remove_alpha(image.crop((xmin, ymin, xmax, ymax)))
                    is_save = False
                    save_folder = None
                    if xmin < xmax and ymin < ymax:
                        if side == const.SIDE_FRONT:
                            if obj_type == const.NUMBER:
                                is_save = True
                                save_folder = front_id_folder
                                # label_front_id[box_name] = box_text
                            elif obj_type != const.LABEL:
                                is_save = True
                                save_folder = front_text_folder
                                # label_front_text[box_name] = box_text
                        elif side == const.SIDE_BACK and obj_type != const.LABEL:
                            is_save = True
                            save_folder = back_folder
                            # label_back[box_name] = box_text

                    if is_save:
                        temp_path_img = FJoin(temp_folder, box_name)
                        save_path_img = FJoin(save_folder, box_name)
                        save_path_gt = FJoin(save_folder, gt_name)
                        try:
                            # neu save dc vao thu muc temp ma khong loi, thi moi save vao thu muc ket qua
                            crop_img.save(temp_path_img, format="TIFF")
                            crop_img.save(save_path_img, format="TIFF")
                            with codecs.open(save_path_gt, 'w', encoding='utf-8') as f:
                                f.write(obj_name)
                                f.close()
                        except TypeError:
                            print("Gap anh loi")


def generate_dataset_step03_xception(input_folder, card_type, thres_height):
    """ generate the Xception input from labeled data of step_02 (got at HyperAI web app)
    Args:
        input_folder: the folder contains labeled data
        card_type: 'cmnd_cccd_2016' or 'cmnd_cu'
        thres_height: chi lay nhung anh co height >= 40 pixels. Dat thres_height = 0 thi khong filter.
    Returns:
        the corresponding Xception input for each input image
    """
    # cat cac anh chua text de lam dau vao cua buoc 3 (recognition)
    file_list = dir_explore(input_folder, const.EXPLORE_BROWSE)
    suffix = "_filter_height" if thres_height > 0 else ""
    output_folder = FJoin(input_folder, 'Xception{}'.format(suffix)) if thres_height > 0 else FJoin(input_folder, 'Xception')
    front_id_folder = FJoin(output_folder, 'front_id{}'.format(suffix))
    front_text_folder = FJoin(output_folder, 'front_text{}'.format(suffix))
    back_folder = FJoin(output_folder, 'back{}'.format(suffix))
    temp_folder = FJoin(output_folder, 'temp')  # store the temporary images before copying to save_folder
    label_folder = FJoin(output_folder, "label_full")
    os.makedirs(front_id_folder, exist_ok=True)
    os.makedirs(front_text_folder, exist_ok=True)
    os.makedirs(back_folder, exist_ok=True)
    os.makedirs(temp_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)
    label_front_id = {}
    label_front_text = {}
    label_back = {}

    for i, file in enumerate(file_list):
        anno_path = FJoin(input_folder, 'Annotation', '{}.xml'.format(file[0]))
        if i % 100 == 0:
            print("anno_path = {}".format(anno_path))
        if os.path.exists(anno_path):
            labeled_objs = get_annotation(anno_path)
            # print("labeled_objs = {}".format(labeled_objs))
            picture_objects = [obj for obj in labeled_objs if obj['type'] == const.PICTURE]
            side = const.SIDE_FRONT if len(picture_objects) == 1 else const.SIDE_BACK
            image = Image.open(FJoin(input_folder, file[0] + file[1]))
            for i, obj in enumerate(labeled_objs):
                obj_type = obj["type"]
                if obj_type != const.PICTURE:
                    obj_name = obj['name']
                    box_name = file[0] + '_' + str(i) + ".png"
                    obj_bbox = obj['bbox']  # xmin, ymin, xmax, ymax
                    xmin, ymin, xmax, ymax = obj_bbox[:]  # [str(coord) for coord in obj_bbox]
                    crop_img = remove_alpha(image.crop((xmin, ymin, xmax, ymax)))
                    is_save = False
                    save_folder = None
                    if xmin < xmax and ymin + thres_height < ymax:
                        if side == const.SIDE_FRONT:
                            if obj_type == const.NUMBER:
                                is_save = True
                                save_folder = front_id_folder
                                label_front_id[box_name] = obj_name
                            elif obj_type != const.LABEL:
                                is_save = True
                                save_folder = front_text_folder
                                label_front_text[box_name] = obj_name
                        elif side == const.SIDE_BACK and obj_type != const.LABEL:
                            is_save = True
                            save_folder = back_folder
                            label_back[box_name] = obj_name

                    if is_save:
                        temp_path_img = FJoin(temp_folder, box_name)
                        save_path_img = FJoin(save_folder, box_name)
                        try:
                            # neu save dc vao thu muc temp ma khong loi, thi moi save vao thu muc ket qua
                            crop_img.save(temp_path_img, format="PNG")
                            crop_img.save(save_path_img, format="PNG")
                        except TypeError:
                            print("Gap anh loi")
    # ghi label ra file json
    label_front_id_file = "label_{}_front_id{}.json".format(card_type, suffix)
    label_front_text_file = "label_{}_front_text{}.json".format(card_type, suffix)
    label_back_file = "label_{}_back{}.json".format(card_type, suffix)
    dict_name_file = {
        label_front_id_file: label_front_id,
        label_front_text_file: label_front_text,
        label_back_file: label_back
    }
    for label_file in [label_front_id_file, label_front_text_file, label_back_file]:
        with open(FJoin(label_folder, label_file), 'w', encoding='utf-8') as f:
            json.dump(dict_name_file[label_file], f, indent=4, ensure_ascii=False)
            f.close()


# chinh tang/giam do sang cho toan bo anh, tuy theo tham so gamma cao/thap
def adjust_gamma(imag, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((k / 255.0) ** inv_gamma) * 255 for k in np.arange(0, 256)]).astype("uint8")
    print("Done adjust_gamma()")
    return cv2.LUT(imag, table)


# bien doi nen xanh la thanh trang
def icomp(imag, average_inte):
    green_threshold = average_inte - 15
    black_threshold = green_threshold - 20

    def find_index():
        imag_g = imag[:, :, 1]
        imag_mean = np.mean(imag, axis=2)
        index_g = np.where(imag_g > green_threshold)  # tim cac pixel co tong mau 'G' lon hon nguong
        index_b = np.where(imag_mean > black_threshold)  # tim cac pixel co tong mau trung binh lon hon nguong
        lst1 = zip(index_g[0], index_g[1])  # ket noi thanh index
        lst2 = zip(index_b[0], index_b[1])  # ket noi thanh index
        index_total = list(set(lst1) & set(lst2))  # tim giao cua hai tap index
        return index_total

    res_img = imag.copy()
    index = find_index()
    for ind in index:
        res_img[ind] = [255, 255, 255]
    # print("Done icomp()")
    return res_img


# lay gia tri trung binh tren kenh 'G' cua tat ca pixels
def average(imag):
    g_channel = imag[:, :, 1]
    return np.mean(g_channel)


# bo loc giam nhieu muoi tieu
def blur(imag, kernel_size):
    print("Done blur()")
    return cv2.medianBlur(imag, kernel_size)


# bo loc lam sac net
def sharpen(imag):
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    print("Done sharpen()")
    return cv2.filter2D(imag, -1, kernel)


# phep bien doi rescale de lam tang do tuong phan
def rescale_minmax(imag):
    """
    # Scale theo min - max cua value_range
    # dau vao la 1 ma tran anh mau
    # dau ra la ma tran moi sau khi scale lai theo tung kenh mau va tong hop lai
    """
    imag_shape = imag.shape
    r_channel = imag[:, :, 0]
    g_channel = imag[:, :, 1]
    b_channel = imag[:, :, 2]
    img_min = [np.amin(r_channel, axis=None), np.amin(g_channel, axis=None), np.amin(b_channel, axis=None)]
    img_max = [np.amax(r_channel, axis=None), np.amax(g_channel, axis=None), np.amax(b_channel, axis=None)]

    dtype = np.uint8
    imag_rescale = np.zeros(imag_shape, dtype=dtype)

    for i in range(imag_shape[2]):  # 3 kenh mau RGB
        imag_rescale[:, :, i] = dtype(255 * ((imag[:, :, i] - img_min[i]) / (img_max[i] - img_min[i])))
    # print("Done rescale_minmax()")
    return imag_rescale


# can bang sang tren tung kenh RGB rieng roi tong hop lai
def equa_rgb(imag):
    imag_shape = imag.shape
    equa = imag.copy()
    r_channel = imag[:,:,0]
    g_channel = imag[:,:,1]
    b_channel = imag[:,:,2]
    equ_r = cv2.equalizeHist(r_channel)
    equ_g = cv2.equalizeHist(g_channel)
    equ_b = cv2.equalizeHist(b_channel)

    for i in range(imag_shape[0]):
        for j in range(imag_shape[1]):
            equa[i,j] = [equ_r[i,j], equ_g[i,j], equ_b[i,j]]

    print("Done equa_rgb()")
    return equa


# can bang sang tren kenh S/V cua he mau HSV roi tong hop lai
def equa_hsv(imag):
    imag = cv2.cvtColor(imag, cv2.COLOR_RGB2HSV)
    equ1 = imag.copy()  # can bang kenh S
    equ2 = imag.copy()  # can bang kenh V
    imag_shape = imag.shape

    s_channel = imag[:,:,1]
    equ_s = cv2.equalizeHist(s_channel)

    v_channel = imag[:,:,2]
    equ_v = cv2.equalizeHist(v_channel)

    for i in range(imag_shape[0]):
        for j in range(imag_shape[1]):
            equ1[i,j,1] = equ_s[i,j]
            equ2[i,j,2] = equ_v[i,j]
    equ1_rgb = cv2.cvtColor(equ1, cv2.COLOR_HSV2RGB)  # chuyen lai ve kenh RGB de ve
    equ2_rgb = cv2.cvtColor(equ2, cv2.COLOR_HSV2RGB)  # chuyen lai ve kenh RGB de ve
    # cv2.imwrite(FJoin(input_path, "equ1_RGB.jpg"), equ1_rgb)
    # cv2.imwrite(FJoin(input_path, "equ2_RGB.jpg"), equ2_rgb)
    print("Done equa_hsv()")
    return equ1_rgb, equ2_rgb


# phep "xoi mon" - noi lien cac vung o gan
def erosion(imag, kernel_size1, kernel_size2, iterations=1):
    kernel = np.ones((kernel_size1, kernel_size2), np.uint8)
    print("Done erosion()")
    return cv2.erode(imag, kernel, iterations=iterations)


# phep "gian no" - phong to cac chi tiet
def dilation(imag, kernel_size1, kernel_size2, iterations=1):
    kernel = np.ones((kernel_size1, kernel_size2), np.uint8)
    print("Done dilation()")
    return cv2.dilate(imag, kernel, iterations=iterations)


# phep "mo" - khu nhieu o ben ngoai doi tuong
def opening(imag, kernel_size1, kernel_size2, iterations=1):
    kernel = np.ones((kernel_size1, kernel_size2), np.uint8)
    print("Done opening()")
    return cv2.morphologyEx(imag, cv2.MORPH_OPEN, kernel, iterations=iterations)


# phep "dong" - khu nhieu o tren bien doi tuong
def closing(imag, kernel_size1, kernel_size2, iterations=1):
    kernel = np.ones((kernel_size1, kernel_size2), np.uint8)
    print("Done closing()")
    return cv2.morphologyEx(imag, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def generate_pre_process(input_path, input_labeled_file):
    """
    Pre-process for old id card
    -
    -
    :param input_path:
    :param input_labeled_file:
    :return:
    """
    result_folder = FJoin(input_path, "front_text")
    os.makedirs(result_folder, exist_ok=True)
    dict_filename_value = {}
    with open(input_labeled_file, 'r', encoding='utf-8') as js:
        json_data = json.load(js)

    image_files = dir_explore(input_path, const.EXPLORE_BROWSE)
    for i, file in enumerate(image_files):
        if i % 100 == 0:
            print("\nExecute for image: {}".format(file[0] + file[1]))
        image = cv2.imread(FJoin(input_path, file[0] + file[1]))
        average_intension = average(image)
        if i % 100 == 0:
            print("average intension for image '{}' is {}".format(file[0] + file[1], average_intension))
        # green_thres = 190 if average_intension >= 190 else 170 if 190 > average_intension >= 170 else 150

        # copy annotation
        gt_name = file[0] + '.gt.txt'

        rand_type = 0  # np.random.choice(range(2))
        if rand_type == 0:
            # if os.path.exists(FJoin(input_path, gt_name)):
            #     copyfile(FJoin(input_path, gt_name), FJoin(result_folder, file[0] + "_icre.gt.txt"))
            icomp_img = icomp(image, average_intension)
            icomp_rescale = rescale_minmax(icomp_img)
            new_filename = file[0] + "_icre" + file[1]
            cv2.imwrite(FJoin(result_folder, new_filename), icomp_rescale)
            # else:
            #     print("Except file is {}".format(file[0] + file[1]))
            if file[0] + file[1] in json_data.keys():
                dict_filename_value[new_filename] = json_data[file[0] + file[1]]
        elif rand_type == 1:
            if os.path.exists(FJoin(input_path, gt_name)):
                copyfile(FJoin(input_path, gt_name), FJoin(result_folder, file[0] + "_opshicbl.gt.txt"))
                opening_img = opening(image, 3, 3, iterations=1)
                opening_sharpen = sharpen(opening_img)
                opening_sharpen_icomp = icomp(opening_sharpen, average_intension)
                opening_sharpen_icomp_blur = blur(opening_sharpen_icomp, 3)
                cv2.imwrite(FJoin(result_folder, file[0] + "_opshicbl" + file[1]), opening_sharpen_icomp_blur)
            else:
                print("Except file is {}".format(file[0] + file[1]))

    with open(FJoin(result_folder, "label_cmnd_cu_front_text_filter_height_icre.json"), 'w', encoding='utf-8') as f:
        json.dump(dict_filename_value, f, indent=4, ensure_ascii=False)
        f.close()

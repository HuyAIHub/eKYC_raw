from PIL import Image, ImageStat
import cv2
import math
import numpy as np


def detect_color(img, card_side, thumb_size=40, mse_cutoff=20, adjust_color_bias=True):
    if card_side == "back":
        mse_cutoff =15
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    bands = pil_img.getbands()
    if bands == ('R', 'G', 'B') or bands == ('R', 'G', 'B', 'A'):
        thumb = pil_img.resize((thumb_size, thumb_size))
        sse, bias = 0, [0, 0, 0]
        if adjust_color_bias:
            bias = ImageStat.Stat(thumb).mean[:3]
            bias = [b - sum(bias) / 3 for b in bias]
        for pixel in thumb.getdata():
            mu = sum(pixel) / 3
            sse += sum((pixel[i] - mu - bias[i]) * (pixel[i] - mu - bias[i]) for i in [0, 1, 2])
        mse = float(sse) / (thumb_size * thumb_size)
        if mse <= mse_cutoff:
            return "gray"
        else:
            return "color"

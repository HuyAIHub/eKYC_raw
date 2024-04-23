import cv2
import glob


def check_blur(image, blur_threshold=100):
    """
    Check if image is blur or not
    :param image: numpy array
    :param blur_threshold: threshold to determine blur or not
    :return: true if image is blur, false if image is not blur
    """
    dim = (500, 850)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    if cv2.Laplacian(gray, cv2.CV_64F) is None:
        print("Laplacian count = 0")
        return False
    else:
        count = cv2.Laplacian(gray, cv2.CV_64F).var()
        return count < blur_threshold
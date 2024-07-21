import numpy as np
from google.colab.patches import cv2_imshow
import cv2


def computedifference(bg_img, input_img):
    difference_three_channel = cv2.absdiff(bg_img, input_img)
    difference_single_channel = np.sum(difference_three_channel, axis=2) / 3.0
    difference_single_channel = difference_single_channel.astype('uint8')

    return difference_single_channel


def computebinarymask(difference_single_channel):
    difference_binary = np.where(difference_single_channel >= 15, 255, 0)
    difference_binary = np.stack((difference_binary,)*3, axis=-1)
    return difference_binary


def replacebackground(bg1_image, bg2_image, ob_image):
    difference_single_channel = computedifference(bg1_image, ob_image)
    binary_mask = computebinarymask(difference_single_channel)

    output = np.where(binary_mask == 255, ob_image, bg2_image)

    return output


if __name__ == '__main__':
    bg1_image = cv2.imread('GreenBackground.png', 1)
    bg1_image = cv2.resize(bg1_image, (678, 381))

    ob_image = cv2.imread('Object.png', 1)
    ob_image = cv2.resize(ob_image, (678, 381))

    bg2_image = cv2.imread('NewBackground.jpg', 1)
    bg2_image = cv2.resize(bg2_image, (678, 381))

    difference_single_channel = computedifference(bg1_image, ob_image)
    cv2_imshow(difference_single_channel)

    binary_mask = computebinarymask(difference_single_channel)
    cv2_imshow(binary_mask)

    output = replacebackground(bg1_image, bg2_image, ob_image)
    cv2_imshow(output)

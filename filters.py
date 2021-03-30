import copy

import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline


def spread_lookup_table(x, y):
    spline = UnivariateSpline(x, y)
    return spline(range(256))


def create_loopup_tables():
    increase_lookup_table = spread_lookup_table([0, 64, 128, 256], [0, 80, 160, 256])
    decrease_lookup_table = spread_lookup_table([0, 64, 128, 256], [0, 50, 100, 256])
    
    return increase_lookup_table, decrease_lookup_table


def black_white(image):
    img = copy.deepcopy(image)

    output = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, output = cv2.threshold(output, 125, 255, cv2.THRESH_BINARY)
    output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

    return output


def invert(img):
    output = cv2.bitwise_not(copy.deepcopy(img))
    return output


def blur(img):
    blurred_image = cv2.blur(copy.deepcopy(img))
    return blurred_image


def sketch(img, kernel_size=21):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_invert = cv2.bitwise_not(img_gray)
    img_smoothing = cv2.GaussianBlur(img_invert, (kernel_size, kernel_size), sigmaX=0, sigmaY=0)

    def dodge(x, y):
        return cv2.divide(x, 255 - y, scale=256)

    output = dodge(img_gray, img_smoothing)
    return cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)


def sketch_with_edge_detection(img, kernel_size=21):
    img = np.copy(img)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_blur = cv2.GaussianBlur(img_gray, (kernel_size, kernel_size), 0)
    edges = cv2.Laplacian(img_gray_blur, cv2.CV_8U, ksize=5)
    edges = 255 - edges

    ret, edge_mask = cv2.threshold(edges, 150, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR)


def sharpen(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


def sepia(image):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    return cv2.filter2D(image, -1, kernel)


def gaussian_blur(image):
    return cv2.GaussianBlur(image, (35, 35), 0)


def emboss(image):
    kernel = np.array([[0, -1, -1],
                       [1, 0, -1],
                       [1, 1, 0]])
    return cv2.filter2D(image, -1, kernel)


def brightness_control(image, level):
    return cv2.convertScaleAbs(image, beta=level)


def image_2d_convolution(image):
    img = copy.deepcopy(image)

    kernel = np.ones((5, 5), np.float32) / 25
    return cv2.filter2D(img, -1, kernel)


def median_filtering(image):
    return cv2.medianBlur(image, 5)


def vignette(image, vignette_scale=2):
    img = np.copy(image)
    img = np.float32(img)

    rows, cols = img.shape[:2]

    k = np.min(img.shape[:2]) / vignette_scale
    kernel_x = cv2.getGaussianKernel(cols, k)
    kernel_y = cv2.getGaussianKernel(rows, k)
    kernel = kernel_y * kernel_x.T

    mask = 255 * kernel / np.linalg.norm(kernel)
    mask = cv2.GaussianBlur(mask, (51, 51), 0)

    img[:, :, 0] += img[:, :, 0] * mask
    img[:, :, 1] += img[:, :, 1] * mask
    img[:, :, 2] += img[:, :, 2] * mask

    img = np.clip(img / 2, 0, 255)
    return np.uint8(img)


def contrast(image, scale):
    img = np.copy(image)

    ycb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycb_img = np.float32(ycb_img)

    y_channel, Cr, Cb = cv2.split(ycb_img)
    y_channel = np.clip(y_channel * scale, 0, 255)

    ycb_img = np.uint8(cv2.merge([y_channel, Cr, Cb]))
    output = cv2.cvtColor(ycb_img, cv2.COLOR_YCrCb2BGR)
    return output


def saturation(image, saturation_scale=1):
    img = np.copy(image)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_img = np.float32(hsv_img)

    H, S, V = cv2.split(hsv_img)
    S = np.clip(S * saturation_scale, 0, 255)
    hsv_img = np.uint8(cv2.merge([H, S, V]))

    output = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return output


def warm(image):
    increase_lookup_table, decrease_lookup_table = create_loopup_tables()

    red_channel, green_channel, blue_channel = cv2.split(image)
    red_channel = cv2.LUT(red_channel, decrease_lookup_table).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increase_lookup_table).astype(np.uint8)

    return cv2.merge((red_channel, green_channel, blue_channel))


def cold(image):
    increase_lookup_table, decrease_lookup_table = create_loopup_tables()

    red_channel, green_channel, blue_channel = cv2.split(image)
    red_channel = cv2.LUT(red_channel, increase_lookup_table).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, decrease_lookup_table).astype(np.uint8)

    return cv2.merge((red_channel, green_channel, blue_channel))


def cartoon(img):
    img = np.copy(img)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)

    edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=5)
    edges = 255 - edges
    ret, edge_mask = cv2.threshold(edges, 150, 255, cv2.THRESH_BINARY)

    img_bilateral = cv2.edgePreservingFilter(img, flags=2, sigma_s=50, sigma_r=0.4)

    output = np.zeros(img_gray.shape)
    output = cv2.bitwise_and(img_bilateral, img_bilateral, mask=edge_mask)

    return output


def moon(image):
    img = np.copy(image)
    origin = np.array([0, 15, 30, 50, 70, 90, 120, 160, 180, 210, 255])
    _curve = np.array([0, 0, 5, 15, 60, 110, 150, 190, 210, 230, 255])

    full_range = np.arange(0, 256)

    _LUT = np.interp(full_range, origin, _curve)

    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_img[:, :, 0] = cv2.LUT(lab_img[:, :, 0], _LUT)

    img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
    img = saturation(img, 0.01)
    return img

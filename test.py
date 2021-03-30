import sys

import cv2

import filters

filters_list = [
    "black_white", "invert", "blur",
    "sketch", "sketch_with_edge_detection",
    "sharpen", "sepia", "gaussian_blur",
    "emboss", "image_2d_convolution",
    "median_filtering", "vignette", "warm",
    "cold", "cartoon", "moon"
]

if __name__ == '__main__':
    if len(sys.argv) not in [3, 4]:
        print("Usage: python test.py <FILTER> <IMAGE SRC> <IMAGE DESTINATION(OPTIONAL)>")
        sys.exit(0)

    if len(sys.argv) == 3:
        _, filter_name, src = sys.argv
        dest = None
    else:
        _, filter_name, src, dest = sys.argv

    filter_name = filter_name.lower()

    if filter_name not in filters_list:
        print("Invalid filter! Possible filters are" + "\n".join(filters_list))
        sys.exit(1)

    image = cv2.imread(src)

    edited_image = getattr(filters, filter_name)(image)

    if not dest:
        cv2.imwrite("edited.jpg", edited_image)
        print("Saved in the current directory as edited.jpg")
    else:
        cv2.imwrite(dest + "edited.jpg", edited_image)
        print(f"Saved at {dest} as edited.jpg")

import numpy as np
import numpy as np
import cv2


def is_image_file(filename):
    img_extension = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in img_extension)



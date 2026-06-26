from compression.codec import Codec

import numpy as np
import cv2

# dtype: uint8, uint16

class PNGCodec(Codec):

    def encode_image(self, image, out_file, dtype):

        if dtype == "uint8":
                image = image * 255
                image = image.astype("uint8")
        elif dtype == "uint16":
                image = image * 65535
                image = image.astype("uint16")
        else:
            raise f"image type {dtype} not supported!"

        cv2.imwrite(out_file, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        # cv2.imwrite(out_file, image)

    def decode_image(self, file_name):
        img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        if img.dtype == np.uint8:
                img = img / 255
        elif img.dtype == np.uint16:
                img = img / 65535
        else:
            raise f"image type {img.dtype} neither uint8 nor uint16!"
        return img

    def file_ending(self):
        return "png"
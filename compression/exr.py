from compression.codec import Codec

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import cv2


# parameters:
# type: ["half", "float"]
# compression: ["none", "rle", "zps", "zip", "piz", "pxr24", "b4a", "b44", "dwaa", "dwab"]

class EXRCodec(Codec):

    def encode_image(self, image, out_file, type="half", compression="none"):

        imwrite_flags = []

        if type == "half":
            imwrite_flags.extend([cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
        elif type == "float":
            imwrite_flags.extend([cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
        else:
            raise NotImplementedError(f"Unknown type: {type}")

        if compression == "rle":
            imwrite_flags.extend([cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_RLE])
        elif compression == "zps":
            imwrite_flags.extend([cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_ZIP])
        elif compression == "zip":
            imwrite_flags.extend([cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_ZIP])
        elif compression == "piz":
            imwrite_flags.extend([cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_PIZ])
        elif compression == "pxr24":
            imwrite_flags.extend([cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_PXR24])
        elif compression == "b4a":
            imwrite_flags.extend([cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_B44])
        elif compression == "b44":
            imwrite_flags.extend([cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_B44A])
        elif compression == "dwaa":
            imwrite_flags.extend([cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_DWAA])
        elif compression == "dwab":
            imwrite_flags.extend([cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_DWAB])
        elif compression == "none":
            pass
        else:
            raise NotImplementedError(f"Unknown compression method: {compression}")

        cv2.imwrite(out_file, image, imwrite_flags)

    def decode_image(self, file_name):
        return cv2.imread(file_name, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)

    def file_ending(self):
        return "exr"
import os
from argparse import ArgumentParser

import numpy as np
import cv2


if __name__ == '__main__':
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--datadir", type=str, required=True)
    args = parser.parse_args()

    if os.path.exists(args.datadir):
        for npy in os.listdir(args.datadir):
            img_idx = int(npy.split('.')[0])
            npy_path = os.path.join(args.datadir, npy)
            lang = np.load(npy_path)
            save_images = []
            for level in range(0, lang.shape[2], 3):
                _lang = (lang[:, :, level: level + 3] * 65535).astype(np.uint16)
                image = cv2.cvtColor(_lang, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(args.datadir, '{0:05d}'.format(img_idx) + f"_{level+1}" + ".png"), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])

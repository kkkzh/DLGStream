from abc import ABC

def normalize_img(img, min_val, max_val):

    # min_clipped_count = (img < min_val).sum()
    # max_clipped_count = (img > max_val).sum()
    # print(f"Clipped {(min_clipped_count + max_clipped_count) / img.size * 100}% of values")

    img = img.clip(min_val, max_val)
    img = (img - min_val) / (max_val - min_val)
    return img

# from print_ranges.py
min_thresholds = {
    "_scaling": -10,
    "_features_dc": 0,
    "_point_feat": -1.0,
    "_rotation": -1,
    "_opacity": -4,
}

max_thresholds = {
    "_scaling": 6,
    "_features_dc": 4,
    "_point_feat": 1.0,
    "_rotation": 2,
    "_opacity": 4,
}

class Codec(ABC):

    def encode_image(self, image, out_file, **kwargs):
        raise NotImplementedError("Subclasses should implement this!")

    def decode_image(self, file_name):
        raise NotImplementedError("Subclasses should implement this!")

    def file_ending(self):
        raise NotImplementedError("Subclasses should implement this!")

    def normalize_to_thresholds(self, img, attr_name, attr_min=None, attr_max=None, unuse_threshold=False):

        # normalize coordinates to 0...1
        if attr_name in ["_xyz", "_xyz_disp", "_xyz_dynamic"] or unuse_threshold:
            xyz_min = img.min()
            xyz_max = img.max()
            return normalize_img(img, xyz_min, xyz_max), xyz_min, xyz_max

        if attr_min is None:
            min_val = min_thresholds[attr_name]
        else:
            min_val = attr_min

        if attr_max is None:
            max_val = max_thresholds[attr_name]
        else:
            max_val = attr_max

        return normalize_img(img, min_val, max_val), min_val, max_val

    def read_file_bytes(self, file_path):
        with open(file_path, "rb") as f:
            return f.read()
    
    def write_file_bytes(self, file_path, bytes):
        with open(file_path, "wb") as f:
            f.write(bytes)

    def encode(self, image, out_file, **kwargs):
        self.encode_image(image, out_file, **kwargs)

    def decode(self, image):
        return self.decode_image(image)

    def encode_with_normalization(self, image, attr_name, out_file, attr_min=None, attr_max=None, unuse_threshold=False, **kwargs):
        img_norm, min_val, max_val = self.normalize_to_thresholds(image, attr_name, attr_min, attr_max, unuse_threshold)
        self.encode(img_norm, out_file, **kwargs)
        return min_val, max_val
    
    def decode_with_normalization(self, file_name, min_val, max_val):
        img_norm = self.decode(file_name)
        return img_norm * (max_val - min_val) + min_val


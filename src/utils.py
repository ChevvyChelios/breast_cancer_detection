import numpy as np

def get_mask_bbox(mask_img):
    mask = np.array(mask_img)
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    return (x_min, y_min, x_max, y_max)

import numpy as np


def gt_box_mnist(img, threshold=0):
    img = np.array(img)  # PIL -> numpy (28x28)
    ys, xs = np.where(img > threshold)
    return xs.min(), xs.max(), ys.min(), ys.max()


def pointing_game_hit(saliency, gt_box):
    saliency = np.array(saliency)
    y, x = np.unravel_index(np.argmax(saliency), saliency.shape)
    xmin, xmax, ymin, ymax = gt_box
    return int(xmin <= x <= xmax and ymin <= y <= ymax)

import numpy as np

# No MNIST não existem bounding boxes
# Usamos como ground truth a bounding box do dígito (pixels > threshold)
def gt_box_mnist(img, threshold=0):
    img = np.array(img)
    ys, xs = np.where(img > threshold)
    return xs.min(), xs.max(), ys.min(), ys.max()


def pointing_game_hit(saliency, gt_box):
    saliency = np.array(saliency)
    y, x = np.unravel_index(np.argmax(saliency), saliency.shape)
    xmin, xmax, ymin, ymax = gt_box
    return int(xmin <= x <= xmax and ymin <= y <= ymax)


def pointing_game_hit_topk(saliency, gt_box, k=10):
    h, w = saliency.shape
    xmin, xmax, ymin, ymax = gt_box

    idx = np.argsort(saliency.flatten())[-k:]

    for i in idx:
        y, x = divmod(i, w)
        if xmin <= x <= xmax and ymin <= y <= ymax:
            return 1
    return 0

def gt_mask_mnist(img, threshold=0):
    """ Ground-truth para MNIST como máscara binária (dígito = 1, fundo = 0).threshold=0 costuma funcionar em MNIST (fundo é 0)."""
    img = np.array(img)
    if img.ndim == 3:
        img = np.squeeze(img)
    return (img > threshold).astype(np.uint8)


def pointing_game_hit_mask(saliency, gt_mask):
    """Hit = 1 se o pixel de maior relevância estiver num pixel do dígito (mask==1). """
    saliency = np.nan_to_num(np.array(saliency))
    gt_mask = np.array(gt_mask)

    if saliency.ndim == 3:
        saliency = np.squeeze(saliency)
    if gt_mask.ndim == 3:
        gt_mask = np.squeeze(gt_mask)

    y, x = np.unravel_index(np.argmax(saliency), saliency.shape)
    return int(gt_mask[y, x] == 1)


def pointing_game_hit_topk_mask(saliency, gt_mask, k=3):
    """Hit@K = 1 se algum dos K pixels mais relevantes estiver num pixel do dígito. """
    saliency = np.nan_to_num(np.array(saliency))
    gt_mask = np.array(gt_mask)

    if saliency.ndim == 3:
        saliency = np.squeeze(saliency)
    if gt_mask.ndim == 3:
        gt_mask = np.squeeze(gt_mask)

    flat_sal = saliency.flatten()
    flat_mask = gt_mask.flatten()

    k = min(k, flat_sal.size)
    topk_idx = np.argpartition(flat_sal, -k)[-k:]
    return int(np.any(flat_mask[topk_idx] == 1))

def sparseness_gini(a, eps=1e-12):
    a = np.array(a, dtype=float).reshape(-1)
    a = np.abs(a)

    if a.sum() < eps:
        return 0.0

    a = np.sort(a + eps)
    n = a.size
    i = np.arange(1, n + 1)

    # fórmula do Gini para valores que não sejam negativos
    gini = (2 * (i * a).sum()) / (n * a.sum()) - (n + 1) / n
    return float(gini)




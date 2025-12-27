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


def complexity_components(saliency, q=0.9):
    s = np.abs(np.array(saliency, dtype=float))

    thr = np.quantile(s, q)

    mask = (s >= thr)

    H, W = mask.shape
    visited = np.zeros((H, W), dtype=bool)

    components = 0

    for r in range(H):
        for c in range(W):
            if mask[r, c] and not visited[r, c]:
                components += 1

                stack = [(r, c)]
                visited[r, c] = True

                while stack:
                    rr, cc = stack.pop()

                    # cima  
                    if rr > 0 and mask[rr - 1, cc] and not visited[rr - 1, cc]:
                        visited[rr - 1, cc] = True
                        stack.append((rr - 1, cc))

                    #baixo
                    if rr < H - 1 and mask[rr + 1, cc] and not visited[rr + 1, cc]:
                        visited[rr + 1, cc] = True
                        stack.append((rr + 1, cc))

                    # esquerda
                    if cc > 0 and mask[rr, cc - 1] and not visited[rr, cc - 1]:
                        visited[rr, cc - 1] = True
                        stack.append((rr, cc - 1))

                    # direita
                    if cc < W - 1 and mask[rr, cc + 1] and not visited[rr, cc + 1]:
                        visited[rr, cc + 1] = True
                        stack.append((rr, cc + 1))

    return components

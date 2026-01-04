import random
import numpy as np
import torch
from torchvision import datasets, transforms

from metrics_utils import (
    gt_box_mnist,
    pointing_game_hit,
    pointing_game_hit_topk,
    sparseness_gini,
    complexity_components,
)
from model import MNISTCNN
from explanations import (
    saliency_gradient,
    integrated_gradients,
    occlusion_map,
    guided_backprop,
    grad_cam
)


def main():
    # ---------- Reprodutibilidade (sedd) ----------
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # -------------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    N = 200
    q = 0.9
    K_LIST = [3]

    transform = transforms.ToTensor()

    # dataset para bbox (PIL) e para o modelo (tensor)
    data_pil = datasets.MNIST(root="data", train=False, download=False)
    data_tensor = datasets.MNIST(root="data", train=False, download=False, transform=transform)

    # amostra aleatória reprodutível
    indices = random.sample(range(len(data_tensor)), N)

    # Modelo
    model = MNISTCNN().to(device)
    model.load_state_dict(torch.load("mnist_cnn.pt", map_location=device))
    model.eval()

    # Acumuladores
    hits_g, hits_ig, hits_occ, hits_gb, hits_cam = 0, 0, 0, 0, 0
    spar_g, spar_ig, spar_occ, spar_gb, spar_cam = 0.0, 0.0, 0.0, 0.0, 0.0
    comp_g, comp_ig, comp_occ, comp_gb, comp_cam = 0.0, 0.0, 0.0, 0.0, 0.0

    # Pointing Game Top-K
    hits_topk = {
        "Gradiente": {k: 0 for k in K_LIST},
        "Integrated Gradients": {k: 0 for k in K_LIST},
        "Occlusion": {k: 0 for k in K_LIST},
        "Guided Backprop": {k: 0 for k in K_LIST},
        "Grad-CAM": {k: 0 for k in K_LIST},
    }

    for i in indices:
        img_pil, _ = data_pil[i]
        x, _ = data_tensor[i]
        x = x.unsqueeze(0).to(device)

        # Target = classe predita
        with torch.no_grad():
            logits = model(x)
            target = int(torch.argmax(logits, dim=1).item())

        gt_box = gt_box_mnist(img_pil)

        # ---------- Gradiente ----------
        saliency = saliency_gradient(model, x, target)
        saliency_norm = saliency / (saliency.max() + 1e-12)
        spar_g += sparseness_gini(saliency_norm)
        comp_g += complexity_components(saliency_norm, q)
        hits_g += pointing_game_hit(saliency, gt_box)

        for k in K_LIST:
            hits_topk["Gradiente"][k] += pointing_game_hit_topk(saliency, gt_box, k=k)
        # ---------- Integrated Gradients ----------
        saliency_ig = integrated_gradients(model, x, target, steps=20)
        saliency_ig_norm = saliency_ig / (saliency_ig.max() + 1e-12)
        spar_ig += sparseness_gini(saliency_ig_norm)
        comp_ig += complexity_components(saliency_ig_norm, q)

        hits_ig += pointing_game_hit(saliency_ig, gt_box)
        for k in K_LIST:
            hits_topk["Integrated Gradients"][k] += pointing_game_hit_topk(saliency_ig, gt_box, k=k)

        # ---------- Occlusion ----------
        saliency_occ = occlusion_map(model, x, target, patch=4)
        saliency_occ_norm = saliency_occ / (saliency_occ.max() + 1e-12)
        spar_occ += sparseness_gini(saliency_occ_norm)
        comp_occ += complexity_components(saliency_occ_norm, q)
        hits_occ += pointing_game_hit(saliency_occ, gt_box)

        for k in K_LIST:
            hits_topk["Occlusion"][k] += pointing_game_hit_topk(saliency_occ, gt_box, k=k)

        # ---------- Guided Backprop ----------
        saliency_gb = guided_backprop(model, x, target)
        saliency_gb_norm = saliency_gb / (saliency_gb.max() + 1e-12)
        spar_gb += sparseness_gini(saliency_gb_norm)
        comp_gb += complexity_components(saliency_gb_norm, q)
        hits_gb += pointing_game_hit(saliency_gb, gt_box)

        for k in K_LIST:
            hits_topk["Guided Backprop"][k] += pointing_game_hit_topk(saliency_gb, gt_box, k=k)

        # ---------- Grad-CAM ----------
        saliency_cam = grad_cam(model, x, target)
        saliency_cam_norm = saliency_cam / (saliency_cam.max() + 1e-12)
        spar_cam += sparseness_gini(saliency_cam_norm)
        comp_cam += complexity_components(saliency_cam_norm, q)
        hits_cam += pointing_game_hit(saliency_cam, gt_box)

        for k in K_LIST:
            hits_topk["Grad-CAM"][k] += pointing_game_hit_topk(saliency_cam, gt_box, k=k)


    print(f"\n--- Resultados (N={N}) ---")

    print(f"Gradiente | Pointing Game: {hits_g / N:.4f}")
    print(f"Gradiente | Sparseness: {spar_g / N:.4f}")
    print(f"Gradiente | Complexity: {comp_g / N:.4f}")

    print(f"\nIntegrated Gradients | Pointing Game: {hits_ig / N:.4f}")
    print(f"Integrated Gradients | Sparseness: {spar_ig / N:.4f}")
    print(f"Integrated Gradients | Complexity: {comp_ig / N:.4f}")

    print(f"\nOcclusion | Pointing Game: {hits_occ / N:.4f}")
    print(f"Occlusion | Sparseness: {spar_occ / N:.4f}")
    print(f"Occlusion | Complexity: {comp_occ / N:.4f}")

    print(f"\nGuided Backprop | Pointing Game: {hits_gb / N:.4f}")
    print(f"Guided Backprop | Sparseness: {spar_gb / N:.4f}")
    print(f"Guided Backprop | Complexity: {comp_gb / N:.4f}")

    print(f"\nGrad-CAM | Pointing Game: {hits_cam / N:.4f}")
    print(f"Grad-CAM | Sparseness: {spar_cam / N:.4f}")
    print(f"Grad-CAM | Complexity: {comp_cam / N:.4f}")

    k = K_LIST[0]
    print(f"\n--- Pointing Game Top-K (k={k}) ---")
    print(f"Gradiente | PG@{k}: {hits_topk['Gradiente'][k] / N:.4f}")
    print(f"Integrated Gradients | PG@{k}: {hits_topk['Integrated Gradients'][k] / N:.4f}")
    print(f"Occlusion | PG@{k}: {hits_topk['Occlusion'][k] / N:.4f}")
    print(f"Guided Backprop | PG@{k}: {hits_topk['Guided Backprop'][k] / N:.4f}")
    print(f"Grad-CAM | PG@{k}: {hits_topk['Grad-CAM'][k] / N:.4f}")


if __name__ == "__main__":
    main()

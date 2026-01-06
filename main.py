import random
import numpy as np
import torch
import csv
from torchvision import datasets, transforms

from metrics_utils import (
    gt_box_mnist,
    pointing_game_hit,
    pointing_game_hit_topk,
    sparseness_gini,
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
    # ---------- Reprodutibilidade (seed) ----------
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
    K_LIST = [3]

    transform = transforms.ToTensor()

    # dataset para bbox (PIL) e para o modelo (tensor)
    data_pil = datasets.MNIST(root="data", train=False, download=True)
    data_tensor = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    # proteger caso N > tamanho dataset
    N = min(N, len(data_tensor))

    # amostra aleatória reprodutível
    indices = random.sample(range(len(data_tensor)), N)

    # Modelo
    model = MNISTCNN().to(device)
    model.load_state_dict(torch.load("mnist_cnn.pt", map_location=device))
    model.eval()

    # Acumuladores
    hits_g, hits_ig, hits_occ, hits_gb, hits_cam = 0, 0, 0, 0, 0
    spar_g, spar_ig, spar_occ, spar_gb, spar_cam = 0.0, 0.0, 0.0, 0.0, 0.0

    results = {
        "Gradiente": {"pg": [], "spar": [], "pg_topk": {k: [] for k in K_LIST}},
        "Integrated Gradients": {"pg": [], "spar": [], "pg_topk": {k: [] for k in K_LIST}},
        "Occlusion": {"pg": [], "spar": [], "pg_topk": {k: [] for k in K_LIST}},
        "Guided Backprop": {"pg": [], "spar": [], "pg_topk": {k: [] for k in K_LIST}},
        "Grad-CAM": {"pg": [], "spar": [], "pg_topk": {k: [] for k in K_LIST}},
    }

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

        pg_val = pointing_game_hit(saliency, gt_box)
        spar_val = sparseness_gini(saliency_norm)

        hits_g += pg_val
        spar_g += spar_val

        results["Gradiente"]["pg"].append(pg_val)
        results["Gradiente"]["spar"].append(spar_val)

        for k in K_LIST:
            topk_val = pointing_game_hit_topk(saliency, gt_box, k=k)
            hits_topk["Gradiente"][k] += topk_val
            results["Gradiente"]["pg_topk"][k].append(topk_val)

        # ---------- Integrated Gradients ----------
        saliency_ig = integrated_gradients(model, x, target, steps=20)
        saliency_ig_norm = saliency_ig / (saliency_ig.max() + 1e-12)

        pg_val = pointing_game_hit(saliency_ig, gt_box)
        spar_val = sparseness_gini(saliency_ig_norm)

        hits_ig += pg_val
        spar_ig += spar_val

        results["Integrated Gradients"]["pg"].append(pg_val)
        results["Integrated Gradients"]["spar"].append(spar_val)

        for k in K_LIST:
            topk_val = pointing_game_hit_topk(saliency_ig, gt_box, k=k)
            hits_topk["Integrated Gradients"][k] += topk_val
            results["Integrated Gradients"]["pg_topk"][k].append(topk_val)

        # ---------- Occlusion ----------
        saliency_occ = occlusion_map(model, x, target, patch=4)
        saliency_occ_norm = saliency_occ / (saliency_occ.max() + 1e-12)

        pg_val = pointing_game_hit(saliency_occ, gt_box)
        spar_val = sparseness_gini(saliency_occ_norm)

        hits_occ += pg_val
        spar_occ += spar_val

        results["Occlusion"]["pg"].append(pg_val)
        results["Occlusion"]["spar"].append(spar_val)

        for k in K_LIST:
            topk_val = pointing_game_hit_topk(saliency_occ, gt_box, k=k)
            hits_topk["Occlusion"][k] += topk_val
            results["Occlusion"]["pg_topk"][k].append(topk_val)

        # ---------- Guided Backprop ----------
        saliency_gb = guided_backprop(model, x, target)
        saliency_gb_norm = saliency_gb / (saliency_gb.max() + 1e-12)

        pg_val = pointing_game_hit(saliency_gb, gt_box)
        spar_val = sparseness_gini(saliency_gb_norm)

        hits_gb += pg_val
        spar_gb += spar_val

        results["Guided Backprop"]["pg"].append(pg_val)
        results["Guided Backprop"]["spar"].append(spar_val)

        for k in K_LIST:
            topk_val = pointing_game_hit_topk(saliency_gb, gt_box, k=k)
            hits_topk["Guided Backprop"][k] += topk_val
            results["Guided Backprop"]["pg_topk"][k].append(topk_val)

        # ---------- Grad-CAM ----------
        saliency_cam = grad_cam(model, x, target)
        saliency_cam_norm = saliency_cam / (saliency_cam.max() + 1e-12)

        pg_val = pointing_game_hit(saliency_cam, gt_box)
        spar_val = sparseness_gini(saliency_cam_norm)

        hits_cam += pg_val
        spar_cam += spar_val

        results["Grad-CAM"]["pg"].append(pg_val)
        results["Grad-CAM"]["spar"].append(spar_val)

        for k in K_LIST:
            topk_val = pointing_game_hit_topk(saliency_cam, gt_box, k=k)
            hits_topk["Grad-CAM"][k] += topk_val
            results["Grad-CAM"]["pg_topk"][k].append(topk_val)


    print(f"\n--- Resultados (N={N}) ---")

    print(f"Gradiente | Pointing Game: {hits_g / N:.4f}")
    print(f"Gradiente | Sparseness: {spar_g / N:.4f}")

    print(f"\nIntegrated Gradients | Pointing Game: {hits_ig / N:.4f}")
    print(f"Integrated Gradients | Sparseness: {spar_ig / N:.4f}")

    print(f"\nOcclusion | Pointing Game: {hits_occ / N:.4f}")
    print(f"Occlusion | Sparseness: {spar_occ / N:.4f}")

    print(f"\nGuided Backprop | Pointing Game: {hits_gb / N:.4f}")
    print(f"Guided Backprop | Sparseness: {spar_gb / N:.4f}")

    print(f"\nGrad-CAM | Pointing Game: {hits_cam / N:.4f}")
    print(f"Grad-CAM | Sparseness: {spar_cam / N:.4f}")

    k = K_LIST[0]
    print(f"\n--- Pointing Game Top-K (k={k}) ---")
    print(f"Gradiente | PG@{k}: {hits_topk['Gradiente'][k] / N:.4f}")
    print(f"Integrated Gradients | PG@{k}: {hits_topk['Integrated Gradients'][k] / N:.4f}")
    print(f"Occlusion | PG@{k}: {hits_topk['Occlusion'][k] / N:.4f}")
    print(f"Guided Backprop | PG@{k}: {hits_topk['Guided Backprop'][k] / N:.4f}")
    print(f"Grad-CAM | PG@{k}: {hits_topk['Grad-CAM'][k] / N:.4f}")

    # ---------------- Exportar resultados ----------------
    # médias por método
    summary_path = "results_summary.csv"
    with open(summary_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metodo", "pointing_game_mean", "sparseness_mean"] + [f"pg@{k}_mean" for k in K_LIST])

        for metodo, vals in results.items():
            pg_mean = float(np.mean(vals["pg"])) if vals["pg"] else 0.0
            spar_mean = float(np.mean(vals["spar"])) if vals["spar"] else 0.0
            topk_means = [float(np.mean(vals["pg_topk"][k])) if vals["pg_topk"][k] else 0.0 for k in K_LIST]
            writer.writerow([metodo, pg_mean, spar_mean] + topk_means)

    # Resultados por amostra
    per_sample_path = "results_per_sample.csv"
    with open(per_sample_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metodo", "sample_idx", "pointing_game", "sparseness"] + [f"pg@{k}" for k in K_LIST])

        for metodo, vals in results.items():
            n = len(vals["pg"])
            for j in range(n):
                row = [
                          metodo,
                          j,
                          int(vals["pg"][j]),
                          float(vals["spar"][j]),
                      ] + [int(vals["pg_topk"][k][j]) for k in K_LIST]
                writer.writerow(row)

    print(f"\nFicheiros guardados: {summary_path} e {per_sample_path}")

    # ---------------- Visualizações ----------------
    try:
        from visualize import generate_visualizations

        print("\n" + "=" * 60)
        print("Gerar Visualizações e Gráficos")
        print("=" * 60)

        generated = generate_visualizations(model, device, num_samples=3)

        if generated:
            print("\n Todos os elementos foram criados com sucesso!")
            print("   (Gráficos + Visualizações)")
        else:
            print("\n Alguns elementos podem não ter sido criados.")

    except ImportError:
        print("\n Ficheiro visualize.py não encontrado")
        print("   As visualizações não foram criadas.")
    except Exception as e:
        print(f"\n Erro: {e}")

    print("\n" + "=" * 60)
    print(" Execução Concluída ")
    print("=" * 60)

if __name__ == "__main__":
    main()

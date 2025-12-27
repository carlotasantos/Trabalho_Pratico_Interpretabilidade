import torch
from torchvision import datasets, transforms

from metrics_utils import gt_box_mnist, pointing_game_hit, sparseness_gini, complexity_components
from model import MNISTCNN
from explanations import saliency_gradient, integrated_gradients


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    N = 200
    q = 0.9

    transform = transforms.ToTensor()

    # dataset para bbox (PIL) e para o modelo (tensor)
    data_pil = datasets.MNIST(root="data", train=True, download=False)
    data_tensor = datasets.MNIST(root="data", train=True, download=False, transform=transform)

    # modelo treinado
    model = MNISTCNN().to(device)
    model.load_state_dict(torch.load("mnist_cnn.pt", map_location=device))
    model.eval()

    # acumuladores
    hits_g, spar_g, comp_g = 0, 0.0, 0
    hits_ig, spar_ig, comp_ig = 0, 0.0, 0

    for i in range(N):
        img_pil, _ = data_pil[i]
        x, _ = data_tensor[i]
        x = x.unsqueeze(0).to(device)

        gt_box = gt_box_mnist(img_pil)

        # ---------- Gradiente Simples ----------
        # ---------- Gradiente Simples ----------
        logits = model(x)
        y_pred = logits.argmax(1).item()

        saliency = saliency_gradient(model, x, y_pred)
        saliency_norm = saliency / (saliency.max() + 1e-12)

        hits_g += pointing_game_hit(saliency, gt_box)
        spar_g += sparseness_gini(saliency_norm)
        comp_g += complexity_components(saliency_norm, q)

        # ---------- Integrated Gradients ----------
        # ---------- Integrated Gradients ----------
        saliency_ig = integrated_gradients(model, x, y_pred, steps=20)
        saliency_ig_norm = saliency_ig / (saliency_ig.max() + 1e-12)

        hits_ig += pointing_game_hit(saliency_ig, gt_box)
        spar_ig += sparseness_gini(saliency_ig_norm)
        comp_ig += complexity_components(saliency_ig_norm, q)

    print(f"\n== Resultados (N={N}) ==")

    print(f"Gradiente | Pointing Game: {hits_g / N:.4f}")
    print(f"Gradiente | Sparseness: {spar_g / N:.4f}")
    print(f"Gradiente | Complexity: {comp_g / N:.4f}")

    print(f"\nIntegrated Gradients | Pointing Game: {hits_ig / N:.4f}")
    print(f"Integrated Gradients | Sparseness: {spar_ig / N:.4f}")
    print(f"Integrated Gradients | Complexity: {comp_ig / N:.4f}")


if __name__ == "__main__":
    main()

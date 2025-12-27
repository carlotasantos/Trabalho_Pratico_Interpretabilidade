import torch
from torchvision import datasets, transforms

from metrics_utils import gt_box_mnist, pointing_game_hit, sparseness_gini, complexity_components
from model import MNISTCNN


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
        x_g = x.clone().detach().requires_grad_(True)

        logits = model(x_g)
        y_pred = logits.argmax(1).item()
        score = logits[0, y_pred]

        model.zero_grad()
        score.backward()

        saliency = x_g.grad.abs().squeeze().cpu().numpy()
        saliency_norm = saliency / (saliency.max() + 1e-12)

        hits_g += pointing_game_hit(saliency, gt_box)
        spar_g += sparseness_gini(saliency_norm)
        comp_g += complexity_components(saliency_norm, q)

        # ---------- Integrated Gradients ----------
        baseline = torch.zeros_like(x)
        grads_sum = torch.zeros_like(x)

        for k in range(1, 21):  # 20 passos
            alpha = k / 20
            x_int = baseline + alpha * (x - baseline)
            x_int.requires_grad_()

            out = model(x_int)
            score_int = out[0, y_pred]

            grad = torch.autograd.grad(score_int, x_int)[0]
            grads_sum += grad

        ig = (x - baseline) * (grads_sum / 20)

        saliency_ig = ig.abs().squeeze().cpu().numpy()
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

import torch
from torchvision import datasets, transforms
from metrics_utils import gt_box_mnist, pointing_game_hit, sparseness_gini, complexity_components

from model import MNISTCNN


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    N = 200  # nº de imagens que avalia

    transform = transforms.ToTensor()

    # dataset para bbox
    train_pil = datasets.MNIST(root="data", train=True, download=False)

    # dataset para o modelo
    train_tensor = datasets.MNIST(root="data", train=True, download=False, transform=transform)

    # modelo treinado
    model = MNISTCNN().to(device)
    model.load_state_dict(torch.load("mnist_cnn.pt", map_location=device))
    model.eval()

    hits = 0
    spar_sum = 0.0
    comp_sum = 0

    for i in range(N):
        img_pil, y_true = train_pil[i]
        x, _ = train_tensor[i]
        x = x.unsqueeze(0).to(device)

        # saliency por gradiente
        x.requires_grad_()
        logits = model(x)
        y_pred = logits.argmax(1).item()
        score = logits[0, y_pred]

        model.zero_grad()
        score.backward()

        saliency = x.grad.abs().squeeze().cpu().numpy()
        s_min = saliency.min()
        s_max = saliency.max()
        saliency_norm = (saliency - s_min) / (s_max - s_min+1e-12)

        spar_sum += sparseness_gini(saliency_norm)
        comp_sum += complexity_components(saliency_norm, q=0.9)


        # pointing game
        gt_box = gt_box_mnist(img_pil)
        hit = pointing_game_hit(saliency, gt_box)
        hits += hit

    score_pg = hits / N
    print(f"Pointing Game score (N={N}): {score_pg:.4f}")

    spar_mean = spar_sum / N
    print(f"Sparseness Gini (N={N}): {spar_mean:.4f}")

    comp_mean = comp_sum / N
    print(f"Complexity Components (N={N}): {comp_mean:.4f}")


if __name__ == "__main__":
    main()

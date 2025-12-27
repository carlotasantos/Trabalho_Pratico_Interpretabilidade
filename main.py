import torch
from torchvision import datasets, transforms
from model import MNISTCNN

from metrics_utils import gt_box_mnist, pointing_game_hit



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    transform = transforms.ToTensor()

    train_pil = datasets.MNIST(root="data", train=True, download=False)  # sem transform
    img_pil, y_true = train_pil[0]

    train_tensor = datasets.MNIST(root="data", train=True, download=False, transform=transform)
    x, _ = train_tensor[0]
    x = x.unsqueeze(0).to(device)

    model = MNISTCNN().to(device)
    model.load_state_dict(torch.load("mnist_cnn.pt", map_location=device))
    model.eval()


    #predição
    #permite gradientes na imagem
    x.requires_grad_()

    logits = model(x)
    y_pred = logits.argmax(1).item()
    score = logits[0, y_pred]

    model.zero_grad()
    score.backward()

    # gradiente da imagem
    saliency = x.grad.abs().squeeze().cpu().numpy()

    gt_box = gt_box_mnist(img_pil)
    hit = pointing_game_hit(saliency, gt_box)


    print("Label verdadeira:", y_true)
    print("Predição do modelo:", y_pred)
    print("Saliency shape:", saliency.shape)
    print("Saliency min/max:", saliency.min(), saliency.max())
    print("GT box:", gt_box)
    print("Pointing Game hit:", hit)

if __name__ == "__main__":
    main()

import torch
from torchvision import datasets, transforms

from model import MNISTCNN


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    transform = transforms.ToTensor()
    train = datasets.MNIST(root="data", train=True, download=False, transform=transform)

    x, y_true = train[0]
    x = x.unsqueeze(0).to(device)

    # carrega o  modelo treinado
    model = MNISTCNN().to(device)
    model.load_state_dict(torch.load("mnist_cnn.pt", map_location=device))
    model.eval()

    # predição
    with torch.no_grad():
        logits = model(x)
        y_pred = logits.argmax(1).item()

    print("Label verdadeira:", y_true)
    print("Predição do modelo:", y_pred)


if __name__ == "__main__":
    main()

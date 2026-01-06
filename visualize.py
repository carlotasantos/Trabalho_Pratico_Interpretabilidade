import numpy
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import random

from model import MNISTCNN
from  explanations import (
    saliency_gradient,
    integrated_gradients,
    occlusion_map,
    guided_backprop,
    grad_cam
)

def set_seed(seed=42):
    """Define a seed para reprodutibilidade."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_sample(sample_idx, device="cpu"):
    """Carrega uma amostra especifica do dataset."""
    transform = transforms.ToTensor()

    data_pil = datasets.MNIST(root="data", train=False, download=True)
    data_tensor = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    img_pil, true_label = data_pil[sample_idx]
    x, _= data_tensor[sample_idx]
    x = x.unsqueeze(0).to(device)

    return img_pil, x, true_label

def compute_all_maps(model, x, target_class, device="cpu"):
    """Calcula todos os mapas de saliência para uma amostra."""
    maps = {}

    try:
        maps['Gradiente'] = saliency_gradient(model, x, target_class)
        maps['Integrated Gradients'] = integrated_gradients(model, x, target_class, steps=20)
        maps['Occlusion'] = occlusion_map(model, x, target_class, patch=4)
        maps['Guided Backprop'] = guided_backprop(model, x, target_class)
        maps['Grad-CAM'] = grad_cam(model, x, target_class)

    except Exception as e:
        print(f"Erro ao calcular mapas de saliência: {e}")

    return maps


def visualize_sample(img_pil, maps, sample_idx, true_label, pred_label, save_path=None):
    """ Cria uma figura com todos os maps de saliência para uma amostra. """
    method_order = [
        "Gradiente",
        "Integrated\nGradients",
        "Occlusion",
        "Guided\nBackprop",
        "Grad-CAM"
    ]

    fig, axes = plt.subplots(2, 6, figsize=(22, 9))

    fig.suptitle(f"VISUALIZAÇÃO - AMOSTRA {sample_idx}",
                 fontsize=18, fontweight='bold', y=0.98)

    fig.text(0.5, 0.94, f"Verdadeiro: {true_label}  |  Previsto: {pred_label}",
             ha='center', fontsize=14, fontweight='bold')

    # Imagem original
    axes[0, 0].imshow(img_pil, cmap='gray')
    axes[0, 0].set_title("IMAGEM\nORIGINAL", fontweight='bold', fontsize=12, pad=12)
    axes[0, 0].axis('off')

    axes[1, 0].imshow(img_pil, cmap='gray')
    axes[1, 0].axis('off')


    for col, method_name in enumerate(method_order, start=1):
        method_key = method_name.replace("\n", " ")  # Para aceder ao dicionário

        if method_key in maps:
            saliency_map = maps[method_key]

            # Normalizar
            if saliency_map.max() > 0:
                saliency_map = saliency_map / saliency_map.max()

            #Mapa puro
            im1 = axes[0, col].imshow(saliency_map, cmap='hot')
            axes[0, col].set_title(f"{method_name}\n(MAPA)",
                                   fontsize=11, fontweight='bold', pad=10)
            axes[0, col].axis('off')

            # Overlay
            axes[1, col].imshow(img_pil, cmap='gray')
            im2 = axes[1, col].imshow(saliency_map, cmap='hot', alpha=0.6)
            axes[1, col].set_title(f"{method_name}\n(OVERLAY)",
                                   fontsize=11, pad=10)
            axes[1, col].axis('off')

        else:
            axes[0, col].text(0.5, 0.5, "ERRO", ha='center', va='center',
                              fontsize=14, color='red', fontweight='bold')
            axes[0, col].set_title(method_name, fontsize=11)
            axes[0, col].axis('off')

            axes[1, col].text(0.5, 0.5, "ERRO", ha='center', va='center',
                              fontsize=14, color='red')
            axes[1, col].axis('off')

    plt.subplots_adjust(top=0.90, bottom=0.05, left=0.05, right=0.95,
                        hspace=0.25, wspace=0.35)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"{save_path}")
    else:
        plt.show()

    plt.close()

def generate_visualizations(model, device, num_samples=3):
    """ Função principal de visualização que pode ser chamada do main.py. """
    print("\n" + "="*60)
    print("Gerar Visualizações")
    print("="*60)

    set_seed(42)

    test_set_size = 10000
    sample_indices = random.sample(range(test_set_size), num_samples)

    print(f"Visualizar {num_samples} amostras: {sample_indices}")
    print("-"*60)

    generated_images = []

    for i, sample_idx in enumerate(sample_indices):
        print(f"\n Amostra {i + 1}/{num_samples} (índice {sample_idx})")

        img_pil, x, true_label = load_sample(sample_idx, device)

        with torch.no_grad():
            logits = model(x)
            pred_label = torch.argmax(logits, dim=1).item()

        print(f" Verdadeiro: {true_label}, Previsto: {pred_label}")

        maps = compute_all_maps(model, x, pred_label, device)

        if maps:
            filename = f"visualization_sample_{i + 1}.png"
            visualize_sample(img_pil, maps, sample_idx, true_label, pred_label, filename)
            generated_images.append(filename)
        else:
            print(f"Nenhum mapa calculado")

    print("\n" + "="*60)
    print(f"{len(generated_images)} visualizações geradas")
    print("="*60)

    return generated_images


if __name__ == "__main__":
    print("A executar visualize.py")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")

    model = MNISTCNN().to(device)
    model.load_state_dict(torch.load("mnist_cnn.pt", map_location=device))
    model.eval()

    generate_visualizations(model, device, num_samples=3)


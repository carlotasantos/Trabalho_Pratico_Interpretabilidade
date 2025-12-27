from torchvision import datasets
from metrics_utils import gt_box_mnist

# carregar dataset (já descarregado)
train = datasets.MNIST(root="data", train=True, download=False)

img, label = train[0]
box = gt_box_mnist(img)

print("Label:", label)
print("GT box:", box)

from torchvision import datasets

# Faz download e guarda em: projeto/data/MNIST/...
datasets.MNIST(root="data", train=True, download=True)
datasets.MNIST(root="data", train=False, download=True)

print("MNIST descarregado para a pasta: data/MNIST")

train = datasets.MNIST(root="data", train=True, download=False)
print(len(train), train[0][0].size, train[0][1])


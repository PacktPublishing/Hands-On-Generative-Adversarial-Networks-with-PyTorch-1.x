import torch

print("PyTorch version: {}".format(torch.__version__))
print("CUDA version: {}".format(torch.version.cuda))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

a = torch.randn(1, 10).to(device)
b = torch.randn(10, 1).to(device)
c = a @ b
print(c.item())

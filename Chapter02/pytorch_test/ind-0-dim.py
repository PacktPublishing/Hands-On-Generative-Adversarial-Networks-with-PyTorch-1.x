import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

a = torch.randn(10, 10).to(device)
b = torch.zeros(10, dtype=torch.long).to(device)
a = F.log_softmax(a, dim=1)

c = F.nll_loss(a, b, reduction='sum')

print(c.item()) # It's fine.
print(c[0])     # Will raise error.

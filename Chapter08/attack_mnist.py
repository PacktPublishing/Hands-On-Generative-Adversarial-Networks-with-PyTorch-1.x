import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms


print("PyTorch version: {}".format(torch.__version__))
print("CUDA version: {}\n".format(torch.version.cuda))

use_cuda = True
device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")

epsilons = [.05, .1, .15, .2, .25, .3]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = nn.Linear(800, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.fc1(x.view(-1, 800)))
        x = self.fc2(x)
        # return F.log_softmax(x, dim=1)
        return x

batch_size = 64
train_data = datasets.MNIST('/home/john/Data/mnist', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                # transforms.Normalize((0.1307,), (0.3081,)),
                                ]))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           shuffle=True, pin_memory=True)

test_data = datasets.MNIST('/home/john/Data/mnist', train=False, download=True,
                           transform=transforms.Compose([
                                transforms.ToTensor(),
                                # transforms.Normalize((0.1307,), (0.3081,)),
                                ]))
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000,
                                          shuffle=False, pin_memory=True)

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=3e-5)
# criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()

def train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 250 == 0:
            print('[{}/{}]\tLoss: {:.6f}'.format(
                batch_idx * batch_size, len(train_data), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader)
    print('\nTest loss: {:.4f}, accuracy: {:.4f}%\n'.format(
        test_loss, 100. * correct / len(test_data)))

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def adv_test(model, device, test_loader, epsilon):
    model.eval()
    correct = 0
    adv_examples = []
    #* grads of params are needed
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        init_pred = init_pred.view_as(target)
        loss = criterion(output, target)
        model.zero_grad()
        loss.backward()

        perturbed_data = fgsm_attack(data, epsilon, data.grad.data)
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]
        # final_pred has shape [1000, 1], target has shape [1000]. Must reshape final_pred
        final_pred = final_pred.view_as(target)
        correct += final_pred.eq(target).sum().item()
        if len(adv_examples) < 5 and not (final_pred == target).all():
            indices = torch.ne(final_pred.ne(target), init_pred.ne(target)).nonzero()
            # indices = torch.arange(5)
            for i in range(indices.shape[0]):
                adv_ex = perturbed_data[indices[i]].squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred[indices[i]].item(), final_pred[indices[i]].item(), adv_ex))
                if (len(adv_examples) >= 5):
                    break
    final_acc = 100. * correct / len(test_data)
    print("Epsilon: {}\tTest Accuracy = {}/{} = {:.4f}".format(
        epsilon, correct, len(test_data), final_acc))
    return final_acc, adv_examples

# Train
model.train()
for epoch in range(5):
    print('Train Epoch: {}'.format(epoch))
    train(model, device, train_loader, optimizer)
    test(model, device, test_loader)

accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex = adv_test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)

# Plot several examples of adversarial samples at each epsilon
cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig,adv,ex = examples[i][j]
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()

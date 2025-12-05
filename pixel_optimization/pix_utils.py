import numpy as np
from torch.utils.data import DataLoader, Dataset
import time
import torch
class Synthetic(Dataset):
    def __init__(self, data, targets):
        self.data = data.detach().float()
        self.targets = targets.detach()

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.data.shape[0]
    
def get_images(indices_class, images_all, c, n): # get random n images from class c
    idx_shuffle = np.random.permutation(indices_class[c])[:n]
    return images_all[idx_shuffle]


def get_loops(ipc):
    if ipc == 1:
        outer_loop, inner_loop = 1, 1
    elif ipc == 10:
        outer_loop, inner_loop = 10, 50
    elif ipc == 20:
        outer_loop, inner_loop = 20, 25
    elif ipc == 30:
        outer_loop, inner_loop = 30, 20
    elif ipc == 40:
        outer_loop, inner_loop = 40, 15
    elif ipc == 50:
        outer_loop, inner_loop = 50, 10
    else:
        outer_loop, inner_loop = 0, 0
        exit('loop hyper-parameters are not defined for %d ipc'%ipc)
    return outer_loop, inner_loop

def update_model(optimizer, steps, loss_function, net, syn_data_data, syn_data_target):
        for s in range(steps):
            net.train()
            prediction_syn = net(syn_data_data)
            loss_syn = loss_function(prediction_syn, syn_data_target)
            optimizer.zero_grad()
            loss_syn.backward()
            optimizer.step()


def evaluate(args, synthetic_datas, testloader, batch_size, ipc, num_train_epochs, n_classes, device):
    from common import define_model

    start_time = time.time()

    accuracies = []
    targets_syn = torch.tensor([np.ones(ipc) * i for i in range(n_classes)], dtype=torch.long, requires_grad=False,  device=device).view(-1)
    
    for data_syn in synthetic_datas:
        loss_fn = torch.nn.CrossEntropyLoss().to(device)
        net = define_model(args, n_classes, e_model=args.eval_model[0]).to(device)
        net.train()

        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        optimizer.zero_grad()

        syn_dataset = Synthetic(data_syn, targets_syn)
        trainloader = DataLoader(syn_dataset, batch_size=batch_size, shuffle=True)

        for it in range(num_train_epochs):

            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)

                prediction = net(images)
                loss = loss_fn(prediction, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        end_time = time.time()
        elapsed_time = end_time - start_time

        net.eval()
        with torch.inference_mode():
            correct = 0
            total = 0
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total

        accuracies.append(accuracy)

    return sum(accuracies) / len(accuracies), elapsed_time
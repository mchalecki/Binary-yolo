import argparse
import random
import torch
from torch import optim

from torch.utils.data.dataset import Subset
from torchvision import transforms

from dataset import YoloDataset, ToTensor
from network import Model

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--test_size', help="Percentage of dataset to be test set", type=float, default=0.1)
parser.add_argument('--epochs', help="Number of epochs", type=int, default=100)
parser.add_argument('--weightsfile', help="File with weights", type=str, default='./yolov3.weights')

CUDA = torch.cuda.is_available()
c_orrd = 1
c_noobj = 1


def train(args):
    yolo_dataset = YoloDataset('./raw_dataset/raw_dataset/faces.csv', './raw_dataset/raw_dataset',
                               transform=transforms.Compose([
                                   ToTensor()
                               ]))
    dataset_indices = set([i for i in range(len(yolo_dataset))])
    test_indices = random.sample(dataset_indices, int(args.test_size * len(yolo_dataset)))
    train_indices = list(dataset_indices - set(test_indices))
    trainset = Subset(yolo_dataset, train_indices)
    testset = Subset(yolo_dataset, train_indices)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=4)
    validloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=4)
    dataloaders = {
        "train": trainloader, "val": validloader
    }
    model = Model()
    if CUDA:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(args.epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                # scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            for i, data in enumerate(trainloader):
                img, y = data['img'], data['y']
                if CUDA:
                    img = img.cuda()
                    y = y.cuda()
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    pred = model(img)

                    y_ = pred.detach()
                    y_ = y_.permute((0, 2, 3, 1))
                    y = y.permute((0, 2, 3, 1))
                    loss = loss_function(y_, y)  # todo it may happen that loss in nan :/
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                print(f"Epoch: {epoch}, batch: {i}, loss {loss.item()}")


def loss_function(y_, y):
    loss = torch.zeros(1, requires_grad=True)
    if CUDA:
        loss = loss.cuda()
    for predicted_single_example, target_single_example in zip(y_, y):
        for predicted_row, target_row in zip(predicted_single_example, target_single_example):
            for predited_cell, target_cell in zip(predicted_row, target_row):
                if target_cell[0] == 0:  # there should be object
                    loss += c_noobj * predited_cell[0] ** 2
                else:
                    loss += c_orrd * (predited_cell[0] - target_cell[0]) ** 2  # p
                    loss += c_orrd * (predited_cell[1] - target_cell[1]) ** 2 + (
                            predited_cell[2] - target_cell[2]) ** 2  # x y
                    loss += (predited_cell[3] - target_cell[3]) ** 2 + (predited_cell[4] - target_cell[4]) ** 2  # wh
    return loss


if __name__ == '__main__':
    args = parser.parse_args()
    train(args)

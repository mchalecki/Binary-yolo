import argparse
import random
import torch
from torch import optim
from torch.optim import lr_scheduler

from torch.utils.data.dataset import Subset
from torchvision import transforms

from dataset import YoloDataset, ToTensor
from network import CustomModel, MyResnet
from utils.saver import CheckpointSaver
from utils.stats import AVG

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--test_size', help="Percentage of dataset to be test set", type=float, default=0.1)
parser.add_argument('--epochs', help="Number of epochs", type=int, default=200)
parser.add_argument('--weightsfile', help="File with weights", type=str, default='./yolov3.weights')
parser.add_argument('--lr', help="Learning rate", type=int, default=0.001)
parser.add_argument('--save_dir_name', help="Where to save checkpoints.", type=str, default="./saves")

CUDA = torch.cuda.is_available()
c_orrd = 6
c_noobj = .5


def train(args):
    yolo_dataset = YoloDataset('./raw_dataset/raw_dataset/faces.csv', './raw_dataset/raw_dataset', IN=195,
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
    model = MyResnet()
    if CUDA:
        model.cuda()

    optimizer = optim.Adam(model.conv_addition.parameters(), lr=args.lr)
    saver = CheckpointSaver(args.save_dir_name, max_checkpoints=3)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    epoch_losses = []
    for epoch in range(args.epochs):
        epoch_avg = AVG()
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            for i, data in enumerate(dataloaders[phase]):
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
                    loss = loss_function(y_, y)
                    epoch_avg.add(loss.item())
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                print(f"Epoch: {epoch}, batch: {i}, loss {loss.item()}")
        if epoch % 50 == 0:
            saver.save(model, optimizer, epoch)
        print(f"Epoch loss: {epoch_avg}")
        epoch_losses.append(str(epoch_avg))

    with open("losses", "w+") as file:
        file.write(str(epoch_losses))


def loss_function(y_, y):
    """
    Calculate loss funtion as in paper
    :param y_: Prediction
    :param y: Ground truth
    :return: 1-dim tensor of loss value
    """
    # loss = torch.zeros(1, requires_grad=True)
    # if CUDA:
    #     loss = loss.cuda()
    loss = torch.sum(
        y[..., 0] * (
                c_orrd * (y_[..., 0] - y[..., 0]) ** 2 +
                c_orrd * (y_[..., 1] - y[..., 1]) ** 2 +
                (y_[..., 3] - y[..., 3]) ** 2 + (y_[..., 4] - y[..., 4]) ** 2) +
        (1 - y[..., 0]) * (
                c_noobj * y_[..., 0] ** 2
        )
    )
    # for predicted_single_example, target_single_example in zip(y_, y):
    #     for predicted_row, target_row in zip(predicted_single_example, target_single_example):
    #         for predited_cell, target_cell in zip(predicted_row, target_row):
    #             loss += (1 - target_cell[0]) * c_noobj * predited_cell[0] ** 2 + \
    #                     target_cell[0] * (c_orrd * (predited_cell[0] - target_cell[0]) ** 2 +
    #                                       c_orrd * (predited_cell[1] - target_cell[1]) ** 2 +
    #                                       (predited_cell[3] - target_cell[3]) ** 2 + (
    #                                               predited_cell[4] - target_cell[4]) ** 2
    #                                       )
    # if target_cell[0] == 0:  # there shouldn't be object
    #     loss += c_noobj * predited_cell[0] ** 2
    # else:
    #     loss += c_orrd * (predited_cell[0] - target_cell[0]) ** 2  # p
    #     loss += c_orrd * (predited_cell[1] - target_cell[1]) ** 2 + (
    #             predited_cell[2] - target_cell[2]) ** 2  # x y
    #     loss += (predited_cell[3] - target_cell[3]) ** 2 + (predited_cell[4] - target_cell[4]) ** 2  # wh
    loss.requires_grad = True
    return loss


if __name__ == '__main__':
    args = parser.parse_args()
    train(args)

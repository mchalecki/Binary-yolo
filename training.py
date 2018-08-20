import argparse
import os
import random
import torch
import logging

from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data.dataset import Subset
from torchvision import transforms
from tensorboardX import SummaryWriter

from dataset import YoloDataset, ToTensor
from network import CustomModel, MyResnet
from paths import DATASET
from utils.saver import CheckpointSaver
from utils.stats import AVG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Training of yolonet')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--test_size', help="Percentage of dataset to be test set", type=float, default=0.1)
parser.add_argument('--epochs', help="Number of epochs", type=int, default=250)
parser.add_argument('--weightsfile', help="File with weights", type=str, default='yolov3.weights')
parser.add_argument('--lr', help="Learning rate", type=int, default=0.001)
parser.add_argument('--save_dir_name', help="Where to save checkpoints.", type=str, default="saves")

CUDA = torch.cuda.is_available()
c_orrd = 6
c_noobj = .5

TRAIN = "train"
VALID = "valid"


def train(args):
    yolo_dataset = YoloDataset(DATASET, "faces.csv", input_size=208,
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
        TRAIN: trainloader, VALID: validloader
    }
    model = MyResnet()
    if CUDA:
        model.cuda()

    optimizer = optim.Adam(model.conv_addition.parameters(), lr=args.lr)
    saver = CheckpointSaver(args.save_dir_name, max_checkpoints=3)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    # writer = SummaryWriter(os.path.join(args.save_dir_name, "log"))
    train_losses, test_losses = [], []
    for epoch in range(args.epochs):
        for phase in dataloaders:
            if phase == TRAIN:
                scheduler.step()
                model.train()
            else:
                model.eval()

            epoch_avg = AVG()
            logger.info(f"-----------------{phase.upper()}-----------------")
            for i, data in enumerate(dataloaders[phase]):
                img, y = data['img'], data['y']
                if CUDA:
                    img = img.cuda()
                    y = y.cuda()
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == TRAIN):
                    pred = model(img)
                    y_ = pred.permute((0, 2, 3, 1))
                    y = y.permute((0, 2, 3, 1))
                    loss = loss_function(y_, y)
                    epoch_avg.add(loss.item())
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                logger.info(f"Epoch: {epoch}, batch: {i}, loss {loss.item()}")
            logger.info(f"Epoch: {epoch}, average loss: {epoch_avg}")
            if phase == TRAIN:
                train_losses.append(str(epoch_avg))
            else:
                test_losses.append(str(epoch_avg))
            # writer.add_scalar(f'{phase}_data/average_loss', str(epoch_avg), epoch)
        if epoch % 20 == 0:
            saver.save(model, optimizer, epoch)

    with open(os.path.join(args.save_dir_name, "losses.txt"), 'w') as file:
        file.write(str(train_losses))
        file.write(str(test_losses))


def loss_function(y_, y):
    """
    Calculate loss funtion as in paper except only for one class.
    The first element of the vector is certainity.
    There is only one class so there is no need for more elements than 4 in vector and loss by these elements.
    :param y_: Prediction
    :param y: Ground truth
    :return: 1-dim tensor of loss value
    """
    loss = torch.sum(
        y[..., 0] * (
                c_orrd * (y_[..., 0] - y[..., 0]) ** 2 +
                c_orrd * (y_[..., 1] - y[..., 1]) ** 2 +
                (y_[..., 3] - y[..., 3]) ** 2 + (y_[..., 4] - y[..., 4]) ** 2) +
        (1 - y[..., 0]) * (
                c_noobj * y_[..., 0] ** 2
        )
    )
    return loss


if __name__ == '__main__':
    args = parser.parse_args()
    train(args)

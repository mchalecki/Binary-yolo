import argparse
import os
import sys

import torch
from skimage import io
from skimage.transform import resize

from network import MyResnet

parser = argparse.ArgumentParser(description='Predict using pretrained model of yolo.')
parser.add_argument('--file', type=str, help="File to predict objects.", required=True)
parser.add_argument('--checkpoint', type=str, help="File from which to load weights.", required=True)
parser.add_argument('--batch_size', type=int, default=1)

IN = 195
CUDA = torch.cuda.is_available()

FILTER_SIZE = 13
def prepro_img(file: str, IN):
    img = io.imread(file)
    in_size = img.shape
    try:
        if in_size[2] == 4:
            img = img[..., :3]  # reduce transparent channel
            in_size = img.shape
    except IndexError:
        print(f"Remove from csv white/black image {box[0]}")
    img = resize(img, (IN, IN))
    img = img.transpose((2, 0, 1))
    img /= 255.0
    img_tensor = torch.FloatTensor(img)
    img_tensor = img_tensor.view(args.batch_size, 3, IN, IN)
    return img_tensor


def predict(args):
    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        model = MyResnet()
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.checkpoint, checkpoint['epoch']))
    else:
        print("No such checkpoint")
        sys.exit(0)
    img = prepro_img(args.file, IN)
    if CUDA:
        img = img.cuda()
        model = model.cuda()

    prediction = model(img)
    prediction = prediction.view(args.batch_size, IN/FILTER_SIZE, IN/FILTER_SIZE, prediction.shape[1])
    certainity = prediction[..., 0]
    print(torch.sum(certainity>0.8))
    pass


if __name__ == '__main__':
    args = parser.parse_args()
    predict(args)

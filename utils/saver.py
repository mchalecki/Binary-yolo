import logging
import os
from collections import deque

from torch import nn
import torch

logger = logging.getLogger(__name__)


class TrackerDeque(deque):

    def __init__(self, maxlen=None):
        self._maxlen = maxlen
        super(TrackerDeque, self).__init__()

    def append(self, x: str) -> str:
        poped = None
        if self._maxlen and self.__len__() + 1 > self._maxlen:
            poped = self.popleft()
        super(TrackerDeque, self).append(x)
        if poped:
            return poped


class CheckpointSaver:

    def __init__(self, dir: str, max_checkpoints: int = None, prefix=None, only_weights=True):
        """
        Performs model saving with max checkpoints.
        :param dir: Directory when checkpoint will be stored.
        :param max_checkpoints: Max number of checkpoint to store.
        :param prefix: Prefix for save filenames.
        :param only_weights: Store whole model or only weights. Default True.
        """
        self.dir = dir
        self.max_checkpoints = max_checkpoints
        self.checkpoitns = TrackerDeque(max_checkpoints)
        self.only_weights = only_weights
        self.prefix = prefix
        os.makedirs(os.path.join(dir, 'log'), exist_ok=True)

    def save(self, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, step: int = None, **kwargs):
        if step is None:
            step = epoch
        if self.prefix:
            checkpoint_path = os.path.join(self.dir, f"{self.prefix}_{str(step)}.pth")
        else:
            checkpoint_path = os.path.join(self.dir, str(step) + ".pth")
        save_state = {
            'epoch': epoch + 1,
            'optimizer': optimizer.state_dict(),
        }
        if self.only_weights:
            save_state['state_dict'] = model.state_dict()
        else:
            save_state['model'] = model
        save_state = dict(save_state, **kwargs)
        torch.save(save_state, checkpoint_path)
        logger.info(f"Saved in {checkpoint_path}")
        popped = self.checkpoitns.append(checkpoint_path)
        if popped:
            try:
                os.remove(popped)
            except OSError:
                pass

    @staticmethod
    def load(path, model, optimizer):
        logger.info(f"Loading checkpoint from {path}")
        if os.path.isfile(path):
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            others = checkpoint.copy()
            others.pop('state_dict')
            others.pop('optimizer')
            logger.info(f"Successfully loaded from {path}")
            return model, optimizer, others
        else:
            logger.info(f"No checkpoint at {path}")


def adjust_learning_rate(optimizer, epoch, lr_default=0.001):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr_default * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

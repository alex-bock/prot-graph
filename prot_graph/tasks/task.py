
import abc

from torch import nn


class Task(nn.Module, abc.ABC):

    def __init__(self):

        super(Task, self).__init__()

        return

    @abc.abstractmethod
    def forward(self):

        return
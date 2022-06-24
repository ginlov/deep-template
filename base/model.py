from torch import nn
from abc import abstractmethod

import numpy as np


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, **inputs):
        """
        Forward pass logic

        :param inputs:
        :return:
        """
        raise NotImplementedError

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + f"\n Trainable parameters: {params}"

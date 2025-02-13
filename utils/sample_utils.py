import torch
import numpy as np
import abc

class LieGroup(abc.ABC):
    @abc.abstractmethod
    def _sample_algebra(self, batch):
        '''
        Sample distribution on Lie algbera
        '''
        pass

    def _sample_group(self, batch):
        lie_algebra_element = self._sample_algebra(batch)
        lie_group_element = torch.matrix_exp(lie_algebra_element)
        return lie_group_element

    def sample(self, batch):
        lie_group_samples = self._sample_group(batch).numpy()
        
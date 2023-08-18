import torch

class Badnets:
    def __init__(self, normalize, device=None):
        self.device = device
        self.normalize = normalize

    def inject(self, inputs):
        _, width, height = inputs.shape[1:]
        inputs[:, :, width - 3, height - 3] = 1
        # print(inputs)
        inputs[:, :, width - 3, height - 2] = 1
        inputs[:, :, width - 2, height - 3] = 1
        inputs[:, :, width - 2, height - 2] = 1
        return inputs

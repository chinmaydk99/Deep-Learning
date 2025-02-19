import torch

class LayerNormalization:
    def __init__(self, parameter_shape, eps = 1e-5):
        self.parameter_shape = parameter_shape
        self.eps = eps
        self.gamma = torch.ones(self.parameter_shape)
        self.beta = torch.zeros(self.parameter_shape)
      
    def forward(self, input):
        dims = [-(i+1) for i in range(len(self.parameter_shape))]
        mean = input.mean(dim = dims, keepdim = True)
        print(f"Mean \n ({mean.size()}): \n {mean}")
        var = ((input-mean)**2).mean(dim = dims, keepdim = True)
        std = (var+self.eps).sqrt()
        print(f"Standard Deviation \n ({std.size()}): \n {std}")
        y = (input-mean)/std
        print(f"y \n ({y.size()}) = \n {y}")
        out = self.gamma*y + self.beta
        print(f"out \n ({out.size()}) = \n {out}")
        return out
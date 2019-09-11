import numpy as np
import chainer
import torch


def torch_gaussian_kl_divergence(mean, ln_var):
    mean = torch.tensor(mean)
    ln_var = torch.tensor(ln_var)
    scale = torch.sqrt(torch.exp(ln_var))
    std = torch.distributions.normal.Normal(
        torch.tensor(0.0), torch.tensor(1.0))
    dist = torch.distributions.normal.Normal(mean, scale)
    return torch.distributions.kl_divergence(dist, std)


def print_both(mean, ln_var):
    print(chainer.functions.gaussian_kl_divergence(np.array(mean),
                                                   np.array(ln_var)),
          torch_gaussian_kl_divergence(mean, ln_var))


print_both(0.0, 1.0)
print_both(1.0, 2.0)
print_both(0.3, 5.3)

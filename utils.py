import numpy as np
from skimage.measure import compare_ssim

import torch
import torch.nn as nn
import torch.autograd as autograd


def PSNR(x, y, peak=1.0):
    """
    x: Predict image(float32), Shape=(height, width, n_channel)
    y: Ground truth image(float32), Shape=(height, width, n_channel)
    peak: peak value of input image, Default: 1.0
    """
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    
    rmse = np.sqrt(np.mean((x-y)**2))
    psnr = 20 * np.log10(peak / rmse)
    
    return psnr
    
    
def SSIM(x, y, peak=1.0):
    """
    x: Predict image(float32), Shape=(height, width, n_channel)
    y: Ground truth image(float32), Shape=(height, width, n_channel)
    peak: based on float-32 image, Default: 1.0
    """

    x = x.astype(np.float32)
    y = y.astype(np.float32)
    
    if x.shape[-1] == 3:
        multichannel = True
    else:
        multichannel = False
    
    ssim = compare_ssim(x, y, data_range=peak, multichannel=multichannel)
    
    return ssim


"""
COPY CODE = compute_gp(), gram_matrix(), TextureLoss()
From https://github.com/S-aiueo32/srntt-pytorch
"""
# Use https://github.com/S-aiueo32/srntt-pytorch/blob/4ea0aa22a54a2d1b1f19c4a43596a693b9e7c067/losses/__init__.py#L21
def compute_gp(netD, real_data, fake_data):
    device = real_data.device
    alpha = torch.rand(real_data.shape[0], 1, 1, 1, device=device)
    alpha = alpha.expand(real_data.size())

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


# Use https://github.com/S-aiueo32/srntt-pytorch/blob/4ea0aa22a54a2d1b1f19c4a43596a693b9e7c067/losses/texture_loss.py#L8
def gram_matrix(features):
    N, C, H, W = features.size()
    feat_reshaped = features.view(N, C, -1)

    # Use torch.bmm for batch multiplication of matrices
    gram = torch.bmm(feat_reshaped, feat_reshaped.transpose(1, 2))

    return gram


# Use https://github.com/S-aiueo32/srntt-pytorch/blob/4ea0aa22a54a2d1b1f19c4a43596a693b9e7c067/losses/texture_loss.py#L18
class TextureLoss(nn.Module):
    """
    creates a criterion to compute weighted gram loss.
    """
    def __init__(self, model, use_weights=False):
        super(TextureLoss, self).__init__()
        self.use_weights = use_weights

        self.model = model
        self.register_buffer('a', torch.tensor(-20., requires_grad=False))
        self.register_buffer('b', torch.tensor(.65, requires_grad=False))

        
    def forward(self, x, maps, weights):
        input_size = x.shape[-1]
        x_feat = self.model(x, ['relu1_1', 'relu2_1', 'relu3_1'])

        if self.use_weights:
            weights = F.pad(weights, (1, 1, 1, 1), mode='replicate')
            for idx, l in enumerate(['relu3_1', 'relu2_1', 'relu1_1']):
                # adjust the scale
                weights_scaled = F.interpolate(
                    weights, None, 2**idx, 'bicubic', True)

                # compute coefficients
                coeff = weights_scaled * self.a.detach() + self.b.detach()
                coeff = torch.sigmoid(coeff)

                # weighting features and swapped maps
                maps[l] = maps[l] * coeff
                x_feat[l] = x_feat[l] * coeff

        # for large scale
        loss_relu1_1 = torch.norm(
            gram_matrix(x_feat['relu1_1']) - gram_matrix(maps['relu1_1']),
        ) / 4. / ((input_size * input_size * 1024) ** 2)

        # for medium scale
        loss_relu2_1 = torch.norm(
            gram_matrix(x_feat['relu2_1']) - gram_matrix(maps['relu2_1'])
        ) / 4. / ((input_size * input_size * 512) ** 2)

        # for small scale
        loss_relu3_1 = torch.norm(
            gram_matrix(x_feat['relu3_1']) - gram_matrix(maps['relu3_1'])
        ) / 4. / ((input_size * input_size * 256) ** 2)

        loss = (loss_relu1_1 + loss_relu2_1 + loss_relu3_1) / 3.

        return loss


class ReconstructLoss(nn.Module):
    def __init__(self):
        super(ReconstructLoss, self).__init__()
        self.criterion = nn.L1Loss()
        
    def forward(self, x,y):
        loss = self.criterion(x,y)
        
        return loss

    
class PerceptualLoss(nn.Module):
    def __init__(self, model, target_layer):
        super(PerceptualLoss, self).__init__()
        self.model = model
        self.target_layer = target_layer
    
    def forward(self, x,y):
        x_feat, *_ = self.model(x, [self.target_layer]).values()
        y_feat, *_ = self.model(y, [self.target_layer]).values()

        # But, Author's code use L2-norm - https://github.com/ZZUTK/SRNTT/blob/master/SRNTT/model.py#L376.
        loss = torch.norm(x_feat - y_feat, p='fro') 

        return loss

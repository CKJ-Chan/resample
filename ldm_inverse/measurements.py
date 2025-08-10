'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
from functools import partial
import yaml
from torch.nn import functional as F
import torch
from motionblur.motionblur import Kernel
import numpy as np

from util.resizer import Resizer
from util.img_utils import Blurkernel, fft2_m

#from torch_radon import Radon, solvers


# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass

    # alias so other code can call A.adjoint(...)
    def adjoint(self, data, **kwargs):
        return self.transpose(data, **kwargs)
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


@register_operator(name='noise')
class DenoiseOperator(LinearOperator):
    def __init__(self, device):
        self.device = device
    
    def forward(self, data):
        return data

    def transpose(self, data):
        return data
    
    def ortho_project(self, data):
        return data

    def project(self, data):
        return data


@register_operator(name='super_resolution')
class SuperResolutionOperator(LinearOperator):
    def __init__(self, in_shape, scale_factor, device):
        self.device = device
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        self.down_sample = Resizer(in_shape, 1/scale_factor).to(device)

    def forward(self, data, **kwargs):
        data = data.to(self.device) # Sending to device
        return self.down_sample(data).to(self.device) # Sending to device

    def transpose(self, data, **kwargs):
        return self.up_sample(data)

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)
    

@register_operator(name='motion_blur')
class MotionBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='motion',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)  # should we keep this device term?

        self.kernel = Kernel(size=(kernel_size, kernel_size), intensity=intensity)
        kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(kernel)
    
    def forward(self, data, **kwargs):
        # A^T * A 
        data = data.to(self.device) # Sending to device
        return self.conv(data).to(self.device)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        kernel = self.kernel.kernelMatrix.type(torch.float32).to(self.device)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)


@register_operator(name='gaussian_blur')
class GaussialBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))

    def forward(self, data, **kwargs):
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)


@register_operator(name='inpainting')
class InpaintingOperator(LinearOperator):
    '''This operator get pre-defined mask and return masked image.'''
    def __init__(self, device):
        self.device = device
    
    def forward(self, data, **kwargs):
        try:
            return data * kwargs.get('mask', None).to(self.device)
        except:
            raise ValueError("Require mask")
    
    def transpose(self, data, **kwargs):
        return data
    
    def ortho_project(self, data, **kwargs):
        return data - self.forward(data, **kwargs)


# Center-tile sampling operator

@register_operator(name='center_tile_sampling')
class CenterTileSamplingOperator(LinearOperator):
    """
    A: split into s×s tiles; sample geometric centre.
       - odd s: exact centre pixel
       - even s: bilinear at the geometric centre (half-pixel)
    forward: (B,C,H,W) -> (B,C,H//s,W//s)
    transpose/adjoint: back-project with matching (bi)linear weights
    """
    def __init__(self, s, device, in_shape=None):
        self.s = int(s)
        self.device = device
        self.in_shape = in_shape

    def _centres_hw(self, H, W, device):
        s = self.s
        h, w = H // s, W // s
        i = torch.arange(h, device=device).float().view(1,1,h,1)
        j = torch.arange(w, device=device).float().view(1,1,1,w)
        yc = i * s + (s - 1) / 2.0
        xc = j * s + (s - 1) / 2.0
        return yc, xc

    def forward(self, data, **kwargs):
        x = data.to(self.device)
        B, C, H, W = x.shape
        s = self.s
        assert H % s == 0 and W % s == 0, "H,W must be divisible by s"
        yc, xc = self._centres_hw(H, W, x.device)
        yc = yc.expand(B, 1, H // s, W // s)
        xc = xc.expand(B, 1, H // s, W // s)

        if s % 2 == 1:
            yi = yc.long(); xi = xc.long()
            b = torch.arange(B, device=x.device)[:, None, None, None]
            c = torch.arange(C, device=x.device)[None, :, None, None]
            return x[b, c, yi.expand(B,C,H//s,W//s), xi.expand(B,C,H//s,W//s)]

        x0 = torch.floor(xc).long().clamp(max=W-1); x1 = (x0 + 1).clamp(max=W-1)
        y0 = torch.floor(yc).long().clamp(max=H-1); y1 = (y0 + 1).clamp(max=H-1)
        wx = xc - x0.float(); wy = yc - y0.float()

        b = torch.arange(B, device=x.device)[:, None, None, None]
        c = torch.arange(C, device=x.device)[None, :, None, None]

        def gather(ix, iy):
            return x[b, c, iy.expand(B,C,H//s,W//s), ix.expand(B,C,H//s,W//s)]

        v00 = gather(x0, y0)
        v01 = gather(x0, y1)
        v10 = gather(x1, y0)
        v11 = gather(x1, y1)

        return (1-wx)*(1-wy)*v00 + (1-wx)*wy*v01 + wx*(1-wy)*v10 + wx*wy*v11

    def transpose(self, data, **kwargs):
        y = data.to(self.device)
        H = kwargs.get('H'); W = kwargs.get('W')
        if H is None or W is None:
            if self.in_shape is not None: _, _, H, W = self.in_shape
            else: H, W = y.shape[-2] * self.s, y.shape[-1] * self.s

        B, C, h, w = y.shape
        x_bp = torch.zeros(B, C, H, W, device=y.device)
        yc, xc = self._centres_hw(H, W, y.device)
        yc = yc.expand(B, 1, h, w); xc = xc.expand(B, 1, h, w)

        b = torch.arange(B, device=y.device)[:, None, None, None]
        c = torch.arange(C, device=y.device)[None, :, None, None]

        if self.s % 2 == 1:
            yi = yc.long(); xi = xc.long()
            x_bp.index_put_((b.expand(B,C,h,w), c.expand(B,C,h,w), yi.expand(B,C,h,w), xi.expand(B,C,h,w)), y, accumulate=True)
            return x_bp

        x0 = torch.floor(xc).long().clamp(max=W-1); x1 = (x0 + 1).clamp(max=W-1)
        y0 = torch.floor(yc).long().clamp(max=H-1); y1 = (y0 + 1).clamp(max=H-1)
        wx = xc - x0.float(); wy = yc - y0.float()

        def add_at(ix, iy, weight):
            x_bp.index_put_((b.expand(B,C,h,w), c.expand(B,C,h,w), iy.expand(B,C,h,w), ix.expand(B,C,h,w)), (y * weight).to(y.dtype), accumulate=True)

        add_at(x0, y0, (1-wx)*(1-wy))
        add_at(x0, y1, (1-wx)*wy)
        add_at(x1, y0, wx*(1-wy))
        add_at(x1, y1, wx*wy)
        return x_bp


class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 

@register_operator(name='phase_retrieval')
class PhaseRetrievalOperator(NonLinearOperator):
    def __init__(self, oversample, device):
        self.pad = int((oversample / 8.0) * 256)
        self.device = device
        
    def forward(self, data, **kwargs):
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2_m(padded).abs()
        return amplitude


@register_operator(name='nonlinear_blur')
class NonlinearBlurOperator(NonLinearOperator):
    def __init__(self, opt_yml_path, device):
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)     
         
    def prepare_nonlinear_blur_model(self, opt_yml_path):
        '''
        Nonlinear deblur requires external codes (bkse).
        '''
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path)) 
        blur_model = blur_model.to(self.device)
        return blur_model
    
    def forward(self, data, **kwargs):
        
#         random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
        np.random.seed(0)
        kernel_np = np.random.randn(1,512,2,2)*1.2
        random_kernel = (torch.from_numpy(kernel_np)).float().to(self.device)
        data = (data + 1.0) / 2.0  #[-1, 1] -> [0, 1]
        blurred = self.blur_model.adaptKernel(data, kernel=random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1) #[0, 1] -> [-1, 1]
        return blurred

# =============
# Noise classes
# =============


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma


@register_noise(name='poisson')
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):
        '''
        Follow skimage.util.random_noise.
        '''

        # TODO: set one version of poisson
       
        # version 3 (stack-overflow)
        import numpy as np
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)

        # version 2 (skimage)
        # if data.min() < 0:
        #     low_clip = -1
        # else:
        #     low_clip = 0

    
        # # Determine unique values in iamge & calculate the next power of two
        # vals = torch.Tensor([len(torch.unique(data))])
        # vals = 2 ** torch.ceil(torch.log2(vals))
        # vals = vals.to(data.device)

        # if low_clip == -1:
        #     old_max = data.max()
        #     data = (data + 1.0) / (old_max + 1.0)

        # data = torch.poisson(data * vals) / float(vals)

        # if low_clip == -1:
        #     data = data * (old_max + 1.0) - 1.0
       
        # return data.clamp(low_clip, 1.0)
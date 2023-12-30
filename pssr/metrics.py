import torch, math
import torch.nn.functional as F

def pixel(mse : float, image_range : int):
    return math.sqrt(mse) * image_range

def psnr(mse, max):
    return 20 * torch.log10(max / torch.sqrt(mse))

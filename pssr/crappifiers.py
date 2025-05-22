import numpy as np
from abc import ABC, abstractmethod
from skimage.util import random_noise
from skimage.filters import gaussian

class Crappifier(ABC):
    r"""Crappifier base class for custom crappifiers. Override the :meth:`crappify` method for logic.
    """
    @abstractmethod
    def crappify(self, image : np.ndarray):
        r"""An abstract function for degrading a low resolution image to simulate undersampling.

        This method is not responsible for downscaling the image, only for injecting noise.

        Args:
            image (np.ndarray) : Low resolution image to crappify.

        Returns:
            crap (np.ndarray) : The low resolution image, only now has it been crappified.
        """
        raise NotImplementedError('"crappify" method not implemented.')
    
    def __call__(self, image : np.ndarray):
        return self.crappify(image)

class MultiCrappifier(Crappifier):
    def __init__(self, *args : list[Crappifier], clip : bool = True):
        r"""Chains multiple crappifiers sequentially for degrading low resolution images.

        Args:
            args (Crappifier) : Crappifiers to be applied in order from first to last.

            clip (bool) : Clip values to image range between each crappifier step. Default is True.
        """
        self.crappifiers = args
        self.clip = clip
    
    def crappify(self, image : np.ndarray):
        for crappifier in self.crappifiers:
            image = crappifier.crappify(image)
            if self.clip:
                image = np.clip(image, 0, 255)
        return image

class AdditiveGaussian(Crappifier):
    def __init__(self, intensity : float = 13, gain : float = 0, spread : float = 0):
        r"""Crappifier using additive Gaussian noise sampling (normally distributed noise). Adds additive Gaussian noise to a low resolution image.

        Approximates :class:`Poisson` noise at high samples.

        Args:
            intensity (float) : Standard deviation of Gaussian distribution. Higher values will introduce more noise to the image. Default is 13.

            gain (float) : Mean of Gaussian distribution. Higher or lower values will raise the mean image value higher or lower respectively. Default is 0.

            spread (float) : Standard deviation of crappifier intensity for training on a range of crappifications. Default is 0.
        """
        self.intensity = intensity
        self.gain = gain
        self.spread = spread

    def crappify(self, image : np.ndarray):
        intensity = max(np.random.normal(self.intensity, self.spread), 0) if self.spread > 0 else self.intensity
        return image.astype(np.float32) + np.random.normal(self.gain, intensity, image.shape)
    
class Poisson(Crappifier):
    def __init__(self, intensity : float = 1, gain : float = 0, spread : float = 0):
        r"""Crappifier using Poisson noise sampling (shot noise). Adds Poisson noise to a low resolution image.

        Args:
            intensity (float) : Interpolated mix of generated Poisson image. 1 is standard distribution, 0 is none, larger values amplify noise. Default is 1.

            gain (float) : Value gain added to the output. Default is 0.

            spread (float) : Standard deviation of crappifier intensity for training on a range of crappifications. Default is 0.
        """
        self.intensity = intensity
        self.gain = gain
        self.spread = spread
        
    def crappify(self, image : np.ndarray):
        return self._interpolate(image.astype(np.float32), np.random.poisson(np.clip(image, 0, np.inf))) + self.gain
    
    def _interpolate(self, x, y):
        intensity = max(np.random.normal(self.intensity, self.spread), 0) if self.spread > 0 else self.intensity
        return x * (1 - intensity) + y * intensity
    
class SaltPepper(Crappifier):
    def __init__(self, intensity : float = 0.5, gain : float = 0, spread : float = 0):
        r"""Crappifier using salt and pepper noise (full value addition/deletion). Adds salt and pepper noise to a low resolution image.

        Args:
            intensity (float) : Percent of values to replace with salt and pepper noise. Default is 0.5.

            gain (float) : Value gain added to the image BEFORE noise. Default is 0.

            spread (float) : Standard deviation of crappifier intensity for training on a range of crappifications. Default is 0.
        """
        self.intensity = intensity / 100
        self.gain = gain
        self.spread = spread
        
    def crappify(self, image : np.ndarray):
        intensity = max(np.random.normal(self.intensity, self.spread), 0) if self.spread > 0 else self.intensity
        return random_noise(np.clip(image.astype(np.float32) + self.gain, 0, 255)/255, mode="s&p", amount=intensity) * 255

class Blur(Crappifier):
    def __init__(self, intensity : float = 2, gain : float = 0, spread : float = 0):
        r"""Crappifier using Gaussian blurring. Adds Gaussian blur to a low resolution image.

        Args:
            intensity (float) : Sigma standard deviation of Gaussian kernel. Higher values introduce more blurring to the image. Default is 2.

            gain (float) : Value gain added to the output. Default is 0.

            spread (float) : Standard deviation of crappifier intensity for training on a range of crappifications. Default is 0.
        """
        self.intensity = intensity
        self.gain = gain
        self.spread = spread

    def crappify(self, image : np.ndarray):
        intensity = max(np.random.normal(self.intensity, self.spread), 0) if self.spread > 0 else self.intensity
        return gaussian(image.astype(np.float32), intensity, channel_axis=0) + self.gain

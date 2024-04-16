import numpy as np
from abc import ABC, abstractmethod

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
        raise NotImplementedError("Crappify method not implemented")

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
        return image + np.random.normal(self.gain, intensity, image.shape)
    
class Poisson(Crappifier):
    def __init__(self, intensity : float = 1, gain : float = 0, spread : float = 0):
        r"""Crappifier using Poisson noise sampling (shot noise). Adds Poisson noise to a low resolution image.

        Args:
            intensity (float) : Interpolated mix of generated Poisson image. 1 is full noise, 0 is none. Default is 1.

            gain (float) : Value gain added to the output.

            spread (float) : Standard deviation of crappifier intensity for training on a range of crappifications. Default is 0. 
        """
        self.intensity = intensity
        self.gain = gain
        self.spread = spread
        
    def crappify(self, image : np.ndarray):
        image_max = max(1, image.max())
        return self._interpolate(image, np.random.poisson(image/255*image_max)/image_max*255) + self.gain
    
    def _interpolate(self, x, y):
        intensity = max(np.random.normal(self.intensity, self.spread), 0) if self.spread > 0 else self.intensity
        return x * (1 - intensity) + y * intensity

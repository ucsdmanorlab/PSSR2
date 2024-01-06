import numpy as np
from abc import ABC, abstractmethod

class Crappifier(ABC):
    r"""Crappifier base class for custom crappifiers. Override the :meth:`crappify` method for logic.
    """
    @abstractmethod
    def crappify(self, image : np.ndarray):
        r"""An abstract function for degrading a low resolution image to simulate undersampling.

        Args:
            image (np.ndarray) : Low resolution image to crappify.

        Returns:
            crap (np.ndarray) : The low resolution image, only now has it been crappified.
        """
        raise NotImplementedError("Crappify method not implemented")

class AdditiveGaussian(Crappifier):
    def __init__(self, mean : float = 0, deviation : float = 3):
        r"""Adds additive Gaussian noise to a low resolution image.

        Args:
            mean (float) : Mean of Gaussian distribution. Higher or lower values will raise the mean image value higher or lower respectively. Default is 0.

            deviation (float) : Standard deviation of Gaussian distribution. Higher values will introduce more noise to the image. Default is 3.
        """
        self.mean = mean
        self.deviation = deviation

    def crappify(self, image : np.ndarray):
        return np.clip(image + np.random.normal(self.mean, self.deviation, image.shape), 0, 255)
    
class Poisson(Crappifier):
    def __init__(self, intensity : float = 1):
        r"""Adds Poisson noise to a low resolution image.

        Args:
            intensity (float) : Interpolated mix of generated Poisson image. 1 is full noise, 0 is none. Default is 1.
        """
        self.intensity = intensity
    def crappify(self, image : np.ndarray):
        return np.clip(self._interpolate(image, np.random.poisson(image/255*image.max())/image.max()*255), 0, 255)
    
    def _interpolate(self, x, y):
        return x * (1 - self.intensity) + y * self.intensity

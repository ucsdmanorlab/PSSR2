data
=====

Datasets and functions for handling and synthetically generating *high-low-resolution* image pairs.

Datasets that do not take in paired data can be used in "LR mode" for predictions, where the dataset loads only unmodified low-resolution images.
Usage is specific to individual datasets.

Users are advised to keep dataloading resolutions to a power of 2 even if the raw input images have a different size.
This is elaborated in :doc:`models`.


Modules
--------

.. code-block:: python

   from pssr.data import ...

.. toctree::
   :titlesonly:

   data/ImageDataset
   data/SlidingDataset
   data/PairedImageDataset
   data/PairedSlidingDataset
   data/preprocess_dataset

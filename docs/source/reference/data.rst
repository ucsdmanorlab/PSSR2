data
=====

**TODO: SlidingPairedDataset and sliding preprocess_hr**

Datasets and functions for handling and synthetically generating *high-low-resolution* image pairs.

Datasets that do not take in paired data can be used in "LR mode", taking in low-resolution data for predictions.
Instructions are specific to individual datasets.

To avoid data loss, image data should be square, non-conforming images will be cropped to fit.

Users are advised to keep image resolutions to a power of 2, elaborated in :doc:`models`.


Modules
--------

.. code-block:: python

   from pssr.data import ...

.. toctree::
   :titlesonly:

   data/ImageDataset
   data/SlidingDataset
   data/PairedImageDataset
   data/preprocess_hr

Getting Started
================

Installation
-------------

It is required that `Python <https://www.python.org>`_ is installed to use **PSSR**.
If you have not already, download it `here <https://www.python.org/downloads>`_.

Before installing **PSSR**, you may want to create an environment for it with

.. code-block:: console

   $ conda env create pssr

or the equivalent for any other environment manager if you chose to do so.

|

It is recommended that **PSSR** be installed via the ``pip`` package manager:

.. code-block:: console

   $ pip install pssr

However, specific versions can also be installed directly from the `GitHub <https://github.com/haydenstites/PSSR>`_ repository:

.. code-block:: console

   $ pip install git+https://github.com/haydenstites/PSSR/dist/pssr-x.x.x-py3-none-any.whl


Running the CLI
----------------

The **PSSR** CLI is included with package installation.
It provides a simple interface for using **PSSR** without having to write any code, and covers most basic use cases.

The CLI can run in either *train* or *predict* mode. It takes in a number of arugments, described below.

.. dropdown:: CLI Arguments

   .. argparse::
      :filename: ../_pssr.py
      :func: parse
      :prog: pssr

Keep in mind that arguments representing an object such as a dataset or model can be defined as a class declaration with additonal arguments in Python syntax.
For example ``--model-type`` could be given as ``-mt ResUNet(hidden=[128, 256], scale=4)``. 

|

If you do not have access to a microscopy dataset, a sample training dataset can be found
`here <https://drive.google.com/file/d/1Sirrh180WrkHgPR0S8_43-f0S2GaK7iZ/view>`__ containing *high-resolution* images of resolution 512.
Real-world *high-low-resolution* image pairs for testing can be found `here <https://drive.google.com/file/d/1BI6K5r65ubn3Vj866ikUUj8VVqHT0j-4/view>`__.
Larger datasets and all data used in the **PSSR** paper can also be found on `3Dem.org <https://3dem.org/public-data/tapis/public/3dem.storage.public/2021_Manor_PSSR/>`_.
If your dataset have different resolution data, ``hr_res`` and ``scale`` can be changed correspondingly.

|

Training
+++++++++

A model can be trained by running

.. code-block:: console

   $ python demo.py -t -dp your/path

where ``your/path`` is replaced with the path of your training dataset (folder containing *high-resolution* images/image sheets).

The *low-resolution* images will generated via :doc:`../reference/crappifiers/Crappifier`, which is explained in :doc:`principles`.

|

By default the trained model will be saved as ``model.pth``.

By default the dataset used is :doc:`../reference/data/ImageDataset`.
If your dataset contains image sheets (e.g. .czi files) rather than many images, you can use :doc:`../reference/data/SlidingDataset` by adding the argument ``-dt SlidingDataset``.
The batch size can also be changed with the ``-b`` argument.

|

Predicting
+++++++++++

A pretrained **PSSR** model for EM data can be found `here <https://drive.google.com/file/d/1DIWlcjljG4fRNCoMSjkNdhtzSZJ4QXHg/view>`__,
a :doc:`../reference/models/ResUNet` with default arguments.

To run the demo in predict mode, omit the ``-t`` argument. The dataset path should be changed to the path containing the *low-resolution* images to be upscaled.
The `-mp` argument can be set to your model path if its different than the default. The predicted upscaled images will be saved to the ``preds`` folder.

.. note::

   :doc:`../reference/data/SlidingDataset` does not automatically detect *low-resolution* inputs.
   ``hr_res`` must be lowered to the size of the *low-resolution* image and ``lr_scale`` must be lowered to 1.

|

If a :doc:`../reference/data/PairedImageDataset` instance with *high-low-resolution* image pairs is given as the dataset, additional performance metrics will be calculated.
To define both *high-resolution* and *low-resolution* data paths, omit the ``-dp`` argument and instead use

.. code-block:: console

   $ python demo.py -mt "PairedImageDataset(hr_path='your/hr', lr_path='your/lr')"

where ``your/hr`` and ``your/lr`` are repleaced by your *high-resolution* and *low-resolution* data paths respectively.

|

If *high-resolution* images are given using an :doc:`../reference/data/ImageDataset`,
then *low-resolution* images will be generated via :doc:`../reference/crappifiers/Crappifier` and performance metrics will still be calculated.


Next Steps
-----------

If you are not familar with **PSSR** or super-resolution, understand the :doc:`principles`.

For usage of **PSSR** beyond the extents of the demo, learn how to implement your own :doc:`workflow`.

Full reference and explanations of all **PSSR** tools is available in :doc:`API Reference <../reference/api>`.

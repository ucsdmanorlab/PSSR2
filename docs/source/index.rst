PSSR2 Documentation
====================

**PSSR2** *(Point-Scanning Super-Resolution 2)* is a user-friendly `PyTorch <https://pytorch.org/>`_-based workflow for super-resolution tasks using microscopy images.
This is the official reimplementation and extention of the methods described in the original paper: `Deep learning-based point-scanning super-resolution imaging <https://www.nature.com/articles/s41592-021-01080-z>`_.
**PSSR2** contains various improvements from its predecessor, which are elaborated in our preprint:
`PSSR2: a user-friendly Python package for democratizing deep learning-based point-scanning super-resolution microscopy <https://www.biorxiv.org/content/10.1101/2024.06.16.599221v1>`_.

If you have never used **PSSR2** before, :doc:`guide/start` outlines installation and basic usage.
Full reference and explanations of all **PSSR2** tools is available in :doc:`API Reference <reference/api>`.

This package is under continuous development. All code can be found at `https://github.com/ucsdmanorlab/PSSR2 <https://github.com/ucsdmanorlab/PSSR2>`_.
If you experience any bugs, unexpected behaviors, or have any suggestions, make sure to `open a ticket <https://github.com/ucsdmanorlab/PSSR2/issues>`_.

Sample data and pretrained models can currently be found `here <https://drive.google.com/drive/folders/1q6a2Z6gRG3Vnx8BM3OW7Y35myw-0f0_H>`__.

User Guide
-----------

The User Guide covers the installation, usage, and principles of **PSSR2**. 

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guide/start
   guide/principles
   guide/workflow
   guide/dataloading


API Reference
--------------

The API Reference covers full documentation of all **PSSR2** modules.

.. toctree::
   :titlesonly:
   :caption: API Reference

   reference/api

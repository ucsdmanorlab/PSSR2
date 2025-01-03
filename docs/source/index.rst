PSSR2 Documentation
====================

**PSSR2** *(Point-Scanning Super-Resolution 2)* is a user-friendly `PyTorch <https://pytorch.org/>`_-based workflow for super-resolution tasks using microscopy images.
This is the official reimplementation and extention of the methods described in the original paper: `Deep learning-based point-scanning super-resolution imaging <https://www.nature.com/articles/s41592-021-01080-z>`_.
**PSSR2** contains various improvements from its predecessor, which are elaborated in the following manuscript:
`PSSR2: a user-friendly Python package for democratizing deep learning-based point-scanning super-resolution microscopy <https://bmcmethods.biomedcentral.com/articles/10.1186/s44330-024-00020-5>`_.
If you utilize **PSSR2** in your publication, please consider `citing it <https://bmcmethods.biomedcentral.com/articles/10.1186/s44330-024-00020-5#citeas>`_.

The functionality of **PSSR2** is accessible in three ways:

- Directly through the `Python package <https://pypi.org/project/pssr>`_
- Through the integrated :doc:`Command Line Interface <reference/CLI>`
- Through the integrated :doc:`Napari plugin <guide/napari>`

If you have never used **PSSR2** before, :doc:`guide/start` outlines installation and basic usage.
Full reference and explanations of all **PSSR2** tools is available in :doc:`API Reference <reference/api>`.

This package is under continuous development. All code can be found at `https://github.com/ucsdmanorlab/PSSR2 <https://github.com/ucsdmanorlab/PSSR2>`_.
If you experience any bugs, unexpected behaviors, or have any suggestions, make sure to `open a ticket <https://github.com/ucsdmanorlab/PSSR2/issues>`_.

User Guide
-----------

The User Guide covers the installation and usage of **PSSR2**. 

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guide/start
   guide/principles
   guide/workflow
   guide/dataloading
   guide/napari


API Reference
--------------

The API Reference covers full documentation of all **PSSR2** modules.

.. toctree::
   :titlesonly:
   :caption: API Reference

   reference/api

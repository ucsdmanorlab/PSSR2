PSSR Documentation
===================

**PSSR** *(Point-Scanning Super-Resolution)* is a standardized `PyTorch <https://pytorch.org/>`_-based workflow for super-resolution tasks using microscopy images.
This is the official reimplementation of the methods described in the original paper: `Deep learning-based point-scanning super-resolution imaging <https://www.nature.com/articles/s41592-021-01080-z>`_,
containing various improvements and new features.

If you have never used **PSSR** before, :doc:`guide/start` outlines installation and basic usage.
Full reference and explanations of all **PSSR** tools is available in :doc:`API Reference <reference/api>`.

This package is under continuous development. All code can be found at `https://github.com/ucsdmanorlab/PSSR <https://github.com/ucsdmanorlab/PSSR>`_.
If you experience any bugs, unexpected behaviors, or have any suggestions, make sure to `open a ticket <https://github.com/ucsdmanorlab/PSSR/issues>`_.


User Guide
-----------

The User Guide covers the installation, usage, and principles of **PSSR**. 

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guide/start
   guide/principles
   guide/workflow
   guide/dataloading


API Reference
--------------

The API Reference covers full documentation of all **PSSR** modules.

.. toctree::
   :titlesonly:
   :caption: API Reference

   reference/api

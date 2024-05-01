models
=======

A repository of super-resolution models that conform to **PSSR** conventions.

Any `PyTorch <https://pytorch.org/>`_ based super-resolution model can be used in place of these if desired.

These models do not necessarily take into account uneven max-poolings therefore the maximum amount of hidden layers (e.g. ``hidden=[128, 256]`` is 2 layers)
cannot exceed the amount of times the *low-resolution* input can be evenly max-pooled to half resolution.
Because of this, users are advised to keep image resolutions to a power of 2.


Modules
--------

.. code-block:: python

   from pssr.models import ...

.. toctree::
   :titlesonly:

   models/ResUNet
   models/RDResUNet

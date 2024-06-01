SSIMLoss
=========

Using any ``mix`` value less than 1 applies corrected L1 loss in addition to SSIM loss, as derived from `Zhao et al., 2018 <https://arxiv.org/pdf/1511.08861.pdf>`_.

.. code-block:: python

   from pssr.util import SSIMLoss

.. autofunction:: pssr.util.SSIMLoss.__init__
   
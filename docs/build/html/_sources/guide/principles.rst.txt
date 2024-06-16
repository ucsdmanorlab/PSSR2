Principles of PSSR
===================

.. note::

   This section can be considered an abbreviation of important concepts covered in the full **PSSR** paper,
   `Deep learning-based point-scanning super-resolution imaging <https://www.nature.com/articles/s41592-021-01080-z>`_.

   This page will soon be updated to contain information contained within the **PSSR2** preprint!


Why Use PSSR
-------------

Data aquisition is often costly and time consuming in microscopy, requiring the balance of numerous parameters to capture an ideal image.
Additionally, high resolution/exposure data can be destructive of samples, especially when measuring fluorescence.

**PSSR** allows these limitations to be mitigated by super-resolving a *low-resolution* image to a *high-resolution* image.

|

Consider a set of *high-low-resolution* image pairs where the *high-resolution* images have 4 times resolution of their corresponding *low-resolution* images.
Although the *high-resolution* images may contain 16 times the amount of pixels, they do not contain 16 times the amount of information about the sample.
For example, some pixels may be redundant in the *high-resolution* images while their structure may be easily inferable in the *low-resolution* images.

This allows super-resolution techniques to super-resolve *low-resolution* images into their *high-resolution* counterparts without hallucinations
(when benchmarked against human experts) provided a suitable model is trained.


The Crappifier
---------------

**PSSR** is based off conventional super-resolution techniques, but differs from the norm in its implementation.
Perhaps the biggest principle of **PSSR** is that of the :doc:`../reference/crappifiers/Crappifier`.

|

In order to train a super-resolution model, *high-low-resolution* image pairs must be provided to the model as the input and target respectively.
However, due to the difficulty and costs of acquiring *high-low-resolution* image pairs for training, this approach is not just impractical,
but may be impossible for non-static live samples such as subcellular organelles.

To circumvent this, a :doc:`../reference/crappifiers/Crappifier` is used during training to computationally degrade *high-resolution* images into their *low-resolution* image counterparts.
This process of crappification does not require *low-resolution* images to be acquired for training.

It first downsamples the *high-resolution* image to a lower resolution, before injecting a layer of noise that approximates the sampling noise of a true *low-resolution* image.

|

Although the model is not trained on true *low-resolution* images, the generated *high-low-resolution* image pairs are representative enough of the ground truth
that the model can make accurate inferences on real world data.

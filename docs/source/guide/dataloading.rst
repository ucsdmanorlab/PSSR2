Advanced Dataloading
=====================

Multidimensional Images
------------------------

**PSSR2** supports the use of multidimensional images.

By default the dataset will provide all frames of each image/tile in a single array.
For example, a single image with a horizontal and vertical resolution of 512 and a Z resolution of 20 will be treated as a single 20x512x512 array.

However, it is often preferred that a model is trained using a lower number of frames.
In all **PSSR2** datasets, the ``n_frames`` argument can be specified as the number of image frames to use at once.
For example, with ``n_frames=5``, the same image that was previously treated as a single 20x512x512 array would now be treated as 4 different 5x512x512 arrays.

The value of ``n_frames`` in the dataset should correspond the value of ``channels`` in the model.
That is, the number of output frames in the dataset should correspond to the amount of input channels in the model.

.. note::

   Images are separated into training/validation sets by tile (x and y position), disregarding additional dimensions.
   This prevents overfitting as different frames from the same tile may be overly similar to one another.

|

Images with Many Dimensions
++++++++++++++++++++++++++++

Typically, images are passed to a neural network with 2 spacial dimensions and a channel dimension.
For simplicity, all additional non-horizontal/vertical image dimensions are flattened to a single channel dimension by the dataset.
For example, an image with a T (time) resolution of 10 and a Z resolution of 5 would have 50 frames (10 T frames x 5 Z frames).

The order of the dimensions being combined has an effect on how the frames are handled.
In all **PSSR2** datasets, the ``stack`` argument determines this order:

-  By default, ``stack="TZ"``, meaning the Z dimension is flattened after the T dimension and changes more often.
   For the previous example of the 10 T frame x 5 Z frame image, ``n_frames=10`` would make the dataset load 5 Z frames at 2 different points in T.

-  Setting ``stack="ZT"`` reverses this behavior, so the T dimension is flattened after the Z dimension and changes more often.
   The image would therefore be treated as a 5 Z frame x 10 T frame image, and ``n_frames=20`` would make the dataset load 10 different T frames at 2 different points in Z.

Color channels are also considered an additional dimension that are flattened after all other dimensions if present.

The dimensions that are flattened last are essentially "bound", meaning that as long as the value of ``n_frames`` is a multiple of the sizes of every non-first dimension,
the channels will all stay together in proper order.

For example, consider a single 512x512 image with 10 T channels x 5 Z channels x 3 C (color) channels.
With ``n_frames=30`` (5 Z frames x 3 C frames x 2 T frames) would be treated as 5 different 30x512x512 arrays at different points in T.

|

2.5-Dimensional Data
+++++++++++++++++++++

The number of frames in the low-resolution image does not necessarily have to correspond to that of the high-resolution image.
Consider temporal super-resolution, where multiple images at different points in time are used to produce a higher quality image at a central point in time.

To do this, ``n_frames`` can be specified as a list with 2 elements, corresponding to the number frames in both high-resolution and low-resolution images.
For example, with a high-resolution training image with 20 T frames, a value of ``n_frames=[5,1]`` would process data equivalently to ``n_frames=5``,
but only return the central T frame for the high-resolution image while returning all 5 T frames for the low-resolution image.


Image Denoising
----------------

While the **PSSR2** package is mainly used for image super-resolution, it can be used for image denoising as well.
Set the ``lr_scale=1`` with an equivalent ``scale`` value in the model, and ``hr_res`` equal to the denoising image resolution.

Multidimensional and 2.5-dimensional images can be used with this approach as well,
meaning that **PSSR2** can be used for things such as temporal super-resolution without necessarily needing to be used for image super-resolution as well.

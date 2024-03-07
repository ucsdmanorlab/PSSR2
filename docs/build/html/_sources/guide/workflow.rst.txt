Workflow
=========

.. note::

    This section explains how to use an example **PSSR** workflow, similar to that in the `demo <https://github.com/haydenstites/PSSR/blob/master/demo.py>`_.
    It does not necessarily apply to all use cases and is meant to be expanded upon.


Training a Basic Model
-----------------------

Before diving into the code, we will first specify our imports.

.. code-block:: python

    import torch
    from pssr.data import ImageDataset
    from pssr.crappifiers import Poisson
    from pssr.models import ResUNet
    from pssr.loss import SSIMLoss
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from pssr.train import train_paired

|

Defining Objects
+++++++++++++++++

Before we define our dataset we must first define our :doc:`../reference/crappifiers/Crappifier`, as it is utilized by our dataset.

.. code-block:: python

    crappifier = Poisson(gain=0, intensity=1)

This sets the ``crappifier`` variable to an instance of the :doc:`../reference/crappifiers/Poisson` crappifier with default arguments.
It will be used to synthetically generate *low-resolution* images to train on, given the *high-resolution* images in our dataset.

|

Which dataset to use depends on the format of your images.

In this example, we will use :doc:`../reference/data/ImageDataset`, assuming that our images are already sliced.

.. code-block:: python

    dataset = ImageDataset("your/hr_path", hr_res=512, lr_scale=4, crappifier=crappifier, extension="tif")

This sets the ``dataset`` variable to an instance of :doc:`../reference/data/ImageDataset`, loading *high-resolution* ``.tif`` images from ``your/hr_path``.
The *high-resolution* images are specified to have a horizontal and vertical resolution of ``hr_res=512``.
If the images provided are not square or are of the wrong resolution, they will be cropped and/or rescaled to fit.

We provide the :doc:`../reference/crappifiers/Crappifier` we defined earlier as an argument that will
generate *low-resolution* images ``lr_scale=4`` times smaller than the *high-resolution* images, for a resolution of 128.

.. note::

    Users are advised to keep image resolutions to a power of 2, elaborated in :doc:`../reference/models`.

|

The last thing we need to define before training is our model.

.. code-block:: python

    model = ResUNet(
        hidden=[64, 128, 256, 512, 1024],
        scale=4,
        depth=3,
    )

This sets the ``model`` variable to an instance of :doc:`../reference/models/ResUNet`.
The ``scale`` argument sets the factor by which the input *low-resolution* images must be upscaled by, and should be equivalent to the ``lr_scale`` argument in our dataset.
The other arguments specify the number of channels per hidden layer, and the depth of each hidden layer (number of hidden convolutions).

|

Train Arguments
++++++++++++++++

As we are training on a synthetic paired *high-low-resolution* dataset, we will use the :doc:`../reference/train/train_paired` function.

For simplicity, we will define our arguments before beginning training.

|

We will first define our loss function.

.. code-block:: python

    loss_fn = SSIMLoss(mix=.8, ms=True)

Although MSE loss can be used to good results, :doc:`../reference/loss/SSIMLoss` can be used optimize visually significant elements of an image, and is often used in super-resolution tasks.
The ``mix`` argument controls the inverse contribution of corrected L1 loss, while the ``ms`` argument enables MS-SSIM, a more robust version of SSIM.

|

We also need to provide an optimizer.

.. code-block:: python

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optim, factor=0.1, patience=5, verbose=True)

This defines the optimizer of our model with starting learning rate of 1e-3.
By defining a scheduler, the learning rate of the optimizer will decay by ``factor=0.1`` after model performance doesn't improve for ``patience=5`` epochs.

|

And finally we define our miscellaneous arguments.

.. code-block:: python

    batch_size = 16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    kwargs = dict(
        shuffle = True,
        num_workers = 4,
        pin_memory = True,
    )

This sets our batch size and training device, along with our torch `DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_ arguments.
The batch size can be adjusted depending on the amount of memory available for training.

|

Training
+++++++++

We can now train our model using the :doc:`../reference/train/train_paired` function.

.. code-block:: python

    losses = train_paired(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        loss_fn=loss_fn,
        optim=optim,
        epochs=20,
        device=device,
        scheduler=scheduler,
        dataloader_kwargs=kwargs,
    )

While training, various metrics will be provided along with the loss to easily monitor progress.

Additionally, at the end of every epoch a collage will be saved to the ``preds`` folder containing
*low-resolution* crappified images, upscaled *high-resolution* predictions, and ground truth *high-resolution* images in that order.

|

After training is over, we should save our model for future use.

.. code-block:: python

    torch.save(model.state_dict(), "model.pth")

|

We can also plot the training losses returned by :doc:`../reference/train/train_paired` to see the progress of our model over time.

.. code-block:: python

    import matplotlib.pyplot as plt

    plt.plot(losses)


Using the Model for Predictions
--------------------------------

We now have our trained model, which takes in *low-resolution* input images and outputs upscaled *high-resolution* images.

There are now two things we can do with our trained model, use it for predictions, or benchmark it.

|

If you decide to run your model predictions in a separate file, you will want to load your trained model before proceeding with

.. code-block:: python

    model.load_state_dict(torch.load("model.pth"))

where ``model`` is an instance of the same architecture you used previously.

|

Predicting Images
++++++++++++++++++

To use our model, we will use the :doc:`../reference/predict/predict_images` function.

.. code-block:: python

    from pssr.predict import predict_images

|

During the training phase, we loaded *high-resolution* images to create synthetic *low-resolution* images using a :doc:`../reference/crappifiers/Crappifier`.
While predicting images, we will instead use experimentally acquired *low-resolution* images to predict upscaled *high-resolution* images.

We can do this by creating the same :doc:`../reference/data/ImageDataset`, only now we provide the path to our *low-resolution* images.

.. code-block:: python

    test_dataset = ImageDataset("your/lr_path", hr_res=512, lr_scale=4, extension="tif")

The *low-resolution* images are implied to have a horizontal and vertical resolution of 128 (``hr_res=512`` / ``lr_scale=4``).
A crappifier does not have to be specified, as it will not be used.

|

We can now use our model to upscale the *low-resolution* images.

.. code-block:: python

    predict_images(model, test_dataset, device)

This will super-resolve *high-resolution* images from our *low-resolution* images and save them to the ``preds`` folder.

|

Benchmarking the Model
+++++++++++++++++++++++

If you have a dataset containing aligned *high-low-resolution* pairs (every *high-resolution* image has an aligned *low-resolution* counterpart),
we can use :doc:`../reference/predict/test_metrics` to quantify the performance of our model on real world data.

.. note::

    Metrics can still be acquired from training datasets with only *high-resolution* images,
    but they will only represent training performance on crappified data and may not represent real world performance.

|

We can do this by creating a new :doc:`../reference/data/PairedImageDataset` instance, containing our *high-low-resolution* image pairs.

.. code-block:: python

    paired_dataset = PairedImageDataset("testdata/EM_pairs_crop/hr", "testdata/EM_pairs_crop/lr", hr_res=512, lr_scale=4)

The images in each folder should be properly aligned and have a similar naming/ordering scheme so that they are returned in the same order when that dataset is iterated.

|

We can then compute metrics for all images.

.. code-block:: python

    test_metrics(model, paired_dataset, device=device)

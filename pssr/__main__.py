import sys
sys.path.append("..")

import torch, argparse
from torch.nn import MSELoss
from pssr.models import ResUNet, ResUNetA, RDResUNet, RDResUNetA, SwinIR
from pssr.data import ImageDataset, SlidingDataset, PairedImageDataset, PairedSlidingDataset
from pssr.crappifiers import MultiCrappifier, Poisson, AdditiveGaussian, SaltPepper
from pssr.util import SSIMLoss, _tab_string
from pssr.train import train_paired
from pssr.predict import predict_images, test_metrics

IS_NAPARI = False

def _handle_declaration(arg, defaults, req=None):
    req = ", ".join(req)+", " if req is not None else ""

    if arg in defaults:
        expression = arg+f"({req})"
    else:
        expression = arg.split("(")[0]+f"({req}"+"(".join(arg.split("(")[1:])

    return eval(expression)

def parse():
    parser = argparse.ArgumentParser(prog="pssr", description="PSSR2 CLI for basic usage", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-t", "--train", action="store_true", help="enable train mode")

    parser.add_argument("-dp", "--data-path", type=str, help="specify dataset path")
    parser.add_argument("-dt", "--data-type", type=str, default="ImageDataset", help="specify dataset type")
    parser.add_argument("-mt", "--model-type", type=str, default="ResUNet", help="specify model type")
    parser.add_argument("-mp", "--model-path", type=str, help="specify model path")

    parser.add_argument("-e", "--epochs", type=int, default=10, help="specify number of training epochs")
    parser.add_argument("-b", "--batch-size", type=int, default=16, help="specify batch size")
    parser.add_argument("-lr", "--lr", type=float, default=1e-3, help="specify learning rate")
    parser.add_argument("-p", "--patience", type=int, default=3, help="specify learning rate decay patience")
    parser.add_argument("-mse", "--mse", action="store_true", help="use MSE loss instead of MS-SSIM loss")

    parser.add_argument("-cp", "--checkpoint", action="store_true", help="save model checkpoints during training")
    parser.add_argument("-sl", "--save-losses", action="store_true", help="save training losses")

    return parser

def main():
    parser = parse()
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        return
    args = parser.parse_args()

    if args.data_path is None:
        print("--data-path(-dp) must be provided")
        return
    elif args.model_path is None and not args.train:
        print("--model-path(-mp) must be provided in predict mode")
        return

    model = _handle_declaration(args.model_type, ["ResUNet", "ResUNetA", "RDResUNet", "RDResUNetA", "SwinIR"])
    dataset = _handle_declaration(args.data_type, ["ImageDataset", "SlidingDataset", "PairedImageDataset", "PairedSlidingDataset"], 
        req=[f"'{item.strip()}'" for item in args.data_path.split(",")] + (["val_split=1"] if not args.train else []))
    
    pssr_head(args.train, model, dataset, None, args.epochs, args.batch_size, args.lr, args.patience, args.mse, args.checkpoint, args.save_losses, args.model_path)
    print("\n")

def pssr_head(train, model, dataset, device, epochs, batch_size, lr, patience, loss_fn, checkpoint, losses, model_path, callbacks = None, stage = None, metrics = None):
    # Shared code with napari plugin
    if stage is not None:
        global IS_NAPARI
        IS_NAPARI = True

        global Status
        from pssr.napari.widgets import Status
        stage.emit(Status.LOADING.value)

        model = eval(model)
        dataset = eval(dataset)

    print(f"\nModel:\n{_tab_string(model.extra_repr())}")
    print(f"\nDataset:\n{_tab_string(str(dataset))}")

    if not IS_NAPARI:
        if torch.cuda.is_available():
            device = "cuda"
            print("\nCUDA enabled device detected, running on GPU.")
        else:
            device = "cpu"
            print("\nCUDA enabled device NOT detected, running on CPU.")
    else:
        if device == "cuda":
            if not torch.cuda.is_available(): raise ValueError("CUDA is specified, but CUDA enabled device is not detected")
            print("\nCUDA enabled, running on GPU.")
        else:
            print("\nCUDA disabled, running on CPU.")

    if model_path:
        if str(model_path) == ".": raise ValueError("Attempted to load model from checkpoint, but path is not provided")
        print(f"Loading {model.__class__.__name__} model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    
    if not IS_NAPARI:
        if train:
            func = lambda : _train_meta(model, dataset, device, epochs, batch_size, lr, patience, loss_fn, checkpoint, losses)
        else:
            func = lambda : _predict_meta(model, dataset, device)
    else:
        if train:
            func = lambda : _train_meta(model, dataset, device, epochs, batch_size, lr, patience, loss_fn, checkpoint, losses, callbacks, stage)
        else:
            func = lambda : _predict_meta(model, dataset, device, callbacks, stage, metrics)
    # Move model to cpu after completion, useful for napari
    _cpu_wrapper(func, model)

def _train_meta(model, dataset, device, epochs, batch_size, lr, patience, loss_fn, checkpoint, losses, callbacks = None, stage = None):
    if not IS_NAPARI:
        loss_fn = MSELoss() if loss_fn else SSIMLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.1, patience=patience, threshold=5e-3, verbose=True)
    checkpoint_dir = "checkpoints" if checkpoint else None

    kwargs = dict(
        num_workers = 4,
        pin_memory = True,
    )

    if IS_NAPARI:
        stage.emit(Status.PROGRESS_TRAIN.value)

    print("\nTraining model...")
    train_losses, val_losses = train_paired(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        loss_fn=loss_fn,
        optim=optim,
        epochs=epochs,
        device=device,
        scheduler=scheduler,
        checkpoint_dir=checkpoint_dir,
        dataloader_kwargs=kwargs,
        callbacks=callbacks,
    )
    print("\nTraining complete!")

    save_path = f"{model.__class__.__name__}_{dataset.hr_res//dataset.lr_scale}-{dataset.hr_res}_{val_losses[-1]:.4f}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved trained model to {save_path}")

    if losses:
        with open(f"{model.__class__.__name__}_train_losses_{val_losses[-1]:.4f}.txt", "w") as file:
            for loss in train_losses:
                file.write(f"{loss:.6f}\n")
        with open(f"{model.__class__.__name__}_val_losses_{val_losses[-1]:.4f}.txt", "w") as file:
            for loss in val_losses:
                file.write(f"{loss:.6f}\n")

def _predict_meta(model, dataset, device, callbacks = None, stage = None, plotter = None):
    if IS_NAPARI:
        stage.emit(Status.PROGRESS_PREDICT.value)

    print("\nPredicting images from low resolution...")
    predict_images(model, dataset, device, norm=not dataset.is_lr, out_dir="preds", callbacks=callbacks)

    if not dataset.is_lr:
        print("\nCalculating metrics...")
        metrics = test_metrics(model, dataset, device, avg=not IS_NAPARI, callbacks=callbacks)

        if IS_NAPARI:
            plotter.emit([metrics["psnr"], metrics["ssim"]])
            metrics = {metric:(sum(values)/len(values)) for metric, values in metrics.items()}

        print("\nMetrics:")
        for metric in metrics:
            print(f"{metric}: {metrics[metric]}")

def _cpu_wrapper(func, model):
    try:
        out = func()
    except Exception as error:
        model.to("cpu")
        raise error
    model.to("cpu")
    return out

if __name__ == "__main__":
    main()

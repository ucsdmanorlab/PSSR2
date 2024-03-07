import torch, argparse
from pssr.models import ResUNet
from pssr.data import ImageDataset, SlidingDataset # PairedImageDataset
from pssr.crappifiers import AdditiveGaussian, Poisson
from pssr.loss import SSIMLoss
from pssr.train import train_paired
from pssr.predict import predict_images, test_metrics

def _handle_declaration(arg, defaults, req=None):
    req = ", ".join(req)+", " if req is not None else ""

    if arg in defaults:
        expression = arg+f"({req})"
    else:
        expression = arg.split("(")[0]+f"({req}"+arg.split("(")[1]

    return eval(expression)

def parse():
    parser = argparse.ArgumentParser(description="PSSR CLI demo for basic usage")

    parser.add_argument("-t", "--train", action="store_true", help="enable train mode")

    parser.add_argument("-dp", "--data-path", type=str, help="specify dataset path")
    parser.add_argument("-dt", "--data-type", type=str, default="ImageDataset", help="specify dataset type e.g. ImageDataset")
    parser.add_argument("-mt", "--model-type", type=str, default="ResUNet", help="specify model type e.g. ResUNet")
    parser.add_argument("-mp", "--model-path", type=str, default="model.pth", help="specify model path")

    parser.add_argument("-e", "--epochs", type=int, default=10, help="specify number of epochs")
    parser.add_argument("-b", "--batch-size", type=int, default=16, help="specify batch size")
    parser.add_argument("-lr", "--lr", type=float, default=1e-3, help="specify learning rate")
    parser.add_argument("-p", "--patience", type=int, default=3, help="specify learning rate decay patience")

    return parser

if __name__ == "__main__":
    args = parse().parse_args()

    model = _handle_declaration(args.model_type, ["ResUNet"])
    dataset = _handle_declaration(args.data_type, ["ImageDataset", "SlidingDataset", "PairedImageDataset"], req=[f"'{args.data_path}'"] if args.train else [f"'{args.data_path}'", "val_split=1"])

    if torch.cuda.is_available():
        device = "cuda"
        print("CUDA enabled device detected, running on GPU.")
    else:
        device = "cpu"
        print("CUDA enabled device NOT detected, running on CPU.")

    kwargs = dict(
        shuffle = True,
        num_workers = 4,
        pin_memory = True,
    )

    if args.train:
        loss_fn = SSIMLoss(mix=.6, ms=True)
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.1, patience=args.patience, threshold=5e-3, verbose=True)

        print("\nTraining model...")
        losses = train_paired(
            model=model,
            dataset=dataset,
            batch_size=args.batch_size,
            loss_fn=loss_fn,
            optim=optim,
            epochs=args.epochs,
            device=device,
            scheduler=scheduler,
            log_frequency=50,
            dataloader_kwargs=kwargs,
        )
        print("\nTraining complete!")

        torch.save(model, args.model_path)
        print(f"Saved trained model to {args.model_path}")

    else:
        model.load_state_dict(torch.load(args.model_path))

        print("\nPredicting images from low resolution...")
        predict_images(model, dataset, device, "preds")

        if not dataset.is_lr:
            metrics = test_metrics(model, dataset, device=device, dataloader_kwargs=kwargs)

            print("\nMetrics:")
            for metric in metrics:
                print(f"{metric}: {metrics[metric]}")

import napari, warnings
import numpy as np
from qtpy.QtWidgets import QWidget, QVBoxLayout
from qtpy.QtCore import QObject, QThread, Signal, Qt
from magicgui.widgets import Container, ComboBox, PushButton, FileEdit, TextEdit, create_widget
from torch.nn import MSELoss
from enum import Enum
from PIL import Image
from ..__main__ import pssr_head
from ._util import ObjectEdit, SignalWrapper
from ..crappifiers import AdditiveGaussian, Poisson, SaltPepper
from ..models import ResUNet, RDResUNet, SwinIR
from ..data import ImageDataset, SlidingDataset, PairedImageDataset, PairedSlidingDataset
from ..util import SSIMLoss

# Matplotlib elements are only defined if installed, not a dependency
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    USE_PLOT = True
except:
    warnings.warn("matplotlib could not be imported, plotting features will be disabled.", stacklevel=2)
    USE_PLOT = False

class Status(Enum):
    IDLE_TRAIN = "Train Model"
    IDLE_PREDICT = "Predict Images"
    LOADING = "Loading..."
    PROGRESS_TRAIN = "Cancel Training"
    PROGRESS_PREDICT = "Cancel Predicting"

class PSSRWidget(QWidget):
    def __init__(self, is_train : bool, viewer : napari.Viewer):
        super().__init__()
        self.viewer = viewer

        self.model = ObjectEdit("Model", [ResUNet, RDResUNet, SwinIR])
        self.dataset = ObjectEdit("Dataset", [ImageDataset, SlidingDataset, PairedImageDataset, PairedSlidingDataset], hide_crappifier=not is_train)

        self.device = ComboBox(name="Device", choices=["cuda", "cpu"])
        self.model_path = FileEdit(name="Model Path")
        self.batch_size = create_widget(value=16, name="Batch Size")

        if is_train:
            self.epochs = create_widget(value=10, name="Epochs")
            self.lr = create_widget(value=0.001, name="Learning Rate", options=dict(step=1e-5))
            self.gamma = create_widget(value=0.5, name="Learning Rate Decay")
            self.loss_fn = ComboBox(name="Loss Function", choices=["MS-SSIM", "SSIM", "MSE"])
            self.checkpoint = create_widget(value=False, name="Save Checkpoints")
            self.losses = create_widget(value=False, name="Save Losses")
            self.resume = create_widget(value=False, name="Load Checkpoint")

            self.resume.changed.connect(lambda : setattr(self.model_path, "visible", self.resume.value))
            self.model_path.visible = self.resume.value

        self.params = Container()
        self.params.append(self.device)
        if is_train:
            self.params.append(self.epochs)
            self.params.append(self.batch_size)
            self.params.append(self.lr)
            self.params.append(self.gamma)
            self.params.append(self.loss_fn)
            self.params.append(self.checkpoint)
            self.params.append(self.losses)
            self.params.append(self.resume)
        else:
            pass
        self.params.append(self.model_path)

        self.trigger = PushButton(text=Status.IDLE_TRAIN.value if is_train else Status.IDLE_PREDICT.value)
        self.trigger.changed.connect(lambda : self.process_wrapper(is_train))

        self.console = TextEdit(value="")
        self.console.read_only = True
        self.console.hide()
        self.err_len = 0

        layout = QVBoxLayout()
        layout.addWidget(self.model)
        layout.addWidget(self.dataset)
        layout.addWidget(self.params.native)
        layout.addWidget(self.trigger.native)
        layout.addWidget(self.console.native)
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)

        self.hide = [self.model, self.dataset, self.params]
        self.plot = None
    
    def process_wrapper(self, train : bool):
        # Functions as cancel button when running
        if self.trigger.text in [Status.LOADING.value, Status.PROGRESS_TRAIN.value, Status.PROGRESS_PREDICT.value]:
            self.worker.abort = True
            return

        if train:
            self.worker = TrainProcess(
                model=self.model.object,
                dataset=self.dataset.object,
                device=self.device.value,
                epochs=self.epochs.value,
                batch_size=self.batch_size.value,
                lr=self.lr.value,
                gamma=self.gamma.value,
                loss_fn=self.loss_fn.value,
                checkpoint=self.checkpoint.value,
                losses=self.losses.value,
                model_path=self.model_path.value if self.resume.value else None,
            )
        else:
            self.worker = PredictProcess(
                model=self.model.object,
                dataset=self.dataset.object,
                device=self.device.value,
                model_path=self.model_path.value,
            )
        self.worker.stage.connect(lambda x : setattr(self.trigger, "text", x))
        self.worker.monitor.connect(lambda x : setattr(eval(f"self.viewer.layers['{x[0]}']", {"self":self}), "data", x[1]))
        self.worker.finished.connect(self._close_thread)
        self.worker.error.connect(lambda x : self._catch_error(x))

        # Capture sys.stdout to "console"
        self.capture = SignalWrapper(self.worker.run)
        self.capture.out.connect(lambda x : self._write_console(x))
        self.capture.err.connect(lambda x : self._write_console(x, err=True))
        self.console.value = ""

        self.thread = QThread()
        self.capture.moveToThread(self.thread)
        self.thread.started.connect(self.capture.capture)

        # Adjust UI
        for widget in self.hide:
            widget.hide()
        self.model.collapse.collapse()
        self.dataset.collapse.collapse()
        self.console.show()

        if USE_PLOT:
            if self.plot is not None:
                self.layout().removeWidget(self.plot)
                self.plot.hide()
                self.plot = None
            
            if train:
                self.plot = LossPlot(size=(6,4))
                self.worker.loss.connect(lambda x : self.plot.add_point(x))
                self.layout().addWidget(self.plot)
            else:
                self.plot = MetricsPlot(size=(6,4))
                self.worker.metrics.connect(lambda x : self.plot.show_metrics(x))
                self.layout().addWidget(self.plot)
                self.plot.hide()
        
        for name in ["LR", "PSSR", "HR"] if train else []:
            try:
                self.viewer.layers.remove(name)
            except:
                pass
            
            self.viewer.add_image(np.zeros(shape=[1]+[self.dataset.arguments["hr_res"]]*2, dtype=np.uint8), name=name)

        self.thread.start()
    
    def _write_console(self, line : str, err : bool = False):
        if len(line.strip()) > 0:
            if err:
                # Remove previous stderr lines
                self.console.value = "\n".join(self.console.value.strip().split("\n")[self.err_len:])
                self.err_len = len(line.split("\n"))
                line = line.strip()
            elif self.err_len > 0:
                self.err_len = 0

            if line[-1] == "\n":
                line = "\n"+line.strip()

            self.console.value = line+"\n"+self.console.value

    def _close_thread(self):
        self.thread.quit()
        self.thread.wait()

        for widget in self.hide:
            widget.show()
    
    def _catch_error(self, error : Exception):
        self._close_thread()
        self._write_console(repr(error))
        raise error

class TrainProcess(QObject):

    stage = Signal(str)
    monitor = Signal(list)

    finished = Signal(bool)
    error = Signal(Exception)

    if USE_PLOT:
        loss = Signal(float)

    def __init__(self, model, dataset, device, epochs, batch_size, lr, gamma, loss_fn, checkpoint, losses, model_path):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.loss_fn = loss_fn
        self.checkpoint = checkpoint
        self.losses = losses
        self.model_path = model_path

        if loss_fn == "MS-SSIM":
            self.loss_fn = SSIMLoss()
        elif loss_fn == "SSIM":
            self.loss_fn = SSIMLoss(ms=False)
        else:
            self.loss_fn = MSELoss()

        self.abort = False

    def run(self):
        try:
            pssr_head(
                train=True,
                model=self.model,
                dataset=self.dataset,
                device=self.device,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                gamma=self.gamma,
                loss_fn=self.loss_fn,
                checkpoint=self.checkpoint,
                losses=self.losses,
                model_path=self.model_path,
                callbacks=[self._abort_callback, self._viewer_callback]+([self._plot_callback] if USE_PLOT else []),
                stage=self.stage,
            )

            self.finished.emit(True)

        except Exception as error:
            self.error.emit(error)

        # Triggers before exception
        finally:
            self.stage.emit(Status.IDLE_TRAIN.value)
    
    def _abort_callback(self):
        if self.abort:
            raise InterruptedError("Model training cancelled by user")

    def _viewer_callback(self, train_locals):
        batch_idx, log_frequency, progress = train_locals["batch_idx"], train_locals["log_frequency"], train_locals["progress"]
        if batch_idx % log_frequency == 0 or batch_idx == len(progress) - 1:
            data = train_locals["last_full"] if batch_idx == len(progress) - 1 else [train_locals["lr"].cpu(), train_locals["hr_hat"].cpu(), train_locals["hr"].cpu()]
            lr, hr_hat, hr = [np.clip(image.detach().numpy(), 0, 255).astype(np.uint8) for image in data]
            channels = max([lr.shape[1], hr_hat.shape[1], hr.shape[1]])

            for name, batched in zip(["LR", "PSSR", "HR"], [lr, hr_hat, hr]):
                if name == "LR":
                    batched = np.stack([[Image.fromarray(channel).resize(hr.shape[-2:], Image.Resampling.NEAREST) for channel in image] for image in lr])

                collage = self._collage_images(batched)
                if collage.shape[0] == 1 and collage.shape[0] < channels:
                    collage = np.repeat(collage, channels, axis=0)

                self.monitor.emit([name, collage])

    if USE_PLOT:
        def _plot_callback(self, train_locals):
            batch_idx, log_frequency, progress = train_locals["batch_idx"], train_locals["log_frequency"], train_locals["progress"]
            if batch_idx % log_frequency == 0 or batch_idx == len(progress) - 1:
                self.loss.emit(train_locals["loss"].item())
    
    def _collage_images(self, batched):
        n_rows = int(np.sqrt(batched.shape[0]))
        n_cols = batched.shape[0]//n_rows

        image_size = batched.shape[-1]
        collage = np.zeros([batched.shape[1], n_rows*image_size, n_cols*image_size])

        for idx in range(n_rows*n_cols):
            row = idx // n_cols
            col = idx % n_cols
            collage[:, row*image_size:(row+1)*image_size, col*image_size:(col+1)*image_size] = batched[idx]
        return collage

class PredictProcess(QObject):

    stage = Signal(str)
    monitor = Signal(list)

    finished = Signal(bool)
    error = Signal(Exception)

    if USE_PLOT:
        metrics = Signal(list)

    def __init__(self, model, dataset, device, model_path):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.device = device
        self.model_path = model_path

        self.abort = False

    def run(self):
        try:
            pssr_head(
                train=False,
                model=self.model,
                dataset=self.dataset,
                device=self.device,
                epochs=None,
                batch_size=None,
                lr=None,
                gamma=None,
                loss_fn=None,
                checkpoint=None,
                losses=None,
                model_path=self.model_path,
                callbacks=[self._abort_callback],
                stage=self.stage,
                metrics=self.metrics,
            )

            self.finished.emit(True)

        except Exception as error:
            self.error.emit(error)

        finally:
            self.stage.emit(Status.IDLE_PREDICT.value)
    
    def _abort_callback(self):
        if self.abort:
            raise InterruptedError("Prediction cancelled by user")

if USE_PLOT:
    class LossPlot(FigureCanvasQTAgg):
        def __init__(self, size):
            fig = Figure(figsize=size)
            super().__init__(fig)

            self.ax = fig.add_subplot()
            self.ax.set_title("Training Loss")
            self.ax.tick_params(
                axis="x",
                which="both",
                bottom=False,
                top=False,
                labelbottom=False,
            )

            fig.set_tight_layout(True)

            self.pos = -1
            self.last = None
        
        def add_point(self, value):
            # Skip plotting ugly first batch
            if self.pos > 0:
                self.ax.plot([self.pos-1, self.pos], [self.last, value], c="blue")
                self.draw()

            self.pos += 1
            self.last = value
    
    class MetricsPlot(FigureCanvasQTAgg):
        def __init__(self, size):
            fig = Figure(figsize=size)
            super().__init__(fig)

            self.ax_psnr = fig.add_subplot(121)
            self.ax_psnr.set_title("PSNR")

            self.ax_ssim = fig.add_subplot(122)
            self.ax_ssim.set_title("SSIM")

            fig.set_tight_layout(True)
        
        def show_metrics(self, data):
            self._boxplot(self.ax_psnr, data[0])
            self._boxplot(self.ax_ssim, data[1])
            self.show()
        
        def _boxplot(self, ax, data):
            ax.boxplot(data, showfliers=False)
            x = np.random.normal(1, 0.02, size=len(data))
            ax.plot(x, data, ".", alpha=0.5)

# Instances accessed by napari
class TrainWidget(PSSRWidget):
    def __init__(self, viewer : napari.Viewer):
        super().__init__(is_train=True, viewer=viewer)

class PredictWidget(PSSRWidget):
    def __init__(self, viewer : napari.Viewer):
        super().__init__(is_train=False, viewer=viewer)

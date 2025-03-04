import inspect
from qtpy.QtWidgets import QWidget, QVBoxLayout
from qtpy.QtCore import QObject, Signal
from superqt import QCollapsible
from magicgui.widgets import Container, ComboBox, ListEdit, CheckBox, LineEdit, FileEdit, PushButton, create_widget
from magicgui.type_map import get_widget_class
from contextlib import redirect_stdout, redirect_stderr
from functools import partial
from pathlib import Path
from ..crappifiers import AdditiveGaussian, Poisson, SaltPepper
from ..models import ResUNet, RDResUNet, SwinIR
from ..data import ImageDataset, SlidingDataset, PairedImageDataset, PairedSlidingDataset

ADVANCED_PARAMS = {
    ResUNet : list(range(4,6+1)),
    RDResUNet : list(range(4,6+1)) + list(range(12,16+1)),
    SwinIR : list(range(8,19+1)),
}

IGNORE_PARAMS = {
    SwinIR : [15],
    ImageDataset : [3,11],
    SlidingDataset : [3,15],
    PairedImageDataset : [9],
    PairedSlidingDataset : [13],
}

class ObjectEdit(QWidget):
    def __init__(self, title : str, objects : list, hide_crappifier : bool = False):
        super().__init__()
        
        self.collapse = QCollapsible(title)

        self.type = ComboBox(name="Type", choices=[item.__name__ for item in objects])
        self.type.changed.connect(self._clear_arguments)
        self.collapse.addWidget(self.type.native)

        self.arg_container = Container()
        self.collapse.addWidget(self.arg_container.native)

        # If object is dataset picker
        if any(item in objects for item in [ImageDataset, SlidingDataset]):
            self.crappifier = ObjectEdit("Crappifier", [AdditiveGaussian, Poisson, SaltPepper])
            self.crappifier.type.changed.connect(lambda : self._set_arguments("crappifier", self.crappifier.object, raw=True))
            self.crappifier.arg_container.changed.connect(lambda : self._set_arguments("crappifier", self.crappifier.object, raw=True))

            self.type.changed.connect(self._assert_crappifier)
            self.collapse.addWidget(self.crappifier)
            if hide_crappifier:
                self.crappifier.collapse.collapse()

        self.advanced_container = Container()
        self.advanced_collapse = QCollapsible("Advanced Options")
        self.advanced_collapse.addWidget(self.advanced_container.native)
        self.collapse.addWidget(self.advanced_collapse)

        # self.debug = PushButton(text="DEBUG: Get Object")
        # self.debug.changed.connect(lambda x : print(self.object))
        # self.collapse.addWidget(self.debug.native)

        self.collapse.expand()
        self._clear_arguments()

        layout = QVBoxLayout()
        layout.addWidget(self.collapse)
        self.setLayout(layout)
    
    @property
    def object(self):
        return f"{self.type.current_choice}({', '.join([f'{key}={value}' for key, value in self.arguments.items()])})"

    def _clear_arguments(self):
        self.arguments = {}
        self.arg_container.clear()
        self.advanced_container.clear()

        choice = eval(self.type.current_choice)
        spec = inspect.getfullargspec(choice)
        advanced_idx = ADVANCED_PARAMS.get(choice, [])
        ignore_idx = IGNORE_PARAMS.get(choice, [])

        # Parameters without a default are None
        defaults = [None]*(len(spec.args[1:])-len(spec.defaults)) + list(spec.defaults)

        for idx, (arg, default) in enumerate(zip(spec.args[1:], defaults)):
            if idx in ignore_idx:
                continue

            widget_type = get_widget_class(annotation=spec.annotations[arg])[0]

            if widget_type is not ListEdit:
                if widget_type not in [CheckBox, LineEdit, FileEdit]:
                    options = dict(max=2**14, min=-1)
                if widget_type is FileEdit:
                    options = dict(mode="d")
                else:
                    options = {}
                widget = create_widget(value=default, annotation=spec.annotations[arg], name=arg, options=options)
            elif str(spec.annotations[arg]).count("list") == 1:
                if type(default) is not list and default:
                    default = [default]
                widget = ListEdit(value=default if default is not None else [0], name=arg, options=(dict(max=2**14, min=-1) if any(item in str(spec.annotations[arg]) for item in ["int", "float"]) else {}))
                if default is None:
                    widget._pop_value()
            else:
                if type(default) is not list and default:
                    default = [default]
                widget = _LargeList(value=default if default is not None else [[0]], name=arg, options=dict(max=2**14, min=-1))
                widget._pop_value()

            widget.changed.connect(partial(self._set_arguments, arg))
            self._set_arguments(arg, default)

            if idx in advanced_idx:
                self.advanced_container.append(widget)
            else:
                self.arg_container.append(widget)
        
        # Crappifier widget already exists, only need to get current state
        if choice in [ImageDataset, SlidingDataset]:
            self._set_arguments("crappifier", self.crappifier.object, raw=True)

        if len(advanced_idx) > 0:
            self.advanced_collapse.show()
        else:
            self.advanced_collapse.hide()
    
    def _set_arguments(self, name, value, raw=False):
        self.arguments[name] = (value if not issubclass(type(value), (str, Path)) or raw else f'"{value}"') if value != [] else None

    def _assert_crappifier(self):
        if eval(self.type.current_choice) in [ImageDataset, SlidingDataset]:
            self.crappifier.show()
        else:
            self.crappifier.hide()

class _LargeList(ListEdit):
    def __init__(self, value, name : str, options : dict = None):
        super().__init__(name=name, layout="vertical")
        self.list_options = {} if options is None else options

        head = self._make_list(value)
        for item in head:
            idx = (len(self)-2)
            self.insert(idx, item)

    def _append_value(self, widget = None):
        if widget == None:
            widget=ListEdit(value=[0], options=self.list_options)
        idx = len(self)-2
        self.insert(idx, widget)
        self[idx].changed.connect(lambda : self.changed.emit(self.value))

        self.changed.emit(self.value)
    
    def _make_list(self, value):
        if type(value[0]) is list:
            return [self._make_list(item) for item in value]
        
        return ListEdit(value=value, options=self.list_options)
    
class SignalWrapper(QObject):

    out = Signal(str)
    err = Signal(str)

    def __init__(self, func):
        super().__init__()
        self.func = func

    def capture(self):
        with redirect_stdout(_SignalCapture(self.out)):
            with redirect_stderr(_SignalCapture(self.err)):
                self.func()
    
class _SignalCapture():
    def __init__(self, signal):
        self.signal = signal

    def write(self, text):
        # Called on every captured out
        self.signal.emit(text)

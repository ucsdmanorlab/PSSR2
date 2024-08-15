import torch
from pssr.predict import predict_images, predict_collage, test_metrics
from pssr.data import ImageDataset
from pssr.models import ResUNet
from _util import get_shape, make_tifs, HR_RES

HR_SHAPE = get_shape(HR_RES)

def test_predict_images(tmp_path):
    make_tifs(tmp_path/"images", get_shape(HR_RES, batch=5))
    dataset = ImageDataset(tmp_path/"images", val_split=1)
    model = ResUNet()

    predict_images(model, dataset, out_dir=tmp_path)
    predict_images(model, dataset, out_dir=tmp_path, norm=True)

def test_predict_collage(tmp_path):
    make_tifs(tmp_path/"images", get_shape(HR_RES, batch=5))
    dataset = ImageDataset(tmp_path/"images", val_split=1)
    model = ResUNet()
    
    predict_collage(model, dataset, out_dir=tmp_path)
    predict_collage(model, dataset, out_dir=tmp_path, norm=True)

def test_test_metrics(tmp_path):
    make_tifs(tmp_path/"images", get_shape(HR_RES, batch=5))
    dataset = ImageDataset(tmp_path/"images", val_split=1)
    model = ResUNet()

    out = test_metrics(model, dataset)
    assert len(out) == 4

    test_metrics(model, dataset, norm=True)
    test_metrics(model, dataset, avg=False)

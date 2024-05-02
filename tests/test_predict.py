import torch
from pssr.predict import predict_images, predict_collage, test_metrics, normalize_preds
from pssr.data import ImageDataset
from pssr.models import ResUNet
from _util import get_shape, get_image, make_tifs, HR_RES, LR_RES

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

def test_normalize_preds(tmp_path):
    make_tifs(tmp_path/"images", get_shape(HR_RES, batch=5))
    dataset = ImageDataset(tmp_path/"images", val_split=1)
    model = ResUNet()

    hr_norm, hr_hat_norm = normalize_preds(hr=dataset[0][0].unsqueeze(0), hr_hat=model(dataset[0][1].unsqueeze(0)).detach())
    assert hr_norm.shape == hr_hat_norm.shape == (1, 1, HR_RES, HR_RES)

    hr_norm, hr_hat_norm = normalize_preds(hr=get_image(HR_SHAPE), hr_hat=get_image(HR_SHAPE))
    assert hr_norm.shape == hr_hat_norm.shape == (2, 1, HR_RES, HR_RES)

    hr_norm, hr_hat_norm = normalize_preds(hr=get_image(HR_SHAPE)[0], hr_hat=get_image(HR_SHAPE)[0])
    assert hr_norm.shape == hr_hat_norm.shape == (1, HR_RES, HR_RES)

    hr_norm, hr_hat_norm = normalize_preds(hr=get_image(HR_SHAPE)[0], hr_hat=get_image(get_shape(LR_RES, batch=0)))
    assert hr_norm.shape == (1, HR_RES, HR_RES)
    assert hr_hat_norm.shape == (1, LR_RES, LR_RES)

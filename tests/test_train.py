import torch
from pssr.train import train_paired, approximate_crappifier
from pssr.data import ImageDataset
from pssr.models import ResUNet
from pssr.crappifiers import AdditiveGaussian
from skopt.space import Real
from _util import get_shape, make_tifs, HR_RES

def test_train_paired(tmp_path):
    make_tifs(tmp_path/"images", get_shape(HR_RES, batch=5))
    dataset = ImageDataset(tmp_path/"images")
    model = ResUNet()

    losses = train_paired(model, dataset, batch_size=1, loss_fn=torch.nn.MSELoss(), optim=torch.optim.AdamW(model.parameters(), lr=1e-3), epochs=1, collage_dir=tmp_path)
    assert len(losses) == 2

def test_approximate_crappifier(tmp_path):
    make_tifs(tmp_path/"images", get_shape(HR_RES, batch=5))
    crappifier = AdditiveGaussian
    dataset = ImageDataset(tmp_path/"images", crappifier=crappifier())

    space = [Real(0, 15), Real(-10, 10)]
    result = approximate_crappifier(crappifier, space, dataset, opt_kwargs=dict(n_calls=5, n_initial_points=5))
    assert len(result.x) == len(space)
    assert type(crappifier(*result.x)) is crappifier
    
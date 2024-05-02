from pssr.crappifiers import MultiCrappifier, AdditiveGaussian, Poisson, SaltPepper
from _util import get_shape, get_image, LR_RES

LR_SHAPE = get_shape(LR_RES)

def test_crappifiers():
    kwargs_list = [
        {},
        dict(intensity=2),
        dict(intensity=0.5),
        dict(gain=10),
        dict(gain=-10),
        dict(spread=0.5),
    ]
    for crappifier in [AdditiveGaussian, Poisson, SaltPepper]:
        for kwargs in kwargs_list:
            out = crappifier(**kwargs)(get_image(LR_SHAPE))
            assert out.shape == LR_SHAPE, f"Crappifier {crappifier.__name__} is broken!"

def test_multi():
    crappifier = MultiCrappifier(AdditiveGaussian(), Poisson(), SaltPepper())
    out = crappifier(get_image(LR_SHAPE))
    assert out.shape == LR_SHAPE, f"Crappifier {crappifier.__name__} is broken!"

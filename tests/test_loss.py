from pssr.loss import SSIMLoss
from _util import get_shape, get_image, HR_RES

HR_SHAPE = get_shape(HR_RES)

def test_ssimloss():
    kwargs_list = [
        {},
        dict(mix=1),
        dict(mix=0),
        dict(ms=False),
    ]
    for kwargs in kwargs_list:
        out = SSIMLoss(**kwargs)(get_image(HR_SHAPE, tensor=True), get_image(HR_SHAPE, tensor=True))
        assert tuple(out.shape) == ()

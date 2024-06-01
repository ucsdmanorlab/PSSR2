from pssr.models import ResUNet, ResUNetA, RDResUNet, RDResUNetA
from _util import get_shape, get_image, LR_RES, HR_RES

def test_resunet():
    kwargs_list = [
        {},
        dict(channels=[3,3]),
        dict(channels=[3,1]),
        dict(dilations=[[1,3,15,31],[1,3,15],[1,3],[1],[1]]),
        dict(pool_sizes=[1,2,4,8]),
        dict(pool_sizes=[1,2,4,8], encoder_pool=True),
    ]

    for kwargs in kwargs_list:
        model = ResUNet(**kwargs)

        channels = kwargs.get("channels", [1,1])
        lr_shape, hr_shape = get_shape(LR_RES, channels[0]), get_shape(HR_RES, channels[1])

        out = model(get_image(lr_shape, tensor=True))
        assert tuple(out.shape) == hr_shape
    
    model = ResUNetA()
    assert model is not None

def test_rdresunet():
    kwargs_list = [
        {},
        dict(channels=[3,3]),
        dict(channels=[3,1]),
        dict(dilations=[[1],[1],[1,3],[1,3,15]]),
        dict(pool_sizes=[1,2,4,8]),
        dict(pool_sizes=[1,2,4,8], encoder_pool=True),
    ]

    for kwargs in kwargs_list:
        model = RDResUNet(**kwargs)

        channels = kwargs.get("channels", [1,1])
        lr_shape, hr_shape = get_shape(LR_RES, channels[0]), get_shape(HR_RES, channels[1])

        out = model(get_image(lr_shape, tensor=True))
        assert tuple(out.shape) == hr_shape
    
    model = RDResUNetA()
    assert model is not None

# TODO: test_swinir

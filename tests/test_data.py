from pssr.data import ImageDataset, SlidingDataset, PairedImageDataset, PairedSlidingDataset
from _util import get_shape, make_tifs, HR_RES, LR_RES, CROP_RES

N_IMAGES = 5
N_CHANNELS = 10
N_FRAMES = 2
TILE_MULT = 2

def test_imagedataset(tmp_path):
    # Single frame
    make_tifs(tmp_path/"sf", get_shape(HR_RES, batch=N_IMAGES))
    dataset = ImageDataset(tmp_path/"sf")
    assert str(dataset)
    assert len(dataset) == N_IMAGES

    hr, lr = dataset[0]
    assert tuple(hr.shape) == get_shape(HR_RES, batch=0)
    assert tuple(lr.shape) == get_shape(LR_RES, batch=0)
    
    # Multi frame
    make_tifs(tmp_path/"mf", get_shape(HR_RES, batch=N_IMAGES, channels=N_CHANNELS))
    dataset = ImageDataset(tmp_path/"mf", n_frames=N_FRAMES)
    assert str(dataset)
    assert len(dataset) == N_IMAGES * (N_CHANNELS // N_FRAMES)

    hr, lr = dataset[0]
    assert tuple(hr.shape) == get_shape(HR_RES, batch=0, channels=N_FRAMES)
    assert tuple(lr.shape) == get_shape(LR_RES, batch=0, channels=N_FRAMES)

    # LR mode
    make_tifs(tmp_path/"lr", get_shape(LR_RES, batch=N_IMAGES))
    dataset = ImageDataset(tmp_path/"lr", val_split=1)
    assert str(dataset)
    assert len(dataset) == N_IMAGES
    assert dataset.is_lr

    lr = dataset[0]
    assert tuple(lr.shape) == get_shape(LR_RES, batch=0)

    # Crop res
    make_tifs(tmp_path/"crop", get_shape(CROP_RES, batch=N_IMAGES))
    dataset = ImageDataset(tmp_path/"crop")
    assert str(dataset)
    assert len(dataset) == N_IMAGES
    assert dataset.crop_res == CROP_RES

    hr, lr = dataset[0]
    assert tuple(hr.shape) == get_shape(HR_RES, batch=0)
    assert tuple(lr.shape) == get_shape(LR_RES, batch=0)

def test_slidingdataset(tmp_path):
    # Single frame
    make_tifs(tmp_path/"sf", get_shape(HR_RES*TILE_MULT, batch=N_IMAGES))
    dataset = SlidingDataset(tmp_path/"sf", extension="tif", overlap=None, preload=False)
    assert str(dataset)
    assert len(dataset) == N_IMAGES * TILE_MULT**2

    hr, lr = dataset[0]
    assert tuple(hr.shape) == get_shape(HR_RES, batch=0)
    assert tuple(lr.shape) == get_shape(LR_RES, batch=0)

    # Preload
    dataset = SlidingDataset(tmp_path/"sf", extension="tif", overlap=None, preload=True)
    assert str(dataset)
    assert dataset.preload
    
    # Multi frame
    make_tifs(tmp_path/"mf", get_shape(HR_RES*TILE_MULT, batch=N_IMAGES, channels=N_CHANNELS))
    dataset = SlidingDataset(tmp_path/"mf", n_frames=N_FRAMES, extension="tif", overlap=None, preload=False)
    assert str(dataset)
    assert len(dataset) == N_IMAGES * (N_CHANNELS // N_FRAMES) * TILE_MULT**2

    hr, lr = dataset[0]
    assert tuple(hr.shape) == get_shape(HR_RES, batch=0, channels=N_FRAMES)
    assert tuple(lr.shape) == get_shape(LR_RES, batch=0, channels=N_FRAMES)

    # LR mode
    make_tifs(tmp_path/"lr", get_shape(LR_RES*TILE_MULT, batch=N_IMAGES))
    dataset = SlidingDataset(tmp_path/"lr", hr_res=LR_RES, lr_scale=-1, extension="tif", overlap=None, preload=False, val_split=1)
    assert str(dataset)
    assert len(dataset) == N_IMAGES * TILE_MULT**2
    assert dataset.is_lr

    lr = dataset[0]
    assert tuple(lr.shape) == get_shape(LR_RES, batch=0)

def test_pairedimagedataset(tmp_path):
    # Single frame
    make_tifs(tmp_path/"sf_hr", get_shape(HR_RES, batch=N_IMAGES))
    make_tifs(tmp_path/"sf_lr", get_shape(LR_RES, batch=N_IMAGES))
    dataset = PairedImageDataset(tmp_path/"sf_hr", tmp_path/"sf_lr")
    assert str(dataset)
    assert len(dataset) == N_IMAGES

    hr, lr = dataset[0]
    assert tuple(hr.shape) == get_shape(HR_RES, batch=0)
    assert tuple(lr.shape) == get_shape(LR_RES, batch=0)
    
    # Multi frame
    make_tifs(tmp_path/"mf_hr", get_shape(HR_RES, batch=N_IMAGES, channels=N_CHANNELS))
    make_tifs(tmp_path/"mf_lr", get_shape(LR_RES, batch=N_IMAGES, channels=N_CHANNELS))
    dataset = PairedImageDataset(tmp_path/"mf_hr", tmp_path/"mf_lr", n_frames=N_FRAMES)
    assert str(dataset)
    assert len(dataset) == N_IMAGES * (N_CHANNELS // N_FRAMES)

    hr, lr = dataset[0]
    assert tuple(hr.shape) == get_shape(HR_RES, batch=0, channels=N_FRAMES)
    assert tuple(lr.shape) == get_shape(LR_RES, batch=0, channels=N_FRAMES)

def test_pairedslidingdataset(tmp_path):
    # Single frame
    make_tifs(tmp_path/"sf_hr", get_shape(HR_RES*TILE_MULT, batch=N_IMAGES))
    make_tifs(tmp_path/"sf_lr", get_shape(LR_RES*TILE_MULT, batch=N_IMAGES))
    dataset = PairedSlidingDataset(tmp_path/"sf_hr", tmp_path/"sf_lr", extension="tif", overlap=None, preload=False)
    assert str(dataset)
    assert len(dataset) == N_IMAGES * TILE_MULT**2

    hr, lr = dataset[0]
    assert tuple(hr.shape) == get_shape(HR_RES, batch=0)
    assert tuple(lr.shape) == get_shape(LR_RES, batch=0)

    # Preload
    dataset = PairedSlidingDataset(tmp_path/"sf_hr", tmp_path/"sf_lr", extension="tif", overlap=None, preload=True)
    assert str(dataset)
    assert dataset.preload
    
    # Multi frame
    make_tifs(tmp_path/"mf_hr", get_shape(HR_RES*TILE_MULT, batch=N_IMAGES, channels=N_CHANNELS))
    make_tifs(tmp_path/"mf_lr", get_shape(LR_RES*TILE_MULT, batch=N_IMAGES, channels=N_CHANNELS))
    dataset = PairedSlidingDataset(tmp_path/"mf_hr", tmp_path/"mf_lr", n_frames=N_FRAMES, extension="tif", overlap=None, preload=False)
    assert str(dataset)
    assert len(dataset) == N_IMAGES * (N_CHANNELS // N_FRAMES) * TILE_MULT**2

    hr, lr = dataset[0]
    assert tuple(hr.shape) == get_shape(HR_RES, batch=0, channels=N_FRAMES)
    assert tuple(lr.shape) == get_shape(LR_RES, batch=0, channels=N_FRAMES)

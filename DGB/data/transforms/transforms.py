from torchvision.transforms import InterpolationMode, Resize, ToTensor, Normalize, Compose, CenterCrop

AVAILABLE_TRANSFORMS = [
    "normalize"
]

INTERPOLATION_MODES = {
    "bilinear": InterpolationMode.BILINEAR
}


def build_transform(cfg, is_train=True, transforms=None):
    """Build Transformation Functions.

    Args:
        cfg (CfgNode): config.
        is_train (bool, optional): for training (True) or test (False).
            Default is True.
        transforms (list, optional): list of strings which will overwrite
            cfg.INPUT.TRANSFORMS if given. Default is None.
    """

    if cfg.INPUT.NO_TRANSFORM:
        print("Note: No Transform is Applied.")
        return None

    if transforms is None:
        transforms = cfg.INPUT.TRANSFORMS

    for transform in transforms:
        assert transform in AVAILABLE_TRANSFORMS

    normalize = Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)

    if is_train:
        return _build_transform_train(cfg, transforms, normalize)
    else:
        return _build_transform_test(cfg, transforms, normalize)


def _build_transform_train(cfg, transforms, normalize):
    interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]

    transform_train = [Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]
    transform_train += [ToTensor()]
    if "normalize" in transforms:
        transform_train += [normalize]
    transform_train = Compose(transform_train)
    print("Training Data Transforms: {}".format(transform_train))

    return transform_train


def _build_transform_test(cfg, transforms, normalize):
    interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]

    transform_test = [Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]
    transform_test += [ToTensor()]
    if "normalize" in transforms:
        transform_test += [normalize]
    transform_test = Compose(transform_test)
    print("Testing Data Transforms: {}".format(transform_test))

    return transform_test

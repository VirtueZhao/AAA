from DGB.utils import Registry, check_availability

BACKBONE_REGISTRY = Registry("BACKBONE")


def build_backbone(cfg, **kwargs):
    available_backbone = BACKBONE_REGISTRY.registered_names()
    check_availability(cfg.MODEL.BACKBONE.NAME, available_backbone)
    print("Loading Backbone: {}".format(cfg.MODEL.BACKBONE.NAME))

    return BACKBONE_REGISTRY.get(cfg.MODEL.BACKBONE.NAME)(**kwargs)

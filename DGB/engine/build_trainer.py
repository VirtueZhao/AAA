from DGB.utils import Registry, check_availability

TRAINER_REGISTRY = Registry("TRAINER")


def build_trainer(cfg):
    available_trainers = TRAINER_REGISTRY.registered_names()
    check_availability(cfg.TRAINER.NAME, available_trainers)
    print("Loading Trainer: {}".format(cfg.TRAINER.NAME))

    return TRAINER_REGISTRY.get(cfg.TRAINER.NAME)(cfg)

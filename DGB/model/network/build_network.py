from DGB.utils import Registry, check_availability

NETWORK_REGISTRY = Registry("NETWORK")


def build_network(name, **kwargs):
    available_models = NETWORK_REGISTRY.registered_names()
    check_availability(name, available_models)
    print("Building Network: {}".format(name))
    return NETWORK_REGISTRY.get(name)(**kwargs)

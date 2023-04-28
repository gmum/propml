import logging

logger = logging.getLogger(__name__)

from ..tresnet import TResnetM, TResnetL, TResnetXL


def create_model(config):
    """Create a model
    """
    model_params = {'do_bottleneck_head': False, 'num_classes': config.n_classes}

    if config.model == 'tresnet_m':
        model = TResnetM(model_params)
    elif config.model == 'tresnet_l':
        model = TResnetL(model_params)
    elif config.model == 'tresnet_xl':
        model = TResnetXL(model_params)
    else:
        print("model: {} not found !!".format(config.model))
        raise NotImplemented

    return model

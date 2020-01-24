from torchvision import models as torchmodels
import zoo


def get_tvision(name, model_args={}, num_classes="same"):
    model = getattr(torchmodels, name)(**model_args)
    if num_classes != "same":
        raise NotImplementedError("[ERROR] \t The behavior of tvision models \
          for different classes is not implemented")
    return model


def get_from_zoo(name, args):
    return getattr(zoo, name)(**args)

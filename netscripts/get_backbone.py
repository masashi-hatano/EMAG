from models.bninception import bninception
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152


def _backbone(base, pretrained):
    if base == "resnet18":
        return resnet18(pretrained)
    elif base == "resnet34":
        return resnet34(pretrained)
    elif base == "resnet50":
        return resnet50(pretrained)
    elif base == "resnet101":
        return resnet101(pretrained)
    elif base == "resnet152":
        return resnet152(pretrained)
    elif base == "bninception":
        return bninception(pretrained=None)
    else:
        raise Exception(
            "Backbone option: resnet18, resnet34, resnet50, resnet101, resnet152, or bninception"
        )


def get_backbone(base, pretrained):
    model = _backbone(base, pretrained)
    return model

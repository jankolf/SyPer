from . import iresnet
from . import mobilefacenet

def get_model(model_id, **kwargs):

    # Resnets
    if model_id == "resnet18" or model_id == "irn18":
        return iresnet.iresnet18(**kwargs)
    
    if model_id == "resnet34" or model_id == "irn34":
        return iresnet.iresnet34(**kwargs)

    if model_id == "resnet50" or model_id == "irn50":
        return iresnet.iresnet50(**kwargs)

    # Mobilenets
    if model_id == "mobilefacenet":
        return mobilefacenet.get_mobilefacenet(**kwargs)


    raise ValueError("Unknown model_id given!")


def unfreeze_model(model_id, model):
    return None
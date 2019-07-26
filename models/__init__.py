from models.resnet import *
from models.MobileNetV3 import *


def get_model(name, n_classes, in_channels=3, norm_type='bn', use_cbam=False, dropout_rate=0.5):
    model = _get_model_instance(name)

    if 'resnet' in name:
        model = model(name=name, n_classes=n_classes, in_channels=in_channels, norm_type=norm_type, use_cbam=use_cbam, dropout_rate=dropout_rate)

    elif 'MobileNetV3' in name:
        model = model(name=name, n_classes=n_classes, in_channels=in_channels, dropout_rate=dropout_rate)

    return model

def _get_model_instance(name):
    try:
        return {
            'resnet18': resnet,
            'resnet34': resnet,
            'resnet50': resnet,
            'resnet101': resnet,
            'resnet152': resnet,
            'MobileNetV3L': MobileNetV3,
            'MobileNetV3S': MobileNetV3,
        }[name]
    except:
        print('Model {} not available'.format(name))

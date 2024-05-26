
from model import resnetv1
from model import FedLMTHyperResnet
from model import HyperTransformer
from model import transformer_fill


def generate_model(model_name, model_rate=1.0, depth_rate=4, cfg=None, LR_ratio=0.2):

    if model_name == 'resnet18':
        if cfg['algo_name'] == 'FedLMT' or cfg['algo_name'] == 'pFedLMT':
            modelobj = FedLMTHyperResnet.hybrid_resnet18(ratio_LR=LR_ratio, decom_rule=cfg['decom_rule'], cfg=cfg).to(cfg['device'])
        else:
            modelobj = resnetv1.resnet18(cfg=cfg).to(cfg['device'])
    elif model_name == 'hyper_transformer':
        modelobj = HyperTransformer.hyper_transformer(model_rate=model_rate, ratio_LR=LR_ratio, cfg=cfg).to(cfg['device'])
    elif model_name == 'transformer':
        modelobj = transformer_fill.transformer(model_rate=model_rate, cfg=cfg).to(cfg['device'])
    else:
        raise NotImplementedError("The model name is not valid")
    return modelobj
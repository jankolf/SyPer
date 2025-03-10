import copy
from collections import OrderedDict

import torch
from torch import nn

from periocular.quantization.modules import QuantAct, Quant_Linear, Quant_Conv2d, QuantActPreLu

__all__ = ['iresnet18', 'iresnet34', 'iresnet50', 'iresnet100']

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return input * x


class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1,use_se=False):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05,)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.downsample = downsample
        self.stride = stride
        self.use_se=use_se
        if (use_se):
         self.se_block=SEModule(planes,16)

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if(self.use_se):
            out=self.se_block(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNet(nn.Module):

    def __init__(self,
                 block, layers, dropout=0.5, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, use_se=False, fc_scale=14*14):
        super(IResNet, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        self.use_se=use_se
        self.fc_scale = fc_scale
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2, use_se=self.use_se)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0],use_se=self.use_se)
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1] ,use_se=self.use_se)
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2] ,use_se=self.use_se)
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05,)
        self.dropout =nn.Dropout(p=dropout, inplace=True) # 7x7x 512
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,use_se=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05, ),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation,use_se=use_se))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation,use_se=use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return x


def _iresnet(arch, block, layers, pretrained, progress, num_features=512, **kwargs):
    model = IResNet(block, layers, num_features=num_features, **kwargs)
    if pretrained:
        raise ValueError()
    return model


def iresnet18(pretrained=False, progress=True, num_features=512, **kwargs):
    return _iresnet('iresnet18', IBasicBlock, [2, 2, 2, 2], pretrained,
                    progress, num_features=num_features, **kwargs)


def iresnet34(pretrained=False, progress=True, num_features=512, **kwargs):
    return _iresnet('iresnet34', IBasicBlock, [3, 4, 6, 3], pretrained,
                    progress, num_features=num_features, **kwargs)


def iresnet50(pretrained=False, progress=True, num_features=512, **kwargs):
    return _iresnet('iresnet50', IBasicBlock, [3, 4, 14, 3], pretrained,
                    progress, num_features=num_features, **kwargs)


def iresnet100(pretrained=False, progress=True, num_features=512, **kwargs):
    return _iresnet('iresnet100', IBasicBlock, [3, 13, 30, 3], pretrained,
                    progress, num_features=num_features, **kwargs)


def quantize_model(model,weight_bit=None, act_bit=None ):
        """
		Recursively quantize a pretrained single-precision model to int8 quantized model
		model: pretrained single-precision model
		"""
        #if not (weight_bit) and not (act_bit ):
        #    weight_bit = self.settings.qw
        #    act_bit = self.settings.qa

        # quantize convolutional and linear layers
        if type(model) == nn.Conv2d:
            quant_mod = Quant_Conv2d(weight_bit=weight_bit)
            quant_mod.set_param(model)
            return quant_mod
        elif type(model) == nn.Linear :
            quant_mod = Quant_Linear(weight_bit=weight_bit)
            quant_mod.set_param(model)
            return quant_mod
        elif type(model) == nn.PReLU :
            quant_mod = QuantActPreLu(act_bit=act_bit)
            quant_mod.set_param(model)
            return quant_mod

        # quantize all the activation
        elif type(model) == nn.ReLU or type(model) == nn.ReLU6 or type(model)==nn.PReLU:
            return nn.Sequential(*[model, QuantAct(activation_bit=act_bit)])


        # recursively use the quantized module to replace the single-precision module
        elif type(model) == nn.Sequential or isinstance(model,nn.Sequential):
                mods = OrderedDict()
                for n, m in model.named_children():
                    if isinstance(m,IBasicBlock):
                        mods[n] = nn.Sequential(*[quantize_model(m,weight_bit=weight_bit, act_bit=act_bit), QuantAct(activation_bit=act_bit)])
                    else:
                        mods[n] = quantize_model(m, weight_bit=weight_bit, act_bit=act_bit)

                    #mods.append(self.quantize_model(m))
                return nn.Sequential(mods)
        #elif isinstance(model,IBasicBlock):
        #  return nn.Sequential(*[quantize_model(model,weight_bit=weight_bit, act_bit=act_bit), QuantAct(activation_bit=act_bit)])


        else:
            q_model = copy.deepcopy(model)
            for attr in dir(model):
                mod = getattr(model, attr)
                if isinstance(mod, nn.Module) and 'norm' not in attr:
                    setattr(q_model, attr, quantize_model(mod,weight_bit=weight_bit, act_bit=act_bit))
            return q_model


def freeze_model( model):
        """
		freeze the activation range
		"""
        if type(model) == QuantAct:
            model.fix()
        elif type(model) == nn.Sequential:
            for n, m in model.named_children():
                freeze_model(m)
        else:
            for attr in dir(model):
                mod = getattr(model, attr)
                if isinstance(mod, nn.Module) and 'norm' not in attr:
                    freeze_model(mod)
            return model


def unfreeze_model( model):
        """
		unfreeze the activation range
		"""
        if type(model) == QuantAct:
            model.unfix()
        elif type(model) == nn.Sequential:
            for n, m in model.named_children():
                unfreeze_model(m)
        else:
            for attr in dir(model):
                mod = getattr(model, attr)
                if isinstance(mod, nn.Module) and 'norm' not in attr:
                    unfreeze_model(mod)
            return model


if __name__ == "__main__":
    import torch
    from countFLOPS import _calc_width, count_model_flops
    from math import ceil

    pretrained = False

    models = [
        (18, iresnet18),
        (34,iresnet34),
        (50,iresnet50)
    ]

    for emb_size in [128, 256, 512]:
        for img_size in [112, 224, 256]:
            for mname, model in models:
                fc_scale = ceil(img_size/16)
                net = model(num_features=emb_size, fc_scale=fc_scale**2)
                
                
                weight_count = _calc_width(net)
                flops=count_model_flops(net, input_res=[img_size, img_size])
                print(f"m={model.__name__} @ EMB={emb_size} & IMG={img_size}x{img_size}: Weight-Count={weight_count}, Flops={flops}")
                

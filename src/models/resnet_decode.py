import sys

from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


def deconv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, output_padding: int = 0):
    """
    Creates a transposed convolution layer to reverse a conv3x3 operation.
    The parameters are adjusted to potentially increase spatial dimensions
    and match the input and output channels as needed.
    """
    # Note: The choice of padding and output_padding here may need to be adjusted
    # based on specific requirements of how the convolution was initially applied.
    # The given defaults aim to provide a starting point for reversing a basic conv3x3.
    return nn.ConvTranspose2d(
        in_planes, 
        out_planes, 
        kernel_size=3, 
        stride=stride, 
        padding=dilation,
        output_padding=output_padding,
        groups=groups, 
        bias=False, 
        dilation=dilation
        )


def deconv1x1(in_planes: int, out_planes: int, stride: int = 1, output_padding: int = 0):
    """
    Creates a transposed convolution layer to reverse a conv1x1 operation.
    This function primarily aims to adjust the number of channels, with the
    ability to modify spatial dimensions if a stride greater than 1 is used.
    """
    # For deconv1x1, padding and output_padding are typically not as critical
    # as for deconv3x3, given the kernel size of 1.
    return nn.ConvTranspose2d(
        in_planes, 
        out_planes, 
        kernel_size=1, 
        stride=stride, 
        bias=False,
        output_padding=output_padding
        )


# class BasicBlock(nn.Module):
#     """The basic block architecture of resnet-18 network.
#     """
#     expansion: int = 1

#     def __init__(
#         self,
#         inplanes: int,
#         planes: int,
#         stride: int = 1,
#         output_padding: int = 0,
#         upsample: Optional[nn.Module] = None,
#         groups: int = 1,
#         base_width: int = 64,
#         dilation: int = 1,
#         norm_layer: Optional[Callable[..., nn.Module]] = None,
#     ) -> None:
#         super().__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         if groups != 1 or base_width != 64:
#             raise ValueError("BasicBlock only supports groups=1 and base_width=64")
#         if dilation > 1:
#             raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
#         # Both self.conv1 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = deconv3x3(planes, inplanes, stride, output_padding=output_padding)
#         self.bn1 = norm_layer(inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = deconv3x3(planes, planes)
#         self.bn2 = norm_layer(planes)
#         self.upsample = upsample
#         self.stride = stride

#     def forward(self, x: Tensor) -> Tensor:
#         identity = x
#         out = self.conv2(x)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv1(out)
#         out = self.bn1(out)

#         if self.upsample is not None:
#             identity = self.upsample(x)

#         out += identity
#         out = self.relu(out)
#         return out

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self, 
        inplanes: int,
        planes: int,
        stride: int = 1,
        output_padding: int = 0,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        outdim: int = 0
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Reverse of conv3x3
        # Both self.deconv1 and self.upsample layers upsample the input when stride != 1
        self.deconv1 = deconv3x3(planes, inplanes, stride, output_padding=output_padding)
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)

        if outdim == 0:
            self.deconv2 = deconv3x3(planes, planes) # this will not change the height and width, only change channels
        else:
            self.deconv2 = deconv3x3(outdim, planes) # this will not change the height and width, only change channels
        self.bn2 = norm_layer(planes)
        
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.deconv2(x)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.deconv1(out)
        out = self.bn1(out)

        if self.upsample is not None:
            identity = self.upsample(x)
        
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        output_padding: int = 0,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        outdim: int = 0
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.deconv2 and self.upsample layers upsample the input when stride != 1

        self.deconv1 = deconv1x1(width, inplanes) # this will not change the height and width, only change channels
        self.bn1 = norm_layer(inplanes)
        self.deconv2 = deconv3x3(width, width, 
                                 stride = stride, 
                                 groups = groups, 
                                 dilation = dilation, 
                                 output_padding=output_padding)
        self.bn2 = norm_layer(width)

        if outdim == 0:
            self.deconv3 = deconv1x1(planes*self.expansion, width) # this will not change the height and width, only change channels
        else:
            self.deconv3 = deconv1x1(outdim, width) # this will not change the height and width, only change channels
        self.bn3 = norm_layer(width)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.deconv3(x)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.deconv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.deconv1(out)
        out = self.bn1(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out
    

class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        fixdim: int = False,
        SOTA: bool = False,
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.SOTA = SOTA
        if SOTA:
            self.deconv1 = nn.ConvTranspose2d(self.inplanes, 3, kernel_size=3, stride=1, padding=1, 
                                              bias=False)
        else:
            self.deconv1 = nn.ConvTranspose2d(self.inplanes, 3, kernel_size=7, stride=2, padding=3, output_padding=1, 
                                              bias=False) # o = 30 + a + 1, output_padding a = 1
        self.bn1 = norm_layer(3)
        self.relu = nn.ReLU(inplace=True)

        self.unmaxpool = nn.ConvTranspose2d(self.inplanes, self.inplanes, kernel_size=3, stride=2, padding=1, output_padding=1) # o = 14 + a +1, output_padding = 1

        self.layer1 = self._make_layer(block, 64, layers[0])  # No upsampling in the first layer
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2, output_padding = 1,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2, output_padding = 1,
                                       dilate=replace_stride_with_dilation[1])
        if not fixdim:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, output_padding = 1,
                                           dilate=replace_stride_with_dilation[2])
            if not SOTA:
                self.unavgpool = nn.ConvTranspose2d(self.inplanes, self.inplanes, kernel_size=1, stride=1, padding=0, 
                                                bias=False)
            else:
                self.unavgpool = nn.ConvTranspose2d(self.inplanes, self.inplanes, kernel_size=4, stride=1, padding=0, 
                                                bias=False)
        else:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, output_padding = 1,
                                           dilate=replace_stride_with_dilation[2],outdim=num_classes)
            if not SOTA:
                self.unavgpool = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=1, stride=1, padding=0, 
                                                bias=False)
            else:
                self.unavgpool = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=1, padding=0, 
                                                bias=False)
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, 
                    block: Type[Union[BasicBlock, Bottleneck]], 
                    planes: int, 
                    blocks: int,
                    stride: int = 1, 
                    output_padding: int = 0,
                    dilate: bool = False, 
                    outdim: int = 0,
                    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        upsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                deconv1x1(planes * block.expansion, self.inplanes, stride, output_padding),
                norm_layer(self.inplanes),
            )
    
        layers = []
        # print("layers = [], self.inplanes", self.inplanes)
        lastlayer = block(self.inplanes, planes, 
                            stride = stride, 
                            output_padding = output_padding,
                            upsample = upsample, 
                            groups = self.groups,
                            base_width = self.base_width, 
                            dilation = previous_dilation, 
                            norm_layer = norm_layer)

        self.inplanes = planes * block.expansion
        # print("self.inplanes = planes * block.expansion, self.inplanes", self.inplanes)
        upsample = None
        if outdim != 0:
            upsample = nn.Sequential(
                deconv1x1(outdim, self.inplanes),
                norm_layer(self.inplanes)
            )
        layers.append(block(self.inplanes, planes, 
                            groups=self.groups,
                            base_width=self.base_width, 
                            dilation=self.dilation,
                            norm_layer=norm_layer, 
                            upsample=upsample, 
                            outdim=outdim))
        
        for _ in range(1, blocks-1):
            layers.append(block(self.inplanes, planes,
                                groups=self.groups,
                                base_width=self.base_width, 
                                dilation=self.dilation,
                                norm_layer=norm_layer))

        layers.append(lastlayer)
        
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor):
        # print("x", x.shape)
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        # print("x = x.view(x.shape[0], x.shape[1], 1, 1)", x.shape)
        x = self.unavgpool(x)
        # print("x = self.unavgpool(x)", x.shape)
        x = self.layer4(x)
        # print("x = self.layer4(x)", x.shape)
        x = self.layer3(x)
        # print("x = self.layer3(x)", x.shape)
        x = self.layer2(x)
        # print("x = self.layer2(x)", x.shape)
        x = self.layer1(x)
        # print("x = self.layer1(x)", x.shape)

        if not self.SOTA:
            x = self.unmaxpool(x)
            # print("x = self.unmaxpool(x)", x.shape)


        x = self.deconv1(x)
        # print("x = self.deconv1(x)", x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        
        return x
    
    def forward(self, x: Tensor):
        return self._forward_impl(x)



def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        sys.exit('No pre-trained model is allowed here!')
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
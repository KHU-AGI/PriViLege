import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from .utils import set_grad, im2col_from_conv


def ensure_tensor(x):
    # Aux functions in case mask arguments are numpy arrays
    if not isinstance(x, torch.Tensor) and x is not None:
        return torch.from_numpy(x)
    return x


def same_device(x_mask, x):
    # Aux function to ensure same device fo weight and mask
    # so _mul doesn't fail
    if x.device != x_mask.device:
        return x_mask.to(x.device)
    return x_mask


def _same_shape(x_mask, x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    return x.shape == x_mask.shape


class WaRPModule(nn.Module):

    def __init__(self, layer):
        super(WaRPModule, self).__init__()

        self.weight = layer.weight
        self.bias = layer.bias
        if self.weight.ndim != 2:
            Co, Ci, k1, k2 = self.weight.shape
            self.basis_coeff = nn.Parameter(torch.Tensor(Co, Ci*k1*k2, 1, 1), requires_grad=True)
            self.register_buffer("UT_forward_conv", torch.Tensor(Ci*k1*k2, Ci, k1, k2))
            self.register_buffer("UT_backward_conv", torch.Tensor(Co, Co, 1, 1))
        else:
            self.basis_coeff = nn.Parameter(torch.Tensor(self.weight.shape), requires_grad=True)


        # use register_buffer so model.to(device) works on fixed tensors like masks
        self.register_buffer("forward_covariance", None)
        self.register_buffer("basis_coefficients", torch.Tensor(self.weight.shape).reshape(self.weight.shape[0], -1))
        self.register_buffer("coeff_mask", torch.zeros(self.basis_coeff.shape))
        self.register_buffer("UT_forward", torch.eye(self.basis_coeff.shape[1]))
        self.register_buffer("UT_backward", torch.eye(self.basis_coeff.shape[0]))

        self.flag = True


class LinearWaRP(WaRPModule):

    def __init__(self, linear_layer):
        """Shaved version of a linear layer for pruning evaluation

        Constructed from an existing layer.

        Arguments:
            linear_layer {torch.nn.Linear} -- Layer to mask. Not modified.
        """
        super(LinearWaRP, self).__init__(linear_layer)
        assert isinstance(linear_layer, nn.Linear), "Layer must be a linear layer"
        for attr in ['in_features', 'out_features']:
            setattr(self, attr, getattr(linear_layer, attr))

        self.batch_count = 0

    def pre_forward(self, input):
        with torch.no_grad():
            if self.bias is not None:
                pass
            forward_covariance = input.t() @ input
        return forward_covariance

    def post_forward(self, input):
        self.h = input.register_hook(set_grad(input))
        return input

    def post_backward(self):
        with torch.no_grad():
            if self.forward_covariance is not None:
                self.forward_covariance = self.forward_curr_cov + (self.batch_count / (self.batch_count + 1)) * \
                (self.forward_covariance - self.forward_curr_cov)
            else:
                self.forward_covariance = self.forward_curr_cov


            self.batch_count += 1


    def forward(self, input):
        if not self.flag:
            self.forward_curr_cov = self.pre_forward(input)
            input = F.linear(input, self.weight, self.bias)
        else:
            weight = self.UT_backward @ (self.basis_coeff * self.coeff_mask).clone().detach() + self.basis_coeff * (
                        1 - self.coeff_mask) @ self.UT_forward
            input = F.linear(input, weight)

        return input

    def __repr__(self):
        s = f"{self.__class__.__name__}("
        s += f'in_features={self.in_features}, '
        s += f'out_features={self.out_features}, '
        s += f'bias={self.bias is not None})'
        return s


class Conv2dWaRP(WaRPModule):

    def __init__(self, conv_layer):
        """Shaved version  of 2D convolutional layer for pruning evaluation

        Constructed from an existing layer.

        [description]

        Arguments:
            linear_layer {torch.nn.Conv2d} -- Layer to mask. Not modified.
        """
        super(Conv2dWaRP, self).__init__(conv_layer)
        assert isinstance(conv_layer, nn.Conv2d), "Layer must be a Conv2d layer"
        for attr in ['in_channels', 'out_channels', 'kernel_size', 'dilation',
                     'stride', 'padding', 'padding_mode', 'groups']:
            setattr(self, attr, getattr(conv_layer, attr))

        self.batch_count = 0

    def pre_forward(self, input):
        with torch.no_grad():
            input_col = im2col_from_conv(input.clone(), self)
            forward_covariance = input_col.t() @ input_col
        return forward_covariance


    def post_forward(self, input):
        self.h = input.register_hook(set_grad(input))
        return input


    def post_backward(self):
        with torch.no_grad():
            if self.forward_covariance is not None:
                self.forward_covariance = self.forward_cov + (self.batch_count / (self.batch_count + 1)) * \
                                          (self.forward_covariance - self.forward_cov)
            else:
                self.forward_covariance = self.forward_cov

            self.batch_count += 1


    def forward(self, input):
        if not self.flag:
            self.forward_cov = self.pre_forward(input)
            if self.padding_mode == 'circular':
                expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                    (self.padding[0] + 1) // 2, self.padding[0] // 2)
                input = F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                                self.weight, self.bias, self.stride,
                                _pair(0), self.dilation, self.groups)
            else:
                input = F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        else:
            # if self.padding_mode == 'circular':
            #     expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
            #                         (self.padding[0] + 1) // 2, self.padding[0] // 2)
            #     input = F.conv2d(F.pad(input, expanded_padding, mode='circular'),
            #                     weight, self.bias, self.stride,
            #                     _pair(0), self.dilation, self.groups)
            # else:
            UTx = F.conv2d(input, self.UT_forward_conv, None, self.stride,
                           self.padding, self.dilation, self.groups)
            AUTx = F.conv2d(UTx, (self.basis_coeff * self.coeff_mask).clone().detach() + self.basis_coeff * (
                        1 - self.coeff_mask), None, 1, 0)
            input = F.conv2d(AUTx, self.UT_backward_conv, self.bias, 1, 0)

        return input

    def __repr__(self):
        s = f"{self.__class__.__name__}("
        s += ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
              ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(**self.__dict__)

# TODO Conv1D Conv3D ConvTranspose
# squeeze out Convs for channel pruning


warped_modules = {
    nn.Linear: LinearWaRP,
    nn.Conv2d: Conv2dWaRP,
}

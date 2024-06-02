import torch
import torch.nn as nn
import torch.nn.functional as F



def set_grad(var):
    def hook(grad):
        var.grad = grad
    return hook


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = F.pad(input_data, [pad, pad, pad, pad], 'constant', 0)
    col = torch.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = torch.permute(col, (0, 4, 5, 1, 2, 3)).reshape(N * out_h * out_w, -1)
    return col

def im2col_from_conv(input_data, conv):
    return im2col(input_data, conv.kernel_size[0], conv.kernel_size[1], conv.stride[0], conv.padding[0])


def get_params(model, recurse=False):
    """Returns dictionary of paramters

    Arguments:
        model {torch.nn.Module} -- Network to extract the parameters from

    Keyword Arguments:
        recurse {bool} -- Whether to recurse through children modules

    Returns:
        Dict(str:numpy.ndarray) -- Dictionary of named parameters their
                                   associated parameter arrays
    """
    params = {k: v.detach().cpu().numpy().copy()
              for k, v in model.named_parameters(recurse=recurse)}
    return params


def get_buffers(model, recurse=False):
    """Returns dictionary of buffers

    Arguments:
        model {torch.nn.Module} -- Network to extract the buffers from

    Keyword Arguments:
        recurse {bool} -- Whether to recurse through children modules

    Returns:
        Dict(str:numpy.ndarray) -- Dictionary of named parameters their
                                   associated parameter arrays
    """
    buffers = {k: v.detach().cpu().numpy().copy()
              for k, v in model.named_buffers(recurse=recurse)}
    return buffers
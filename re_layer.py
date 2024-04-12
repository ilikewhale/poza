import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def time_to_batch(x, block_size):
    """Splits time dimension (i.e. dimension 1) of `x` into batches.

    Within each batch element, the `k*block_size` time steps are transposed,
    so that the `k` time steps in each output batch element are offset by
    `block_size` from each other.

    The number of input time steps must be a multiple of `block_size`.

    Args:
        x: Tensor of shape [nb, k*block_size, n] for some natural number k.
        block_size: number of time steps (i.e. size of dimension 1) in the output
            tensor.

    Returns:
        Tensor of shape [nb*block_size, k, n]
    """
    shape = x.size()
    y = x.view(shape[0], shape[1] // block_size, block_size, shape[2])
    y = y.permute(0, 2, 1, 3)
    y = y.view(shape[0] * block_size, shape[1] // block_size, shape[2])
    return y

def batch_to_time(inputs, rate, crop_left=0):
    ''' Reshape to 1d signal, and remove excess zero-padding.
    
    Used to perform 1D dilated convolution.
    
    Args:
      inputs: (tensor)
      crop_left: (int)
      rate: (int)
    Ouputs:
      outputs: (tensor)
    '''
    shape = inputs.size()
    batch_size = shape[0] // rate
    width = shape[1]
    
    out_width = width * rate
    num_channels = shape[2]
    
    perm = (1, 0, 2)
    new_shape = (out_width, -1, num_channels) 
    transposed = inputs.permute(1, 0, 2)
    reshaped = transposed.reshape(new_shape)
    outputs = reshaped.permute(1, 0, 2)
    cropped = outputs[:, crop_left:, :]
    return cropped


def conv1d(inputs,
           out_channels,
           filter_width=2,
           stride=1,
           padding=0,
           bias=False,
           gain=np.sqrt(2),
           activation=nn.ReLU()):
    '''One dimension convolution helper function.
    
    Sets variables with good defaults.
    
    Args:
      inputs:
      out_channels:
      filter_width:
      stride:
      paddding:
      gain:
      activation:
      bias:
      
    Outputs:
      outputs:
    '''
    in_channels = inputs.size(1)

    stddev = gain / np.sqrt(filter_width**2 * in_channels)
    w_init = torch.Tensor(in_channels, out_channels, filter_width).normal_(std=stddev)

    conv1d_layer = nn.Conv1d(in_channels, out_channels, filter_width, stride=stride, padding=padding, bias=bias)
    conv1d_layer.weight.data = w_init

    outputs = conv1d_layer(inputs)

    if activation:
        outputs = activation(outputs)

    return outputs

def dilated_conv1d(inputs,
                   out_channels,
                   filter_width=2,
                   rate=1,
                   padding='VALID',
                   name=None,
                   gain=np.sqrt(2),
                   activation=F.relu):
    '''
    
    Args:
      inputs: (tensor)
      output_channels:
      filter_width:
      rate:
      padding:
      name:
      gain:
      activation:

    Outputs:
      outputs: (tensor)
    '''
    assert name
    _, width, _ = inputs.size()
    inputs_ = time_to_batch(inputs, rate=rate)
    outputs_ = conv1d(inputs_,
                      out_channels=out_channels,
                      filter_width=filter_width,
                      padding=padding,
                      gain=gain,
                      activation=activation)
    _, conv_out_width, _ = outputs_.size()
    new_width = conv_out_width * rate
    diff = new_width - width
    outputs = batch_to_time(outputs_, rate=rate, crop_left=diff)

    # Add additional shape information.
    outputs_shape = list(outputs.size())
    outputs_shape[1] = width
    outputs_shape[2] = out_channels
    outputs.set_shape(outputs_shape)

    return outputs

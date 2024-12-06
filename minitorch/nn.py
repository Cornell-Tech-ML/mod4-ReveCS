from typing import Tuple, Optional

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    
    new_height = height // kh
    new_width = width // kw

    reshaped = input.contiguous().view(
        batch,
        channel,
        new_height,
        kh,
        new_width,
        kw
    )

    output = reshaped.permute(0, 1, 2, 4, 3, 5).contiguous()
    output = output.view(batch, channel, new_height, new_width, kh * kw)
    
    return output, new_height, new_width

def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Applies 2D average pooling over an input signal composed of several input planes.

    Args:
        input: Input tensor of shape (batch x channel x height x width)
        kernel: Tuple of (kernel_height, kernel_width) specifying size of pooling region

    Returns:
        Tensor of shape (batch x channel x new_height x new_width) where new_height and
        new_width are determined by the kernel size
    """
    output, new_height, new_width = tile(input, kernel)
    return output.mean(4).contiguous().view(output.shape[0], output.shape[1], new_height, new_width)

fastmax = FastOps.reduce(operators.max, -float("inf"))

def argmax(input: Tensor, dim: int) -> Tensor:
    """Returns a tensor with 1s in positions where the input tensor has its maximum value along the specified dimension.

    Args:
        input: Input tensor
        dim: Dimension along which to find argmax

    Returns:
        A tensor of the same shape as input with 1s at the positions of maximal values along dim
    """
    max = fastmax(input, dim)
    return max == input

class Max(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for max operation.

        Args:
            ctx: Context for saving values needed in backward pass
            a: Input tensor to find max values over
            dim: Dimension along which to find max values

        Returns:
            Tensor containing max values along specified dimension
        """
        ctx.save_for_backward(a, int(dim.item()))
        return fastmax(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for max operation.

        Args:
            ctx: Context containing saved tensors from forward pass
            grad_output: Gradient of the loss with respect to the output

        Returns:
            Tuple of:
                - Gradient of the loss with respect to the input tensor
                - Gradient of the loss with respect to the dimension (always 0.0)
        """
        input, dim = ctx.saved_values
        return grad_output * argmax(input, dim), 0.0


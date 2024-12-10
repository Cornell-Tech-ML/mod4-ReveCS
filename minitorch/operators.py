"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


def mul(x: float, y: float) -> float:
    """Multiplies two floating-point numbers.

    Args:
    ----
        x: The first number.
        y: The second number.

    Returns:
    -------
        The product of x and y.

    """
    return x * y


def id(x: float) -> float:
    """Returns the input value.

    Args:
    ----
        x: The input value.

    Returns:
    -------
        The input value.

    """
    return x


def add(x: float, y: float) -> float:
    """Adds two floating-point numbers.

    Args:
    ----
        x: The first number.
        y: The second number.

    Returns:
    -------
        The sum of x and y.

    """
    return x + y


def neg(x: float) -> float:
    """Negates a floating-point number.

    Args:
    ----
        x: The number to negate.

    Returns:
    -------
        The negated number.

    """
    return -1.0 * x


def lt(x: float, y: float) -> float:
    """Checks if one floating-point number is less than another.

    Args:
    ----
        x: The first number.
        y: The second number.

    Returns:
    -------
        True if x is less than y, False otherwise.

    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Checks if two floating-point numbers are equal.

    Args:
    ----
        x: The first number.
        y: The second number.

    Returns:
    -------
        True if x is equal to y, False otherwise.

    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Returns the larger of two floating-point numbers.

    Args:
    ----
        x: The first number.
        y: The second number.

    Returns:
    -------
        The maximum of x and y.

    """
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Checks if two floating-point numbers are close in value.

    Args:
    ----
        x: The first number.
        y: The second number.

    Returns:
    -------
        True if x and y are close, False otherwise.

    """
    return 1.0 if (x - y < 1e-2) and (y - x < 1e-2) else 0.0


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function of a floating-point number.

    Args:
    ----
        x: The input number.

    Returns:
    -------
        The result of the sigmoid function applied to x.

    """
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Applies the ReLU activation function to a floating-point number.

    Args:
    ----
        x: The input number.

    Returns:
    -------
        The result of the ReLU function applied to x.

    """
    return x if x >= 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Calculates the natural logarithm of a floating-point number.

    Args:
    ----
        x: The input number.

    Returns:
    -------
        The result of the natural logarithm applied to x.

    """
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Calculates the exponential function of a floating-point number.

    Args:
    ----
        x: The input number.

    Returns:
    -------
        The result of the exponential function applied to x.

    """
    return math.exp(x)


def inv(x: float) -> float:
    """Calculates the reciprocal of a floating-point number.

    Args:
    ----
        x: The input number.

    Returns:
    -------
        The result of the reciprocal function applied to x.

    """
    return 1.0 / x


def log_back(x: float, y: float) -> float:
    """Computes the derivative of log times a second arg.

    Args:
    ----
        x: The input value.
        y: The second argument to multiply with the derivative.

    Returns:
    -------
        The result of the derivative of log(x) multiplied by y.

    """
    return y / (x + EPS)


def inv_back(x: float, y: float) -> float:
    """Computes the derivative of the reciprocal function times a second arg.

    Args:
    ----
        x: The input value.
        y: The second argument to multiply with the derivative.

    Returns:
    -------
        The result of the derivative of the reciprocal function applied to x.

    """
    return (-1.0 / x**2) * y


def relu_back(x: float, y: float) -> float:
    """Computes the derivative of the ReLU function times a second arg.

    Args:
    ----
        x: The input value.
        y: The second argument to multiply with the derivative.

    Returns:
    -------
        The result of the derivative of the ReLU function applied to x.

    """
    return y if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.


def map(f: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Applies a given function to each element of an iterable and returns a new iterable.

    Args:
    ----
        f: The function to apply to the iterable.
        iter: The iterable to map over.

    Returns:
    -------
        A new iterable with the results of applying f to each element of iter.

    """

    def _map(iter: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in iter:
            ret.append(f(x))
        return ret

    return _map


def zipWith(
    f: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Combines elements from two iterables by applying a given function to corresponding elements and returns a new iterable.

    Args:
    ----
        f: The function to apply to both iterables.
        iter1: The first iterable.
        iter2: The second iterable.

    Returns:
    -------
        A new iterable with the results of applying f to corresponding elements of iter1 and iter2.

    """

    def _zipWith(iter1: Iterable[float], iter2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(iter1, iter2):
            ret.append(f(x, y))
        return ret

    return _zipWith


def reduce(
    f: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Reduces an iterable to a single value by applying a given function cumulatively and returns the value.

    Args:
    ----
        f: The function to apply cumulatively.
        start: The starting iterable.

    Returns:
    -------
        The result of applying f cumulatively to the elements of iter.

    """

    def _reduce(iter: Iterable[float]) -> float:
        val = start
        for l in iter:
            val = f(val, l)
        return val

    return _reduce


def negList(iter: Iterable[float]) -> Iterable[float]:
    """Negates each element in the input iterable.

    Args:
    ----
        iter: The input iterable.

    Returns:
    -------
        A new iterable with each element negated.

    """
    return map(neg)(iter)


def addLists(iter1: Iterable[float], iter2: Iterable[float]) -> Iterable[float]:
    """Adds corresponding elements from two iterables.

    Args:
    ----
        iter1: The first iterable.
        iter2: The second iterable.

    Returns:
    -------
        A new iterable with the results of adding corresponding elements of iter1 and iter2.

    """
    return zipWith(add)(iter1, iter2)


def sum(iter: Iterable[float]) -> float:
    """Sums all elements in the input iterable.

    Args:
    ----
        iter: The input iterable.

    Returns:
    -------
        The sum of all elements in the iterable.

    """
    return reduce(add, 0.0)(iter)


def prod(iter: Iterable[float]) -> float:
    """Multiplies all elements in the input iterable.

    Args:
    ----
        iter: The input iterable.

    Returns:
    -------
        The product of all elements in the iterable.

    """
    return reduce(mul, 1.0)(iter)

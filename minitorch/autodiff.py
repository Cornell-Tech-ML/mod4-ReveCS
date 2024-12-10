from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    """Computes an approximation to the derivative of f with respect to one arg.

    See :doc:derivative or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f: The function to differentiate.
        *vals: The point at which to compute the derivative.
        arg: The index of the argument to compute the derivative with respect to. Defaults to 0.
        epsilon: The small value to use for the finite difference calculation. Defaults to 1e-6.

    Returns:
    -------
        Any: An approximation of the derivative of f with respect to the specified argument.

    """
    # TODO: Implement for Task 1.1.
    # Create a list of arguments
    args = list(vals)

    # Compute f(x + epsilon)
    args[arg] = vals[arg] + epsilon
    f_plus = f(*args)

    # Compute f(x - epsilon)
    args[arg] = vals[arg] - epsilon
    f_minus = f(*args)

    # Compute the slope
    slope = (f_plus - f_minus) / (2 * epsilon)
    return slope


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate the derivative of the variable with respect to the input.

        Args:
        ----
            x: The derivative to accumulate.

        """
        ...

    @property
    def unique_id(self) -> int:
        """Unique identifier for the variable.

        Returns
        -------
            int: The unique identifier for the variable.

        """
        ...

    def is_leaf(self) -> bool:
        """Check if the variable is a leaf.

        Returns
        -------
            bool: True if the variable is a leaf, False otherwise.

        """
        ...

    def is_constant(self) -> bool:
        """Check if the variable is a constant.

        Returns
        -------
            bool: True if the variable is a constant, False otherwise.

        """
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Get the parents of the variable.

        Returns
        -------
            Iterable[Variable]: The parents of the variable.

        """
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Apply the chain rule to the variable.

        Args:
        ----
            d_output: The derivative of the output with respect to the variable.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: The derivatives of the variable with respect to the input.

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # TODO: Implement for Task 1.4.
    topo_order = []
    visited = set()

    def visit(v: Variable) -> None:
        if v.unique_id in visited or v.is_constant():
            return

        visited.add(v.unique_id)
        if not v.is_leaf():
            for parent in v.parents:
                if not parent.is_constant():
                    visit(parent)

        topo_order.append(v)

    visit(variable)
    return topo_order[::-1]
    # raise NotImplementedError("Need to implement for Task 1.4")


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph to compute derivatives for the leaf nodes.

    Args:
    ----
        variable: The right-most variable.
        deriv: The derivative we want to propagate backward to the leaves.

    Returns:
    -------
        None: Updates the derivative values of each leaf through `accumulate_derivative`.

    """
    # TODO: Implement for Task 1.4.
    # Perform a topological sort on the computation graph to visit nodes in reverse order
    topo_order = topological_sort(variable)
    derivatives = {variable.unique_id: deriv}

    for v in topo_order:
        deriv = derivatives[v.unique_id]
        if v.is_leaf():
            v.accumulate_derivative(deriv)
        else:
            for parent, parent_deriv in v.chain_rule(deriv):
                if parent.is_constant():
                    continue
                derivatives.setdefault(parent.unique_id, 0.0)
                derivatives[parent.unique_id] = (
                    derivatives[parent.unique_id] + parent_deriv
                )

    # derivatives.clear()
    # raise NotImplementedError("Need to implement for Task 1.4")


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation.

        Args:
        ----
            *values: The values to store.

        """
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Get the saved values.

        Returns
        -------
            Tuple[Any, ...]: The saved values.

        """
        return self.saved_values

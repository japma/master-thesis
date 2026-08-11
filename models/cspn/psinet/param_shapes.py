from __future__ import annotations

from models.cspn.psinet.factorized_leaf_layer import FactorizedLeafLayer
from models.cspn.psinet.sum_layer import EinsumLayer, EinsumMixingLayer


def derive_layer_param_shapes(einet_layers: list) -> list[tuple[int, ...]]:
    if not isinstance(einet_layers[0], FactorizedLeafLayer):
        raise AssertionError("einet_layers[0] must be the FactorizedLeafLayer.")

    shapes: list[tuple[int, ...]] = [tuple(einet_layers[0].ef_array.params_shape)]
    for layer in einet_layers[1:]:
        if isinstance(layer, (EinsumLayer, EinsumMixingLayer)):
            shapes.append(tuple(layer.params_shape))
        else:
            raise AssertionError(
                f"Unexpected layer type in einet_layers[1:]: {type(layer)}"
            )
    return shapes

import torch
from torch._tensor import Tensor

from models.cspn.psinet.exponential_family_array import (
    ExponentialFamilyArray,
    NormalArray,
)
from models.cspn.psinet.factorized_leaf_layer import FactorizedLeafLayer
from models.cspn.psinet.graph import (
    check_graph,
    get_leaves,
    get_roots,
    get_sums,
    topological_layers,
)
from models.cspn.psinet.sum_layer import EinsumLayer, EinsumMixingLayer


class Args:
    """
    Arguments for EinsumNetwork class.

    num_var: number of random variables (RVs). An RV might be multidimensional though -- see num_dims.
    num_dims: number of dimensions per RV. E.g. you can model an 32x32 RGB image as an 32x32 array of three dimensional
              RVs.
    num_input_distributions: number of distributions per input region (K in the paper).
    num_sums: number of sum nodes per internal region (K in the paper).
    num_classes: number of outputs of the PC.
    exponential_family: which exponential family to use; (sub-class ExponentialFamilyTensor).
    exponential_family_args: arguments for the exponential family, e.g. trial-number N for Binomial.
    """

    def __init__(
        self,
        num_var: int = 20,
        num_dims: int = 1,
        num_input_distributions: int = 10,
        num_sums: int = 10,
        num_classes: int = 1,
        exponential_family: type[ExponentialFamilyArray] = NormalArray,
        exponential_family_args=None,
    ) -> None:
        self.num_var = num_var
        self.num_dims = num_dims
        self.num_input_distributions = num_input_distributions
        self.num_sums = num_sums
        self.num_classes = num_classes
        self.exponential_family = exponential_family
        if exponential_family_args is None:
            exponential_family_args = {}
        self.exponential_family_args = exponential_family_args


class EinsumNetwork(torch.nn.Module):
    """
    Implements Einsum Networks (EiNets).

    The basic philosophy of EiNets is to summarize many PC nodes in monolithic GPU-friendly parallel operations.
    An EiNet can be seen as a special layered feed-forward neural network, consisting of a sequence of layers. Each
    layer can in principle get input from all layers before.

    As a general design principle, each layer in EinsumNetworks produces a tensor of log-densities in the forward pass,
    of generic shape
            (batch_size, vector_length, num_nodes)
    where
        batch_size is the number of samples in a mini-batch.
        vector_length is the length of the vectorized operations; this is called K in the paper -- in the paper we
                      assumed this constant over the whole EiNet, but this can be partially relaxed.
        num_nodes is the number of nodes which are realized in parallel using this layer.
    Thus, in classical PCs, we would interpret the each layer as a collection of vector_length * num_nodes PC nodes.

    The class EinsumNetork mainly governs the layer-wise layout, initialization, forward() calls, etc.

    Every einet_layer's parameters come from one of two sources, selected by whether param_nn is given:
      - param_nn is a module (e.g. a CSPN-style hypernetwork conditioned on labels `y`): params = param_nn(y, x).
      - param_nn is None: each layer uses its own directly-trained parameters (populated by initialize()),
        batch-expanded to match the input. This is the plain, unconditional PC case -- see LabelPC.
    Either way, forward()/backtrack() always work with a list of per-layer parameter tensors of shape
    (batch, *layer.params_shape).
    """

    def __init__(self, graph, param_nn=None, args=None) -> None:
        """Make an EinsumNetwork."""
        super().__init__()

        self.param_nn = param_nn
        check_flag, check_msg = check_graph(graph)
        if not check_flag:
            raise AssertionError(check_msg)
        self.graph = graph

        self.args = args if args is not None else Args()

        if len(get_roots(self.graph)) != 1:
            raise AssertionError(
                "Currently only EinNets with single root node supported."
            )

        root = get_roots(self.graph)[0]
        if tuple(range(self.args.num_var)) != root.scope:
            raise AssertionError("The graph should be over tuple(range(num_var)).")

        for node in get_leaves(self.graph):
            node.num_dist = self.args.num_input_distributions

        for node in get_sums(self.graph):
            if node is root:
                node.num_dist = self.args.num_classes
            else:
                node.num_dist = self.args.num_sums

        # Algorithm 1 in the paper -- organize the PC in layers
        self.graph_layers = topological_layers(self.graph)

        # input layer
        einet_layers = [
            FactorizedLeafLayer(
                self.graph_layers[0],
                self.args.num_var,
                self.args.num_dims,
                self.args.exponential_family,
                self.args.exponential_family_args,
            )
        ]

        # internal layers
        for c, layer in enumerate(self.graph_layers[1:]):
            if c % 2 == 0:  # product layer
                einet_layers.append(
                    EinsumLayer(self.graph, layer, einet_layers)
                )
            else:  # sum layer
                # the Mixing layer is only for regions which have multiple partitions as children.
                multi_sums = [n for n in layer if len(graph.succ[n]) > 1]
                if multi_sums:
                    einet_layers.append(
                        EinsumMixingLayer(graph, multi_sums, einet_layers[-1])
                    )

        self.einet_layers = torch.nn.ModuleList(einet_layers)

    def initialize(self, init_dict=None) -> None:
        """
        Initialize layers.

        :param init_dict: None; or
                          dictionary int->initializer; mapping layer index to initializers; or
                          dictionary layer->initializer;
                          the init_dict does not need to have an initializer for all layers
        :return: None
        """
        if init_dict is None:
            init_dict = {}
        if all(isinstance(k, int) for k in init_dict):
            init_dict = {self.einet_layers[k]: init_dict[k] for k in init_dict}
        for layer in self.einet_layers:
            layer.initialize(init_dict.get(layer, "default"))

    def set_marginalization_idx(self, idx) -> None:
        """Set indices of marginalized variables."""
        self.einet_layers[0].set_marginalization_idx(idx)

    def get_marginalization_idx(self):
        """Get indices of marginalized variables."""
        return self.einet_layers[0].get_marginalization_idx()

    def _own_layer_params(self, layer) -> Tensor:
        """The parameter tensor a layer would use when there is no param_nn -- unbatched,
        shape layer.params_shape (or layer.ef_array.params_shape for the leaf layer)."""
        if isinstance(layer, FactorizedLeafLayer):
            return layer.ef_array.params
        return layer.params

    def _get_params(
        self, x: Tensor | None, y: Tensor | None, batch_size: int | None = None
    ) -> list[Tensor]:
        """Resolve the per-layer parameter list, batched to shape (batch, *layer.params_shape)."""
        if self.param_nn is not None:
            return self.param_nn(y, x)
        if batch_size is None:
            if x is not None:
                batch_size = x.shape[0]
            elif y is not None:
                batch_size = y.shape[0]
            else:
                raise ValueError(
                    "Need x or y to determine batch size when param_nn is None."
                )
        params = []
        for layer in self.einet_layers:
            p = self._own_layer_params(layer)
            params.append(p.unsqueeze(0).expand(batch_size, *p.shape))
        return params

    def forward(self, x: Tensor, y: Tensor | None = None):
        """Evaluate the EinsumNetwork feed forward.
        x: x for p(x|y) (target variable), shape=[B, N]
        y: evidence for p(x|y) (conditional variable) shape=[B, M]. Only required when this net has a
           param_nn (a conditioning hypernetwork); unconditional PCs (param_nn is None) don't need it.
        """
        params = self._get_params(x=x, y=y)

        input_layer = self.einet_layers[0]
        input_layer(x, params[0])
        for i, einsum_layer in enumerate(self.einet_layers[1:]):
            j = i + 1  # increment i by 1 as enumerate starts with 0, we need +1
            einsum_layer(params[j])
        return self.einet_layers[-1].prob[:, :, 0]

    def backtrack(
        self, y: Tensor | None = None, num_samples: int=1, class_idx: int=0, x=None, mode="sampling", **kwargs
    ) -> Tensor | None:
        """
        Perform backtracking; for sampling or MPE approximation.
        """
        if x is not None:
            batch_size = x.shape[0]
        elif y is not None:
            batch_size = y.shape[0]
        else:
            raise ValueError("backtrack() needs x or y to determine the batch size.")

        params = self._get_params(x=None, y=y, batch_size=batch_size)

        if len(params) != len(self.einet_layers):
            raise AssertionError(
                f"param source produced {len(params)} param tensors but there are "
                f"{len(self.einet_layers)} einet_layers. These must match 1:1."
            )
        layer_to_params = dict(zip(self.einet_layers, params, strict=False))
        if x is not None:
            if y is not None and x.shape[0] != y.shape[0]:
                raise AssertionError(
                    f"x and y must share the same batch size, got x.shape[0]={x.shape[0]} "
                    f"and y.shape[0]={y.shape[0]}."
                )
            self.forward(x, y)
        else:
            assert y is not None  # guaranteed by the batch_size resolution above
            dummy_x = torch.zeros(
                batch_size, self.args.num_var, device=y.device, dtype=torch.float32
            )
            self.forward(dummy_x, y)

        num_samples = batch_size

        sample_idx = {l: [] for l in self.einet_layers}
        dist_idx = {l: [] for l in self.einet_layers}
        reg_idx = {l: [] for l in self.einet_layers}

        root = self.einet_layers[-1]
        sample_idx[root] = list(range(num_samples))
        dist_idx[root] = [class_idx] * num_samples
        reg_idx[root] = [0] * num_samples

        for layer in reversed(self.einet_layers):
            if not sample_idx[layer]:
                continue

            layer_params = layer_to_params[layer]

            if type(layer) is EinsumLayer:
                ret = layer.backtrack(
                    layer_params,
                    dist_idx[layer],
                    reg_idx[layer],
                    sample_idx[layer],
                    use_evidence=(x is not None),
                    mode=mode,
                    **kwargs,
                )
                (
                    dist_idx_left,
                    dist_idx_right,
                    reg_idx_left,
                    reg_idx_right,
                    layers_left,
                    layers_right,
                ) = ret

                for c, layer_left in enumerate(layers_left):
                    sample_idx[layer_left].append(sample_idx[layer][c])
                    dist_idx[layer_left].append(dist_idx_left[c])
                    reg_idx[layer_left].append(reg_idx_left[c])

                for c, layer_right in enumerate(layers_right):
                    sample_idx[layer_right].append(sample_idx[layer][c])
                    dist_idx[layer_right].append(dist_idx_right[c])
                    reg_idx[layer_right].append(reg_idx_right[c])

            elif type(layer) is EinsumMixingLayer:
                ret = layer.backtrack(
                    layer_params,
                    dist_idx[layer],
                    reg_idx[layer],
                    sample_idx[layer],
                    use_evidence=(x is not None),
                    mode=mode,
                    **kwargs,
                )
                dist_idx_out, reg_idx_out, layers_out = ret

                for c, layer_out in enumerate(layers_out):
                    sample_idx[layer_out].append(sample_idx[layer][c])
                    dist_idx[layer_out].append(dist_idx_out[c])
                    reg_idx[layer_out].append(reg_idx_out[c])

            elif type(layer) is FactorizedLeafLayer:
                unique_sample_idx = sorted(set(sample_idx[layer]))
                if unique_sample_idx != sample_idx[root]:
                    raise AssertionError("This should not happen.")

                dist_idx_sample = []
                reg_idx_sample = []
                for sidx in unique_sample_idx:
                    dist_idx_sample.append(
                        [
                            dist_idx[layer][c]
                            for c, i in enumerate(sample_idx[layer])
                            if i == sidx
                        ]
                    )
                    reg_idx_sample.append(
                        [
                            reg_idx[layer][c]
                            for c, i in enumerate(sample_idx[layer])
                            if i == sidx
                        ]
                    )

                samples = layer.backtrack(
                    layer_params, dist_idx_sample, reg_idx_sample, mode=mode, **kwargs
                )

                if self.args.num_dims == 1:
                    samples = torch.squeeze(samples, 2)

                if x is not None:
                    marg_idx = set(layer.get_marginalization_idx() or [])
                    keep_idx = [
                        i for i in range(self.args.num_var) if i not in marg_idx
                    ]
                    samples[:, keep_idx] = x[:, keep_idx]

                return samples
        return None

    def sample(self, y: Tensor | None = None, num_samples: int=1, class_idx: int=0, x=None, **kwargs) -> Tensor | None:
        return self.backtrack(
            y,
            num_samples=num_samples,
            class_idx=class_idx,
            x=x,
            mode="sample",
            **kwargs,
        )

    def mpe(self, y: Tensor | None = None, num_samples: int=1, class_idx: int=0, x=None, **kwargs) -> Tensor | None:
        return self.backtrack(
            y,
            num_samples=num_samples,
            class_idx=class_idx,
            x=x,
            mode="argmax",
            **kwargs,
        )

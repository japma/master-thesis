import torch
from torch._tensor import Tensor

from models.cspn.psinet.layer import Layer


class FactorizedLeafLayer(Layer):
    """
    Computes all EiNet leaves in parallel, where each leaf is a vector of factorized distributions, where factors are
    from exponential families.

    In FactorizedLeafLayer, we generate an ExponentialFamilyArray with array_shape = (num_dist, num_replica), where
        num_dist is the vector length of the vectorized distributions (K in the paper), and
        num_replica is picked large enough such that "we compute enough leaf densities". At the moment we rely that
            the PC structure (see Class Graph) provides the necessary information to determine num_replica. In
            particular, we require that each leaf of the graph has the field einet_address.replica_idx defined;
            num_replica is simply the max over all einet_address.replica_idx.
            In the future, it would convenient to have an automatic allocation of leaves to replica, without requiring
            the user to specify this.
    The generate ExponentialFamilyArray has shape (batch_size, num_var, num_dist, num_replica). This array of densities
    will contain all densities over single RVs, which are then multiplied (actually summed, due to log-domain
    computation) together in forward(...).
    """

    def __init__(self, leaves, num_var, num_dims, exponential_family, ef_args) -> None:
        """
        :param leaves: list of PC leaves (DistributionVector, see Graph.py)
        :param num_var: number of random variables (int)
        :param num_dims: dimensionality of RVs (int)
        :param exponential_family: type of exponential family (derived from ExponentialFamilyArray)
        :param ef_args: arguments of exponential_family
        """
        super().__init__()

        self.nodes = leaves
        self.num_var = num_var
        self.num_dims = num_dims

        num_dist = list({n.num_dist for n in self.nodes})
        if len(num_dist) != 1:
            raise AssertionError(
                "All leaves must have the same number of distributions."
            )
        num_dist = num_dist[0]

        replica_indices = {n.einet_address.replica_idx for n in self.nodes}
        if sorted(replica_indices) != list(range(len(replica_indices))):
            raise AssertionError(
                "Replica indices should be consecutive, starting with 0."
            )
        num_replica = len(replica_indices)

        # this computes an array of (batch, num_var, num_dist, num_repetition) exponential family densities
        # see ExponentialFamilyArray
        self.ef_array = exponential_family(
            num_var, num_dims, (num_dist, num_replica), **ef_args
        )

        # self.scope_tensor indicates which densities in self.ef_array belongs to which leaf.
        # TODO: it might be smart to have a sparse implementation -- I have experimented a bit with this, but it is not
        # always faster.
        self.register_buffer(
            "scope_tensor", torch.zeros((num_var, num_replica, len(self.nodes)))
        )
        for c, node in enumerate(self.nodes):
            self.scope_tensor[node.scope, node.einet_address.replica_idx, c] = 1.0
            node.einet_address.layer = self
            node.einet_address.idx = c

    # --------------------------------------------------------------------------------
    # Implementation of Layer interface

    def initialize(self, initializer=None) -> None:
        self.ef_array.initialize(initializer)

    def forward(self, x=None, params=None) -> None:
        """
        Compute the factorized leaf densities. We are doing the computation in the log-domain, so this is actually
        computing sums over densities.

        We first pass the data x into self.ef_array, which computes a tensor of shape
        (batch_size, num_var, num_dist, num_replica). This is best interpreted as vectors of length num_dist, for each
        sample in the batch and each RV. Since some leaves have overlapping scope, we need to compute "enough" leaves,
        hence the num_replica dimension. The assignment of these log-densities to leaves is represented with
        self.scope_tensor.
        In the end, the factorization (sum in log-domain) is realized with a single einsum.

        :param x: input data (Tensor).
                  If self.num_dims == 1, this can be either of shape (batch_size, self.num_var, 1) or
                  (batch_size, self.num_var).
                  If self.num_dims > 1, this must be of shape (batch_size, self.num_var, self.num_dims).
        :return: log-density vectors of leaves
                 Will be of shape (batch_size, num_dist, len(self.nodes))
                 Note: num_dist is K in the paper, len(self.nodes) is the number of PC leaves
        """
        self.prob = torch.einsum(
            "bxir,xro->bio", self.ef_array(x, params), self.scope_tensor
        )

    def backtrack(self, params, dist_idx, node_idx, mode: str = "sample", **kwargs) -> Tensor:
        """
        Backtrackng mechanism for EiNets.

        :param params: batched leaf parameters for this layer, as produced during the forward pass this
                       backtracking call corresponds to (shape (batch, *self.ef_array.params_shape)).
        :param dist_idx: list of N indices into the distribution vectors, which shall be sampled.
        :param node_idx: list of N indices into the leaves, which shall be sampled.
        :param mode: 'sample' or 'argmax'; for sampling or MPE approximation, respectively.
        :param kwargs: keyword arguments
        :return: samples (Tensor). Of shape (N, self.num_var, self.num_dims).
        """
        if len(dist_idx) != len(node_idx):
            raise AssertionError("Invalid input.")

        with torch.no_grad():
            phi = self.ef_array.reparam(params)

            N = len(dist_idx)

            values = torch.zeros(
                (N, self.num_var, self.num_dims),
                device=phi.device,
                dtype=phi.dtype,
            )

            for n in range(N):
                if len(dist_idx[n]) != len(node_idx[n]):
                    raise AssertionError("Invalid input.")

                cur_value = torch.zeros(
                    self.num_var,
                    self.num_dims,
                    device=phi.device,
                    dtype=phi.dtype,
                )

                phi_n = phi[n]  # (num_var, num_dist, num_replica, num_stats)

                for c, k in enumerate(node_idx[n]):
                    scope = list(self.nodes[k].scope)
                    rep = self.nodes[k].einet_address.replica_idx
                    d = dist_idx[n][c]

                    phi_selected = phi_n[scope, d, rep, :]  # (len(scope), num_stats)

                    phi_selected = phi_selected.unsqueeze(1).unsqueeze(1)

                    if mode == "sample":
                        sample = self.ef_array._sample(1, phi_selected, **kwargs)
                        cur_value[scope, :] = sample[0, :, :, 0, 0]
                    elif mode == "argmax":
                        argmax = self.ef_array._argmax(phi_selected, **kwargs)
                        cur_value[scope, :] = argmax[:, :, 0, 0]
                    else:
                        raise AssertionError(f"Unknown backtracking mode {mode}")

                values[n, :, :] = cur_value

            return values

    # --------------------------------------------------------------------------------

    def set_marginalization_idx(self, idx) -> None:
        """Set indicices of marginalized variables."""
        self.ef_array.set_marginalization_idx(idx)

    def get_marginalization_idx(self):
        """Get indicices of marginalized variables."""
        return self.ef_array.get_marginalization_idx()

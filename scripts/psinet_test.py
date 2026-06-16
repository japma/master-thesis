from torchinfo import summary

from models.cspn.psinet.einsum_network import EinsumNetwork, Args
from models.cspn.psinet.graph import check_graph, random_binary_trees
from models.cspn.psinet.exponential_family_array import NormalArray
from models.cspn.psinet.nns import MLP

num_var = 28 * 28

cspn_graph = random_binary_trees(num_var=num_var, depth=4, num_repetitions=10)

ok, msg = check_graph(cspn_graph)
assert ok, msg


args = Args(
    num_var=num_var,
    num_dims=1,
    num_input_distributions=10,
    num_sums=10,
    num_classes=1,
    exponential_family=NormalArray,
    exponential_family_args={"min_var": 1e-3, "max_var": 1.0},
    use_em=True,
)

conditioning_network = MLP(
    in_dim=num_var,
    out_dims=[(args.num_input_distributions, args.num_classes)],
    h_dims=[100],
)

einet = EinsumNetwork(cspn_graph, conditioning_network, args)
einet.initialize()

summary(einet)

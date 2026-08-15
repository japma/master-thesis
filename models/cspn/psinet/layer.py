import torch


class Layer(torch.nn.Module):
    """
    Abstract layer class. Specifies functionality every layer in an EiNet should implement.
    """

    def __init__(self) -> None:
        super().__init__()
        self.prob = None

    def default_initializer(self):
        """
        Produce suitable initial parameters for the layer.
        :return: initial parameters
        """
        raise NotImplementedError

    def initialize(self, initializer=None):
        """
        Initialize the layer, e.g. with return value from default_initializer(self).
        :param initializer: 'default', or custom (typically a Tensor)
                            'default' means that the layer simply calls its own default_initializer(self), in stores
                            the parameters internally.
                            custom (typically a Tensor) means that you pass your own initializer.
        :return: None
        """
        raise NotImplementedError

    def forward(self, x=None):
        """
        Compute the layer. The result is always a tensor of log-densities of shape (batch_size, num_dist, num_nodes),
        where num_dist is the vector length (K in the paper) and num_nodes is the number of PC nodes in the layer.

        :param x: input data (Tensor).
                  If self.num_dims == 1, this can be either of shape (batch_size, self.num_var, 1) or
                  (batch_size, self.num_var).
                  If self.num_dims > 1, this must be of shape (batch_size, self.num_var, self.num_dims).
                  Not all layers use this argument.
        :return: log-density tensor of shape (batch_size, num_dist, num_nodes), where num_dist is the vector length
                 (K in the paper) and num_nodes is the number of PC nodes in the layer.
        """
        raise NotImplementedError

    def backtrack(self, *args, **kwargs):
        """
        Defines routines for backtracking in EiNets, for sampling and MPE approximation.

        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def reparam_function(self):
        """Return a function which transforms a tensor of unconstrained values into feasible parameters."""
        raise NotImplementedError

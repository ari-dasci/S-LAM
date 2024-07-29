import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    Positional Encoding class for Transformers.

    This module implements positional encoding for a Transformer model. Positional encoding helps
    maintain the temporal information of the data. The chosen implementation follows the vanilla
    method as originally defined in the "Attention Is All You Need" paper.

    Parameters
    ----------
    d_model : int
        The dimensionality of the model, representing the number of features or channels in
        the input and output embeddings.
    dropout : float, optional
        The dropout probability applied to the positional encodings, by default 0.1.
    max_len : int, optional
        The maximum sequence length for which positional encodings are calculated, by default 5000.

    Attributes
    ----------
    pe : torch.Tensor
        The positional encodings matrix.

    Methods
    -------
    forward(x, pos=0)
        Forward pass through the Positional Encoding layer.

    Examples
    --------
    Create a PositionalEncoding instance:

    >>> positional_encoder = PositionalEncoding(d_model=512, dropout=0.1, max_len=1000)
    >>> input_data = torch.rand(32, 1000, 512)
    >>> output_data = positional_encoder(input_data)

    Notes
    -----
    PositionalEncoding is a crucial component in Transformer models to capture the order of
    tokens in input sequences.

    References
    ----------
    Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., ... & Polosukhin, I. (2017).
    Attention is all you need. In Advances in neural information processing systems (pp. 30-31).

    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # Not registered as parameter (it is not) but saved along them
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        """
        Apply positional encoding to the input data.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to which positional encoding is applied.
        pos : int, optional
            The starting position for adding positional encodings, by default 0.

        Returns
        -------
        torch.Tensor
            The input tensor with positional encoding added and dropout applied.

        """
        x = x + self.pe[pos:pos+x.size(0), :]
        return self.dropout(x)
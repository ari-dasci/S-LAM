import torch.nn as nn
import torch
import torch.functional as F
from attn import AttentionLayer

class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer class for the Transformer model.

    This module represents a single layer of the Transformer encoder. It consists of
    multi-head self-attention, feedforward neural networks, and residual connections with
    layer normalization.

    Parameters
    ----------
    d_model : int
        The dimensionality of the model, representing the number of features or channels in
        the input and output embeddings.
    nhead : int
        The number of attention heads in the multi-head self-attention mechanism.
    attn : nn module for attention
        The attention module to use in the Transformer Encoder Layer.
    w_size : int
        The window size for the attention mechanism.
    dim_feedforward : int, optional
        The dimensionality of the feedforward neural networks, by default 16.
    dropout : float, optional
        The dropout probability applied to various parts of the layer, by default 0.
    l1 : nn module, optional
        The linear transformation module to use for the first part of the feedforward network,
        by default nn.Linear.
    l1_args : dict, optional
        The arguments to pass to the first linear transformation module, by default {"dtype":torch.float64}.
    l2 : nn module, optional  
        The linear transformation module to use for the second part of the feedforward network,
        by default nn.Linear.

    Attributes
    ----------
    self_attn : nn.MultiheadAttention
        Multi-head self-attention mechanism.
    l1 : nn module
        Linear transformation for the first part of the feedforward network.
    dropout : nn.Dropout
        Dropout layer applied to the input and output of the self-attention and feedforward networks.
    l2 : nn module
        Linear transformation for the second part of the feedforward network.
    dropout1 : nn.Dropout
        Dropout layer applied to the output of the first part of the feedforward network.
    dropout2 : nn.Dropout
        Dropout layer applied to the output of the second part of the feedforward network.
    activation : nn.LeakyReLU
        Activation function used in the feedforward network.

    Methods
    -------
    forward(src, src_mask=None, src_key_padding_mask=None, is_causal=None)
        Forward pass through the Transformer Encoder Layer.

    Examples
    --------
    Create a TransformerEncoderLayer instance:

    >>> encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1)
    >>> input_data = torch.rand(32, 100, 512)
    >>> output_data = encoder_layer(input_data)

    Notes
    -----
    The Transformer Encoder Layer is a building block in the Transformer model, responsible for
    processing the input sequence and capturing dependencies between tokens.

    References
    ----------
    Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., ... & Polosukhin, I. (2017).
    Attention is all you need. In Advances in neural information processing systems (pp. 30-31).

    """
    def __init__(self, d_model, nhead, attn, attn_params, w_size, dim_feedforward=None, dropout=0, 
                 l1=nn.Linear, l1_args={"dtype":torch.float64}, 
                 l2=nn.Linear, l2_args={"dtype":torch.float64}):
        super(TransformerEncoderLayer, self).__init__()
        self.dim_feedforward = dim_feedforward if dim_feedforward is not None else d_model
        # attn parameter must be the class itself, not an instance
        self.attn = attn(**attn_params)
        self.self_attn = AttentionLayer(self.attn, w_size, d_model, nhead)
        #self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, dtype=torch.float64)
        self.l1 = l1(d_model, self.dim_feedforward, **l1_args)
        #self.linear1 = nn.Linear(d_model, self.dim_feedforward, dtype=torch.float64)
        self.dropout = nn.Dropout(dropout)
        self.l2 = l2(self.dim_feedforward, d_model, **l2_args)
        #self.linear2 = nn.Linear(self.dim_feedforward, d_model, dtype=torch.float64)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.normalization1 = nn.LayerNorm(d_model)
        self.normalization2 = nn.LayerNorm(d_model)

        self.activation = nn.LeakyReLU(True)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=None):
        """
        Apply forward pass through the Transformer Encoder Layer.

        Parameters
        ----------
        src : torch.Tensor
            The input tensor to the layer.
        src_mask : torch.Tensor, optional
            Mask to prevent attention to certain positions in the input sequence, by default None.
        src_key_padding_mask : torch.Tensor, optional
            Mask to prevent attention to padding positions in the input sequence, by default None.
        is_causal : bool, optional
            A flag to control the causality of the self-attention, by default None.

        Returns
        -------
        torch.Tensor
            The output tensor after processing through the layer.

        """
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.normalization1(src)
        residual_src = src
        src2 = self.l2(self.dropout(self.activation(self.l1(src))))
        src = src + self.dropout2(src2)
        # Add residual connection and layer normalization
        src = src + residual_src
        src = self.normalization2(src)
        return src

# This might not be necessary with the new encoder layer
""" 
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm2(x+y), attn
 """
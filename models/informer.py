import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerDecoder
import pytorch_lightning as pl

from attn import FullAttention, ProbAttention, LocalAttention, AttentionLayer
from embed import PositionalEncoding
from base_module import Transformer_BaseModule
from encoder import TransformerEncoderLayer
from decoder import TransformerDecoderLayer


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1,2)
        return x

class InformerModel(Transformer_BaseModule):
    """
    Informer Model class for time-series prediction.

    This module represents a basic Transformer model for sequence-to-sequence tasks, consisting of
    an encoder and a decoder. The encoder and decoder each contain multiple layers of Transformer
    Encoder and Decoder Layers.

    Parameters
    ----------
    feats : int
        The number of features or channels in the input data.
    lr : float, optional
        The learning rate for training, by default 0.001.
    n_window : int, optional
        The window size for temporal context, by default 10.
    batch : int, optional
        The batch size for training, by default 128.

    Attributes
    ----------
    name : str
        The name of the model, set to 'Transformer_basic'.
    lr : float
        The learning rate for training.
    batch : int
        The batch size for training.
    n_feats : int
        The number of features in the input data.
    n_window : int
        The window size for temporal context.
    n : int
        The total number of input features, calculated as `n_feats * n_window`.
    pos_encoder : PositionalEncoding
        The positional encoding module to capture temporal information in the input data.
    transformer_encoder : nn.TransformerEncoder
        The Transformer Encoder module for encoding the input sequence.
    transformer_decoder : nn.TransformerDecoder
        The Transformer Decoder module for decoding the output sequence.
    fcn : nn.Sigmoid
        The sigmoid activation function applied to the model's output.

    Methods
    -------
    forward(src, tgt)
        Forward pass through the Transformer Model for sequence-to-sequence tasks.

    Examples
    --------
    Create a TransformerModel instance:

    >>> model = TransformerModel(feats=256, lr=0.001, n_window=15, batch=64)
    >>> input_data = torch.rand(64, 100, 256)
    >>> target_data = torch.rand(64, 100, 256)
    >>> output_data = model(input_data, target_data)

    Notes
    -----
    The Transformer Model is a powerful architecture for various sequence-to-sequence tasks,
    such as machine translation and text generation. It leverages self-attention mechanisms to capture
    dependencies between elements in the input and output sequences.

    References
    ----------
    Zhou, Haoyi, et al. "Informer: Beyond efficient transformer for long sequence time-series forecasting." 
    Proceedings of the AAAI conference on artificial intelligence. Vol. 35. No. 12. 2021. 

    """
    def __init__(self, feats, loss, optimizers, metrics, 
                n_window=10, factor=5, d_model=512, n_heads=8, num_encoder_layers=3, num_decoder_layers=2, d_ff=512, 
                output_attention = 'False', attn='prob', neigh_size = None, splits = None, **kwargs):
        super(TransformerModel, self).__init__(loss, optimizers, metrics, **kwargs)    
        self.name = 'Transformer_basic'
        self.n_feats = feats
        self.n_window = n_window
        self.n = self.n_feats * self.n_window

        self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
        # Attention
        Attn = ProbAttention if attn=='prob' else LocalAttention if attn=='local' else FullAttention

        # Por que d_model y nheads es feats?????
        encoder_layers = TransformerEncoderLayer(d_model=feats, nhead=feats, attn=Attn, 
                                                 w_size=self.n_window, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        decoder_layers = TransformerDecoderLayer(d_model=feats, nhead=feats, attn=nn.MultiheadAttention, 
                                                 w_size=self.n_window, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_decoder_layers)

        # Encoder
        self.encoder = Encoder(
            [
                TransformerEncoderLayer(
                    d_model=feats, nhead=feats,
                    attn = AttentionLayer(Attn(neigh_size = neigh_size, factor = factor, attention_dropout=0.1, output_attention=output_attention, splits = splits), 
                                n_window, feats, feats, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

    def forward(self, src, tgt):
        """
        Apply forward pass through the Transformer Model for sequence-to-sequence tasks.

        Parameters
        ----------
        src : torch.Tensor
            The input tensor representing the source sequence.
        tgt : torch.Tensor
            The input tensor representing the target sequence.

        Returns
        -------
        torch.Tensor
            The output tensor representing the predicted sequence.

        """
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        x = self.transformer_decoder(tgt, memory)
        return x
    


    class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]

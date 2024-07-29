import transformer

class LocalTransformerModel(Transformer_BaseModule):
    """
    Transformer Model with local attention class for sequence-to-sequence tasks.

    This module implements our transformer arquitecture with local attention for sequence-to-sequence tasks, 
    consisting of an encoder and a decoder. The encoder and decoder each contain multiple layers of Transformer
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
    Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., ... & Polosukhin, I. (2017).
    Attention is all you need. In Advances in neural information processing systems (pp. 30-31).

    """
    def __init__(self, feats, loss, optimizers, metrics, n_window=10, num_encoder_layers=1, num_decoder_layers=1, **kwargs):
        super(TransformerModel, self).__init__(loss, optimizers, metrics, **kwargs)    
        self.name = 'Transformer_local'
        self.n_feats = feats
        self.n_window = n_window
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        decoder_layers = TransformerDecoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_decoder_layers)

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
        # It seems like more than 1 layer generates non-learning schema, loss goes crazy
        # Possible solution to add residual 
        x = self.transformer_decoder(tgt, memory)
        return x
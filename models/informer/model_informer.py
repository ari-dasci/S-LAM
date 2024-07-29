import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from encoder_informer import Encoder, EncoderLayer, ConvLayer, EncoderStack
from decoder_informer import Decoder, DecoderLayer
from attn_informer import FullAttention, ProbAttention, AttentionLayer
from embed_informer import DataEmbedding

class Informer(pl.LightningModule):
    def __init__(self, loss, optimizers, metrics, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device='cuda:0', **kwargs):
        super(Informer, self).__init__()

        self.kwargs = kwargs
        self.loss = loss
        self.optimizers_ = optimizers
        self.metrics = metrics

        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        self.dev = device

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
    
    def training_step(self, batch, batch_idx):
        x,ts_x,y,ts_y = batch
        y_hat, y_true = self._process_one_batch(x, y, ts_x, ts_y)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        #self.log('train_acc', self.accuracy(y_hat, y))
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x,ts_x,y,ts_y = batch
        y_hat, y_true = self._process_one_batch(x, y, ts_x, ts_y)
        val_loss = self.loss(y_hat, y)
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x,ts_x,y,ts_y = batch
        y_hat, y_true = self._process_one_batch(x, y, ts_x, ts_y)

        # compute test loss
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True)

        # compute the remaining test metrics
        for name, f in self.metrics.items():
            value = f(y_hat, y)
            self.log(str('test_' + name), value, on_step=True, on_epoch=True)
        return y_hat


    def configure_optimizers(self):
        opt = self.optimizers_(self.parameters(), **self.kwargs)
        return opt
    
    def _process_one_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.double().to(self.dev)
        batch_y = batch_y.double().to(self.dev)

        batch_x_mark = batch_x_mark.double().to(self.dev)
        batch_y_mark = batch_y_mark.double().to(self.dev)

        # decoder input
        dec_inp = torch.zeros([batch_y.shape[0], self.pred_len, batch_y.shape[-1]]).double().to(self.dev)
        dec_inp = torch.cat([batch_y[:,:0,:], dec_inp], dim=1).double().to(self.dev)
        # encoder - decoder
        #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        outputs = self(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        f_dim = -1
        batch_y = batch_y[:,-self.pred_len:,f_dim:].to(self.dev)

        return outputs, batch_y


class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=[3,2,1], d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(e_layers))) # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
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

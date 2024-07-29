# Models

This folder contains the files necessary to define the transformer models, the attention mechanisms and encoder-decoder files.

- attn.py: this file contains all attention mechanisms implemented. The last class (AttentionLayer) is used to generalize the attention mechanisms. This enables the module to be used with as many mechanisms as desired via the same interface for the main transformer model.
- base_module.py: transformer base module. This is an implementation to make easier to code the model itself. It relies on pytorch lightning base module to implement the train, test and validation.
- decoder.py: implementation of the decoder layer of the transformer.
- encoder.py: implementation of the encoder layer of the transformer.
- embed.py: implementation of the encoding of the transformer. By default the vanilla implementation is used, this is positional encoding.
- informer.py: implementation of the informer model relying on our structure. TODO: this is not functional yet.
- local_transformers.py: implementation of local transformers. TODO: this is not functional yet.
- masks.py: masks used in local attention from informer. This file is auxiliary but related to the models.
- transformer.py: main file of this folder as it implements and generalizes the transformer model. The main purpose is that all transformer models can be created from this implementation changing the parameters and attention mechanisms.

https://github.com/zhouhaoyi/Informer2020
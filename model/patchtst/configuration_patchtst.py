""" GridTST model configuration"""

from transformers import PretrainedConfig

class PatchTSTConfig(PretrainedConfig):

    model_type = "patchtst"

    def __init__(
        self,
        num_channels=800,
        seq_len=96,
        label_len=96,
        stride=8,
        patch_len=16,
        d_model=128,
        ffn_dim=256,
        dropout=0.2,
        num_heads=16,
        attention_dropout=0.0,
        num_layers=3,
        qkv_bias=True,
        init_std=0.2,
        head_dropout=0.0,
        revin_affine=False,
        norm_type='batchnorm', # layernorm 
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.norm_type = norm_type
        self.revin_affine = revin_affine
        self.dropout = dropout
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.label_len = label_len
        self.stride = stride
        self.patch_len = patch_len
        self.d_model = d_model
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.qkv_bias = qkv_bias
        self.init_std = init_std
        self.head_dropout = head_dropout
        self.num_patches = int((self.seq_len-self.patch_len)/self.stride + 1) + 1

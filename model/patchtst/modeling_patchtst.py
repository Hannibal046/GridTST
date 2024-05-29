from transformers import PreTrainedModel
from .configuration_patchtst import PatchTSTConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

class RevIN(nn.Module):
    # code from https://github.com/ts-kim/RevIN, with minor modifications
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

class PatchTSTSelfAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self,config):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.attention_dropout = config.attention_dropout
        self.head_dim = self.d_model // self.num_heads

        if (self.head_dim * self.num_heads) != self.d_model:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.d_model}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=config.qkv_bias)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=config.qkv_bias)
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=config.qkv_bias)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=config.qkv_bias)
        self.attention_out_dropout = nn.Dropout(config.dropout)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states,
        attention_mask = None,
        output_attentions = False,
    ):
        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)


        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.d_model)

        attn_output = self.out_proj(attn_output)
        attn_output = self.attention_out_dropout(attn_output)

        return attn_output
    
class PatchTSTFFN(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model,config.ffn_dim)
        self.activation_fn = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(config.ffn_dim,config.d_model)

    def forward(self,hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class PatchTSTEncoderLayer(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.num_channels = config.num_channels

        ## Attention
        self.self_attention = PatchTSTSelfAttention(config)
        self.attention_dropout = nn.Dropout(config.dropout)
        if config.norm_type == 'batchnorm':
            self.attention_norm = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(config.d_model), Transpose(1,2))
        elif config.norm_type == 'layernorm':
            self.attention_norm = nn.LayerNorm(config.d_model)

        ## FFN
        self.ffn = PatchTSTFFN(config)
        self.ffn_dropout = nn.Dropout(config.dropout)
        if config.norm_type == 'batchnorm':
            self.ffn_norm = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(config.d_model), Transpose(1,2))
        elif config.norm_type == 'layernorm':
            self.ffn_norm = nn.LayerNorm(config.d_model)

        
        
    def forward(
        self,
        hidden_states,
    ):  
        ## Self-Attention
        _,seq_len,d_model = hidden_states.shape
        residual = hidden_states 
        hidden_states = self.self_attention(hidden_states)
        hidden_states = residual + self.attention_dropout(hidden_states)
        hidden_states = self.attention_norm(hidden_states)

        ## FFN
        residual = hidden_states
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + self.ffn_dropout(hidden_states)
        hidden_states = self.ffn_norm(hidden_states)
        
        return hidden_states

class PatchTSTFlattenHead(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(config.num_patches*config.d_model,config.label_len)
        self.dropout = nn.Dropout(config.head_dropout)
    def forward(
        self,hidden_states #[batch_size*num_feature,num_patch,patch_len]
    ):
        # hidden_states = hidden_states.permute(0,2,1)  
        hidden_states = self.flatten(hidden_states)
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
    
class PatchTSTPreTrainedModel(PreTrainedModel):
    config_class = PatchTSTConfig
    base_model_prefix = "model"
    main_input_name = "input_values"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Embedding):
            module.weight.data.uniform_(-0.02, 0.02)

class PatchTSTEncoder(PatchTSTPreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        self.num_channels = config.num_channels

        self.layers = nn.ModuleList(
            PatchTSTEncoderLayer(config) for _ in range(config.num_layers)
        )
        self.post_init()
    
    def forward(self,hidden_states):
        batch_size,num_channels,seq_len,d_model = hidden_states.shape
        hidden_states = hidden_states.view(batch_size*num_channels,seq_len,d_model)
        for idx,layer in enumerate(self.layers):
            hidden_states = layer(hidden_states)
        return hidden_states
        
class PatchTSTModel(PatchTSTPreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        
        self.input_embedding = nn.Linear(config.patch_len,config.d_model)
        self.pos_embedding = nn.Embedding(config.num_patches,config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.encoder = PatchTSTEncoder(config)
        self.post_init()
    
    def forward(
        self,input_values # [bs,num_channels,num_patches,patch_len]
    ) :
        batch_size,num_channels,seq_len,patch_len = input_values.shape
        input_values = input_values.reshape(batch_size*num_channels,seq_len,patch_len)
        hidden_states = self.dropout(
                    self.input_embedding(input_values)
                    +self.pos_embedding(torch.arange(seq_len,device=input_values.device)))
        hidden_states = self.encoder(hidden_states.reshape(batch_size,num_channels,seq_len,-1))
        return hidden_states
    
class PatchTSTForTimeSeriesPrediction(PatchTSTPreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        self.patch_len = config.patch_len
        self.stride = config.stride
        self.padding = nn.ReplicationPad1d((0,config.stride))
        self.revin_layers = RevIN(config.num_channels,affine=config.revin_affine)
        self.model = PatchTSTModel(config)
        self.head = PatchTSTFlattenHead(config)
        self.post_init()
        
    def patchify(self,input_values):
        input_values =  self.padding(input_values)# [batch_size,num_channels,seq_len+self.stride]
        input_values = input_values.unfold(dimension=-1,size=self.patch_len,step=self.stride) # [batch_size,num_channels,num_patch,patch_len]
        return input_values

    def forward(
        self,
        input_values, # [batch_size,seq_len,num_channels]
        labels = None
    ):
        input_values = self.revin_layers(input_values,'norm')
        input_values = input_values.permute(0,2,1) # [batch_size,num_channels,seq_len]

        patched_input_values = self.patchify(input_values) #[batch_size,num_channels,num_patch,patch_len]
        batch_size,num_channels,seq_len,patch_len = patched_input_values.shape
        hidden_states = self.model(patched_input_values)
        output = self.head(hidden_states) # [batch_size*num_feature,label_len]
        output = output.reshape(batch_size,num_channels,-1)

        output = self.revin_layers(output.permute(0,2,1),'denorm')

        loss = None
        if labels is not None:
            loss = F.mse_loss(output,labels.float())
            return loss,output
        else:
            return output
    

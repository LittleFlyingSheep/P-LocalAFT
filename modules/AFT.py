import torch
from torch import nn
from typing import Optional

class AFT_local_attention(nn.Module):
    '''
    The implement of the AFT-local attention strategy.
    '''
    def __init__(self,
                 d_model: int,
                 audio_seq_len: int,
                 seq_len: int,
                 local_window_size: int,
                 bias: bool = True,
                 dropout: float = .1
                 ):
        super().__init__()

        # Local window size $s$
        self.load_window_size = local_window_size

        # These transform the 'query', 'key', 'value' vectors.
        self.query = nn.Linear(d_model, d_model, bias=bias)
        self.key = nn.Linear(d_model, d_model, bias=bias)
        self.value = nn.Linear(d_model, d_model, bias=bias)

        # Pair-wise positional biases $w \in \mathbb{R}^{T \times T}$
        self.pos_bias = nn.Parameter(torch.zeros(seq_len, audio_seq_len), requires_grad=True)

        # Mask for $w_{t,t'}$
        self.local_mask = nn.Parameter(self.create_local_mask(seq_len, audio_seq_len, local_window_size), requires_grad=False)

        # Activation $\sigma$
        self.activation = nn.Sigmoid()

        # Output layer
        self.output = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)


    @staticmethod
    def create_local_mask(seq_len, audio_seq_len, local_window_size):
        """
        #### Create local mask

        Considering the local_window_size $s$, which decides the mask diag size.

        This creates a mask for
            \begin{align}
            m_{t,t'} =
            \begin{cases}
            1, & \text{for $\lvert t-t' \rvert \lt s$} \\
            0, & \text{otherwise}
            \end{cases}
            \end{align}
        """

        # Initialize to ones
        local_mask = torch.ones(seq_len, audio_seq_len, dtype=torch.bool)
        # Make $t' - t \ge s$ zero
        local_mask = torch.tril(local_mask, local_window_size-1) # 返回矩阵下三角部分，其余值为0
        # Make $t - t' \ge s$ zero
        local_mask = torch.triu(local_mask, -(local_window_size - 1)) # 返回矩阵上三角部分，其余值为0

        return local_mask

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None,
                ):
        """
        `query`, `key` and `value` are the tensors that store
        collection of token embeddings for  *query*, *key* and *value*.
        They have shape `[seq_len, batch_size, d_model]`.
        `mask` has shape `[seq_len, seq_len, batch_size]` and
        `mask[i, j, b]` indicates whether for batch `b`,
        query at position `i` has access to key-value at position `j`.
        """

        # `query`, `key` and `value`  have shape `[seq_len, batch_size, d_model]`
        q_seq_len, _, _ = query.shape
        k_seq_len, _, _ = key.shape

        # Sequence mask shape check, use it to remain the tril matrix.
        if mask is not None:
            # `mask` has shape `[seq_len_q, seq_len_k, batch_size]`,
            # where first dimension is the query dimension.
            # If the query dimension is equal to $1$ it will be broadcasted.
            assert mask.shape[0] == 1 or mask.shape[0] == query.shape[0]
            assert mask.shape[1] == key.shape[0]
            assert mask.shape[2] == 1 or mask.shape[2] == query.shape[1]

        # Transform query, key and value embeddings
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # padding_mask
        if padding_mask is not None:
            query.masked_fill_(padding_mask, float('-inf'))
            key.masked_fill_(padding_mask, float('-inf'))

        '''
        Get
            \begin{align}
            m_{t,t'} =
            \begin{cases}
            1, & \text{for $\lvert t-t' \rvert \lt s$} \\
            0, & \text{otherwise}
            \end{cases}
            \end{align}
        '''
        # using the mask
        # print(self.local_mask.shape, self.pos_bias.shape, q_seq_len, k_seq_len)
        pos_bias = self.pos_bias[:q_seq_len, :k_seq_len] * self.local_mask[:q_seq_len, :k_seq_len]
        pos_bias = pos_bias.unsqueeze(-1)
        if mask is not None:
            # hop the mask is a tril, and get its inverse to remain the tril of $w$.
            pos_bias.masked_fill_(~mask, float('-inf')) # exp(-inf) = 0

        # if q_seq_len < k_seq_len: print(pos_bias.squeeze(-1))
        '''
        \begin{align}
        Y_t &= \sigma(Q_t) \odot
        \frac{\sum_{t'=1}^T \exp(K_{t'} + w_{t,t'}) \odot V_{t'}}
        {\sum_{t'=1}^T \exp(K_{t'} + w_{t,t'})} \\
        &= \sigma(Q_t) \odot
           \frac{\sum_{t'=1}^T \exp(w_{t,t'}) \odot \exp(K_{t'}) \odot V_{t'}}
           {\sum_{t'=1}^T \exp(w_{t,t'}) \odot \exp(K_{t'})}
        \end{align}

        We compute $\exp(w_{t,t'})$, $\exp(K_{t'}) \odot V_{t'}$ and $\exp(K_{t'})$
        separately and do a matrix multiplication. We use einsum for clarity.

        We subtract $\max_{t'}(K_{t'})$ and $\max_{t'}(w_{t,t'})$ before calculating the exponents to stabilize
        the softmax calculation.

        If $x_i$ is large $\exp(x_i)$ becomes huge and the computation of
        $\frac{\sum\exp(x_i)y_i}{\sum\exp(x_i)}$becomes unstable.
        Subtracting a constant before calculating the exponent from numerator and denominator will cancel out.
        and can help stabilize the computation.
        So we subtract $\max(x_i)$ to stabilize the computation.
        '''
        max_key = key.max(dim=0, keepdims=True)[0] # the value of max function, which is the max one of 0 dimension.
        max_pos_bias = pos_bias.max(dim=1, keepdims=True)[0] # the value of max function, which is the max one of 1 dimension.

        # $\exp \big(K_{t'}- \max_{t'}(K_{t'})\big)$, to avoid the huge value from exp computation.
        exp_key = torch.exp(key - max_key)
        # $\exp \big(w_{t,t'} - \max_{t'}(w_{t,t'})\big)$, to avoid the huge value from exp computation.
        exp_pos_bias = torch.exp(pos_bias - max_pos_bias)

        # The numerator part $\sum_{t'=1}^T \exp(w_{t,t'}) \odot \exp(K_{t'}) \odot V_{t'}$
        num = torch.einsum('ijb,jbd->ibd', exp_pos_bias, exp_key * value)
        # The denominator part $\sum_{t'=1}^T \exp(w_{t,t'}) \odot \exp(K_{t'})$
        den = torch.einsum('ijb,jbd->ibd', exp_pos_bias, exp_key)

        '''
        Output $$Y_t = \sigma(Q_t) \odot
                \frac{\sum_{t'=1}^T \exp(w_{t,t'}) \odot \exp(K_{t'}) \odot V_{t'}}
                {\sum_{t'=1}^T \exp(w_{t,t'}) \odot \exp(K_{t'})}$$     
        '''

        y = self.activation(query) * num / den

        y = self.output(y)
        y = self.dropout(y)

        # Add residual and norm
        return y

class AFT_local_encoder_layer(nn.Module):
    def __init__(self,
                 d_model: int,
                 audio_seq_len: int,
                 local_window_size: int,
                 bias: bool = True,
                 dropout: float = .1,
                 ffn_dim: int = 2048,
                 ):
        super().__init__()

        # Attention
        self.aft_local_atten = AFT_local_attention(d_model, audio_seq_len, audio_seq_len, local_window_size, bias, dropout)
        # Layer norm after attention
        self.layer_norm_atten = nn.LayerNorm(d_model)

        # Feedforward
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        # Layer norm after feedforward
        self.layer_norm_ffn = nn.LayerNorm(d_model)

    def forward(self,
                x: torch.Tensor,
                # query: torch.Tensor,
                # key: torch.Tensor,
                # value: torch.Tensor,
                encoder_mask: Optional[torch.Tensor] = None
                ):

        # Remain the input for residual
        residual_atten = x
        # AFT attention computation
        attention = self.aft_local_atten(x, x, x, encoder_mask)
        # Layer norm for attention
        attention = self.layer_norm_atten(residual_atten + attention)

        # Remain the hidden for residual
        residual_ffn = attention
        # Feed forward
        output = self.feedforward(attention)
        # Layer norm for feed forward
        output = self.layer_norm_ffn(residual_ffn + output)

        return output

class AFT_local_encoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 audio_seq_len: int,
                 local_window_size: int,
                 bias: bool = True,
                 dropout: float = .1,
                 ffn_dim: int = 2048,
                 num_layers: int = 3,
                 ):
        super().__init__()

        self.encoder_layers = nn.ModuleList(
            [AFT_local_encoder_layer(d_model, audio_seq_len, local_window_size, bias, dropout, ffn_dim)
             for _ in range(num_layers)]
        )

    def forward(self,
                x: torch.Tensor,
                encoder_mask: Optional[torch.Tensor] = None
                ):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, encoder_mask)

        return x

class AFT_local_decoder_layer(nn.Module):
    def __init__(self,
                 d_model: int,
                 audio_seq_len: int,
                 seq_len: int,
                 local_window_size: int,
                 bias: bool = True,
                 dropout: float = .1,
                 ffn_dim: int = 2048,
                 ):
        super().__init__()

        # Masked attention
        self.aft_local_masked_atten = AFT_local_attention(
            d_model=d_model, seq_len=seq_len, audio_seq_len=seq_len,
            local_window_size=local_window_size, bias=bias,
            dropout=dropout)
        # Layer norm after attention
        self.layer_norm_masked_atten = nn.LayerNorm(d_model)

        # Attention
        self.aft_local_atten = AFT_local_attention(
            d_model=d_model, seq_len=seq_len, audio_seq_len=audio_seq_len,
            local_window_size=local_window_size, bias=bias,
            dropout=dropout
        )
        # Layer norm after attention
        self.layer_norm_atten = nn.LayerNorm(d_model)

        # Feedforward
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        # Layer norm after feedforward
        self.layer_norm_ffn = nn.LayerNorm(d_model)

    def forward(self,
                x: torch.Tensor,
                memory: torch.Tensor,
                decoder_mask: Optional[torch.Tensor] = None,
                decoder_padding_mask: Optional[torch.Tensor] = None
                ):
        '''
        decoder_mask: should be a tril matrix with unsqueeze(-1), like shape (Seq, Seq, 1)
        '''

        # Remain the input for residual
        residual_masked_atten = x
        # Masked AFT attention computation
        attention = self.aft_local_masked_atten(x, x, x, decoder_mask, decoder_padding_mask)
        # Layer norm for attention
        attention = self.layer_norm_masked_atten(residual_masked_atten + attention)

        # Remain the hidden for residual
        residual_atten = attention
        # AFT attention computation
        attention = self.aft_local_atten(attention, memory, memory)
        # Layer norm for attention
        attention = self.layer_norm_atten(residual_atten + attention)

        # Remain the hidden for residual
        residual_ffn = attention
        # Feed forward
        output = self.feedforward(attention)
        # Layer norm for feed forward
        output = self.layer_norm_ffn(residual_ffn + output)

        return output

class AFT_local_decoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 audio_seq_len: int,
                 seq_len: int,
                 local_window_size: int,
                 bias: bool = True,
                 dropout: float = .1,
                 ffn_dim: int = 2048,
                 num_layers: int = 3,
                 ):
        super().__init__()

        self.decoder_layers = nn.ModuleList(
            [AFT_local_decoder_layer(d_model, audio_seq_len, seq_len, local_window_size, bias, dropout, ffn_dim)
             for _ in range(num_layers)]
        )

    def forward(self,
                x: torch.Tensor,
                memory: torch.Tensor,
                decoder_mask: Optional[torch.Tensor] = None,
                decoder_padding_mask: Optional[torch.Tensor] = None
                ):

        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, memory, decoder_mask, decoder_padding_mask)

        return x

if __name__ == '__main__':
    a = AFT_local_attention.create_local_mask(3, 2)
    print(a)
    x = torch.ones(3,2,2)
    n = AFT_local_attention(2, 3, 2)
    y = n(x, x, x)
    print(y.shape)
    # print(y)
    m = AFT_local_encoder_layer(2, 3, 2)
    z = m(x)
    print(z.shape)
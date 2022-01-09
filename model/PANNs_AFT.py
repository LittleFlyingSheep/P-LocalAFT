import torch
from torch import Tensor,nn
from torch.nn import (TransformerEncoder, TransformerDecoder,
                      TransformerEncoderLayer, TransformerDecoderLayer)
import torch.nn.functional as F
import math
from einops.layers.torch import Rearrange

from modules.SpecAugment import SpecAugmentation
from modules.PANNs import Cnn10
from modules.AFT import AFT_local_encoder, AFT_local_decoder

NHEAD = 8
SOS_IDX = 0
PAD_IDX = 4367
EOS_IDX = 9

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    """Initialize a LayerNorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim, dim),
        #     nn.Dropout(dropout)
        # )
        self.dropout = dropout
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.activation1 = nn.GELU()
        # self.activation2 = nn.GELU()

        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        init_layer(self.fc2)

    def forward(self, x):
        # return self.net(x)

        x = F.dropout(self.activation1(self.fc1(x)), training=self.training, p=self.dropout)
        x = F.dropout(self.fc2(x), training=self.training, p=self.dropout)
        return x

class MLP_Conv_Block(nn.Module):
    def __init__(self, frequency_dim, out_frequency_dim, hid_dim=256, dropout=.2):
        super(MLP_Conv_Block, self).__init__()

        self.conv = nn.Conv1d(in_channels=frequency_dim, out_channels=out_frequency_dim,
                              kernel_size=3, stride=1, padding=1)
        self.mlp = FeedForward(out_frequency_dim, hid_dim, dropout)

        self.bn1 = nn.BatchNorm1d(out_frequency_dim)
        self.bn2 = nn.BatchNorm1d(out_frequency_dim)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x, pool_type='avg', pool_size=2):
        # x: (B, T, F)

        # Transpose T and F -> (B, F, T) for conv, and turn back for MLP.
        x = self.bn1(F.relu_(self.conv(x.transpose(1, 2)))).transpose(1, 2)

        # Transpose T and F -> (B, T, F) for BatchNorm. -> x: (B, F', T)
        x = self.bn2(F.relu_(self.mlp(x).transpose(1, 2)))


        if pool_type == 'max':
            x = F.max_pool1d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool1d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool1d(x, kernel_size=pool_size)
            x2 = F.max_pool1d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        # x: (B, F', T) -> x: (B, T', F')
        return x.transpose(1, 2)

class MLP_Conv_Mixer(nn.Module):
    def __init__(self, frequency_dim, hid_dim=256, dropout=.2, emb_size=128):
        super(MLP_Conv_Mixer, self).__init__()

        self.bn0 = nn.BatchNorm1d(frequency_dim)
        self.dropout = dropout

        self.block1 = MLP_Conv_Block(frequency_dim, frequency_dim*2, hid_dim, dropout)
        self.block2 = MLP_Conv_Block(frequency_dim*2, frequency_dim*4, hid_dim, dropout)
        self.block3 = MLP_Conv_Block(frequency_dim*4, frequency_dim*4, hid_dim, dropout)
        self.block4 = MLP_Conv_Block(frequency_dim*4, frequency_dim*4, hid_dim, dropout)
        # self.block5 = MLP_Conv_Block(frequency_dim*4, frequency_dim*4, hid_dim, dropout)

        self.fc = nn.Linear(frequency_dim*4, frequency_dim*4)
        self.fc_de = nn.Linear(frequency_dim*4, emb_size)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc)

    def forward(self, x):
        # x:(B, T, F)
        x = x.transpose(1, 2)
        x = self.bn0(x).transpose(1, 2)

        x = F.dropout(self.block1(x), training=self.training, p=self.dropout)
        x = F.dropout(self.block2(x), training=self.training, p=self.dropout)
        x = F.dropout(self.block3(x), training=self.training, p=self.dropout)
        x = F.dropout(self.block4(x), training=self.training, p=self.dropout)
        # x = F.dropout(self.block5(x), training=self.training, p=self.dropout)

        x = F.dropout(self.fc_de(self.fc(x)), training=self.training)
        # x:(B, T', F') -> (T', B, F')
        return x.transpose(0, 1)

class Seq2SeqMLPConvTransformer(nn.Module):
    def __init__(self,
                 emb_size: int,
                 audio_seq_len: int,  # is (the max audio seq length)
                 seq_len: int,  # is (the max seq length)
                 nb_classes: int,  # tgt_vocab_size
                 local_window_size: int,
                 bias: bool = True,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 num_decoder_layers: int = 3,
                 spec_aug: bool = False
                 ):
        super(Seq2SeqMLPConvTransformer, self).__init__()

        self.encoder = Cnn10(spec_aug=spec_aug)
        dict = torch.load('./modules/CNN10_encoder.pth')
        self.encoder.load_state_dict(dict)

        # decoder_layer = TransformerDecoderLayer(d_model=emb_size, nhead=NHEAD,
        #                                         dim_feedforward=dim_feedforward)
        # self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.transformer_decoder = AFT_local_decoder(
            d_model=emb_size, audio_seq_len=audio_seq_len, seq_len=seq_len,
            local_window_size=local_window_size,
            bias=bias, dropout=dropout, ffn_dim=dim_feedforward, num_layers=num_decoder_layers
        )

        self.generator = nn.Linear(emb_size, nb_classes)

        self.nb_classes = nb_classes
        # global PAD_IDX
        # PAD_IDX = nb_classes - 1

        self.bn = nn.BatchNorm1d(emb_size)

        self.tgt_tok_emb = TokenEmbedding(nb_classes, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.src_mask = None
        self.tgt_mask = None
        self.src_padding_mask = None
        self.tgt_padding_mask = None

    def forward(self, src: Tensor,  # (B, T, F)
                tgt: Tensor,  # (B, Seq_tgt)
                with_pad = False,
                mixup_param = None,
                ):
        if mixup_param and self.training:
            lam, index = mixup_param
            src = lam * src + (1-lam) * src[index]

        # if self.spec_augmenter and self.training:
        #     # SpecAugment for src
        #     src = self.spec_augmenter(src.unsqueeze(1)).squeeze(1)

        # The sequence without the last word as input. \
        # Its target output is the sequence without the first word (<sos>).
        trg = tgt.transpose(0, 1).contiguous()[:-1, :]   # (Seq_tgt, B)
        tgt_emb = self.tgt_tok_emb(trg)
        if self.training and mixup_param is not None:
            lam, index = mixup_param
            tgt_emb = lam * tgt_emb + (1 - lam) * tgt_emb[:, index]
        tgt_emb = self.positional_encoding(tgt_emb)
        # print(tgt_emb.shape)
        tgt_emb = self.bn(tgt_emb.transpose(1, 2)).transpose(1, 2)

        # get the mask
        tgt_mask, tgt_padding_mask = create_tgt_mask(trg, with_pad)

        # src: (B, T, F) -> (B, T', F'), F'=4*F; -> memory: (T', B, F') for decoder
        memory = self.encoder(src)
        # print(memory.shape)

        outs = self.transformer_decoder(tgt_emb, memory, tgt_mask, tgt_padding_mask)
        return self.generator(outs).transpose(0, 1).contiguous()

    def greedy_decode(self, src, max_steps=22, start_symbol=0):
        device = src.device
        src = src.transpose(0, 1)   # (Seq, B)
        if self.src_mask is None or self.src_mask.shape[0] != src.shape[0]:
            self.src_mask, self.src_padding_mask = create_src_mask(src)
        memory = self.encode(src,self.src_mask)
        ys = torch.ones(1,src.shape[1]).fill_(start_symbol).type(torch.long).to(device)
        for i in range(max_steps):
            memory = memory.to(device)
            tgt_mask = (generate_square_subsequent_mask(ys.shape[0], device)
                        .type(torch.bool)).to(device)

            out = self.decode(ys, memory, tgt_mask)
            prob = self.generator(out[-1,:])
            _, next_word = torch.max(prob, dim=1)

            ys = torch.cat([ys, next_word.unsqueeze(0)],dim=0)
            # if next_word == EOS_IDX:
            #     break
        ys = nn.functional.one_hot(ys.transpose(0,1),self.nb_classes).transpose(0,1).float()
        return ys[1:, :, :].transpose()

    def init_vars(self, src, k_beam, max_steps=22):
        device = src.device

        # print(src.shape)
        memory = self.encode(src)  # (Seq, B:1, hid)
        outputs = torch.LongTensor([[SOS_IDX]]).to(device)

        tgt_mask, _ = create_tgt_mask(outputs)

        out = self.generator(self.decode(outputs, memory, tgt_mask))
        out = F.softmax(out, dim=-1).transpose(0, 1)  # (Seq, B:1, nb_classes) -> (B:1, Seq, nb_classes)
        out[:, :, -1] = 0  # ignore <pad>

        probs, ix = out[:, -1].topk(k_beam)  # (B:1, nb_classes) -> (B:1, k_beam)
        # log_scores = Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)  # maybe need check
        # log_scores = Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)  # maybe need check
        log_scores = torch.log(probs)
        # print(log_scores.shape)

        outputs = torch.zeros(k_beam, max_steps).long().to(device)  # (Seq/nb_classes, k_beam)
        outputs[:, 0] = SOS_IDX
        outputs[:, 1] = ix[0]  # k_beam words of ix are the candidates.

        # Check the memory is necessary!
        k_memorys = torch.zeros(memory.shape[0], k_beam, memory.shape[-1]).to(device)
        k_memorys[:, :] = memory[:, :]  # expand memory to k numbers (Seq, k_beam, hid)

        return outputs, k_memorys, log_scores

    def beam_search(self, src, k_beam=5, max_steps=22, with_pad=True):
        # to do init
        # src = src.transpose(0, 1)  # (Seq, B)
        # print(src.shape)
        outputs, k_memorys, log_scores = self.init_vars(src, k_beam=k_beam)
        # print(k_memorys)
        # exit()

        device = src.device
        ind = None  # an important variable to decide the final output sequence
        EOS_check = torch.zeros(k_beam).bool()
        for i in range(2, max_steps):
            tgt = outputs[:, :i].transpose(0, 1)
            tgt_mask, _ = create_tgt_mask(tgt)

            out = self.generator(self.decode(tgt, k_memorys, tgt_mask))
            out = F.softmax(out, dim=-1).transpose(0, 1)  # (Seq, k_beam, nb_classes) -> (k_beam, Seq, nb_classes)
            # a, b = torch.max(out,dim=-1)
            # print(outputs[:, :i], a, b, out.shape)
            # if i==5: exit()
            out[:, :, -1] = 0  # ignore <pad>

            outputs, log_scores, EOS_check = self.k_best_outputs(outputs, out, log_scores, i, k_beam, EOS_check)

            ones = (outputs == EOS_IDX).nonzero()  # Occurrences of end symbols for all input summaries.
            summary_lengths = torch.zeros(len(outputs), dtype=torch.long).to(device)

            for vec in ones:
                # i = vec[0]
                i = vec[0]
                EOS_check[i] = True
                if summary_lengths[i] == 0:  # First end symbol has not been found yet
                    # summary_lengths[i] = vec[1]  # Position of first end symbol
                    summary_lengths[i] = vec[1]  # Position of first end symbol

            num_finished_summaries = len([s for s in summary_lengths if s > 0])

            if num_finished_summaries == k_beam:
                # alpha = 0.7
                # div = 1 / (summary_lengths.type_as(log_scores) ** alpha)
                # _, ind = torch.max(log_scores * div, 1)
                _, ind = torch.max(log_scores, 1)
                ind = ind.data[0]
                break

        if ind is None:
            _, ind = torch.max(log_scores, 1)
            ind = ind.data[0]
            ys = outputs[ind][1:]
        else:
            ys = outputs[ind][1:]
        ys = F.one_hot(ys.unsqueeze(0), self.nb_classes).float()
        # print(ys.shape)
        return ys

    def k_best_outputs(self, outputs, out, log_scores, i, k_beam, EOS_check):
        device = log_scores.device
        probs, ix = out[:, -1].topk(k_beam)
        probs[EOS_check] = 1
        # print(probs.shape)
        # log_scores (B:1, nb_classes), maybe need check
        # log_probs = torch.Tensor([math.log(p) for p in probs.view(-1)]).view(k_beam, -1) \
        log_probs = torch.log(probs).to(device) \
                    + log_scores.transpose(0, 1)
        k_probs, k_ix = log_probs.view(-1).topk(k_beam)
        row = k_ix // k_beam
        col = k_ix % k_beam
        # print(ix,'\n',row,'\n',col)

        # print(row, col, outputs.shape, ix.shape)
        outputs[:, :i] = outputs[row, :i]
        outputs[:, i] = ix[row, col]
        EOS_check = EOS_check[row]

        log_scores = k_probs.unsqueeze(0)
        return outputs, log_scores, EOS_check

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def encode(self, src: Tensor):
        return self.encoder(src)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        tgt_emb = self.bn(self.positional_encoding(self.tgt_tok_emb(tgt)).transpose(1, 2)).transpose(1, 2)
        # print(tgt_emb.shape, memory.shape, tgt_mask.shape)
        return self.transformer_decoder(tgt_emb, memory,
            tgt_mask)

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2) # tensor: (maxlen, 1, emb_size)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding) # register a buffer for pos_embedding.
        # print(f'pos_embedding:{pos_embedding.shape}')

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding +
                            self.pos_embedding[:token_embedding.size(0),:])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens: Tensor):
        # print(tokens)
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size) # why "* math.sqrt(self.emb_size)"?

def generate_square_subsequent_mask(sz,device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1).contiguous()
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_tgt_mask(tgt, with_pad=True):
    device = tgt.device
    tgt_seq_len = tgt.shape[0]

    # sequence mask
    tgt_mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len)).bool().to(device)
    # padding mask
    if with_pad:
        tgt_padding_mask = (tgt == PAD_IDX)
    else:
        tgt_padding_mask = (tgt == EOS_IDX)
        index = (tgt_padding_mask == False).int().sum(dim=0, keepdim=False)
        for i in range(tgt.shape[1]):
            if index[i] < tgt.shape[0]:
                tgt_padding_mask[index[i], i] = 0
            else:
                pass
    # unsqueeze(-1) to make the tgt_mask shape as (Seq, Seq, 1) for decoder_mask, and
    # the shape as (Seq, B, 1) for decoder_padding_mask
    return tgt_mask.unsqueeze(-1), tgt_padding_mask.unsqueeze(-1)

# def create_tgt_mask(tgt, with_pad=False):
#     device = tgt.device
#     tgt_seq_len = tgt.shape[0]
#
#     # print(PAD_IDX)
#     tgt_mask = generate_square_subsequent_mask(tgt_seq_len,device)
#     if with_pad:
#         tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1).contiguous()
#     else:
#         tgt_padding_mask = (tgt == EOS_IDX)
#         index = (tgt_padding_mask == False).int().sum(dim=0, keepdim=False)
#         for i in range(tgt.shape[1]):
#             if index[i] < tgt.shape[0]:
#                 tgt_padding_mask[index[i], i] = 0
#             else:
#                 pass
#         tgt_padding_mask = tgt_padding_mask.transpose(0, 1).contiguous()
#
#     return tgt_mask, tgt_padding_mask

if __name__=='__main__':
    # tgt = torch.ones(2,5)
    # tgt[0, 3:] = 9
    # tgt[1, 4:] = 9
    # tgt = tgt.transpose(0, 1)
    # print(tgt)
    # print(create_tgt_mask(tgt)[1])

    model = Seq2SeqMLPConvTransformer(frequency_dim=64, hidden_dim=256, emb_size=128, dim_feedforward=512, dropout=.2,
                 num_decoder_layers=3, spec_aug=False, nb_classes=4368)
    #
    print('Total amount of parameters: ',
                      f'{sum([i.numel() for i in model.encoder.parameters()])}')
    x = torch.ones(2, 2548, 64)
    y = model(x)
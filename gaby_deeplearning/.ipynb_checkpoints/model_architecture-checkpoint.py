## Gabriel A. Santiago Plaza

##inpired by Attention is all you need paper
## book: Dive into Deep Learning
## Chord Conditioned LSTM
## Hierarchical multi-heaad attention LSTM for polyphonic symbolic melody generation
## MMT-Bert chord aware symbolic music generation


import torch
import torch.nn as nn
import math

## positional encoding, is it worth?


## Libro encoding,
class PositionalEncoding(nn.Module):  # @save
    """Positional encoding."""

    def __init__(self, emb_dimension, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = torch.zeros((1, max_len, emb_dimension))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000,
            torch.arange(0, emb_dimension, 2, dtype=torch.float32) / emb_dimension,
        )


        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        print('X')	
        print(X.shape)
        print('P')
        print(self.P.shape)
        X = X + self.P[:, : X.shape[1], :].to(X.device)
        return self.dropout(X)


## other encoder, lowkey the same??


class SinusoidalPosEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


### MOdel Definition

## this block will be reused for all the attention blocks,
## the cross attention, the self attention and the masked attention


## ATTENTION BLOCK, ATTENTION MECHANISM,
# it has multihead attention and will be used for encoder and decoder.
class AttentionBlock(nn.Module):
    ## hidden size sujetivo a paper
    ## dropout sujetivo a paper
    ## num heads sujetivo a paper
    def __init__(self, hidden_size=128, num_heads=4, dropout=0.2, masking=True):
        super(AttentionBlock, self).__init__()
        self.masking = masking

        ## Multihead??????
        self.multihead_attention = nn.MultiheadAttention(
            hidden_size, num_heads=num_heads, batch_first=True, dropout=dropout
        )

    ## Para el encoder como queremos self attention, x_in = key_value_in
    ## for cross attentnion x_in output del decoder, y kv_in del encoder,
    ## eso significa que para lo que el output quiere, utilizaré la  informacion del encoder

    ## key_mask es para el padding if necessary
    ## mask
    def forward(self, x_in, key_value_in, key_mask=None):
        ## x_in is input for query, key_value_in is input for key and value

        ##if masking is enabled, do causal masking

        print('key_padding_mask')
        if(key_mask is not None):
            print(key_mask.shape)
        if self.masking:
            # print(x_in.shape)
            batch_size, l, h = x_in.shape

            mask = torch.triu(torch.ones(l, l, device=x_in.device), 1).bool()
            print('mask')
            
            print(mask.shape)

        else:
            mask = None
            ###
        return self.multihead_attention(
            x_in, key_value_in, key_value_in, attn_mask=mask, key_padding_mask=key_mask
        )[0]


### Transformer Block
class TransformerBlock(nn.Module):
    ## hid paper
    ## num_heads paper
    ## dropout paper
    def __init__(
        self, hidden_size=128, num_heads=4, is_decoder=False, masking=True, dropout=0.2
    ):
        super(TransformerBlock, self).__init__()
        self.is_decoder = is_decoder

        ## layer norm
        self.norm1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)

        ## self attention for both is important, encoder will self attend, decoder
        ## will self attend only previous and cross attend  everything
        self.attn1 = AttentionBlock(
            hidden_size=hidden_size, num_heads=num_heads, masking=masking
        )

        # layer normalization for the output of first attention layer
        if self.is_decoder:
            self.norm2 = nn.LayerNorm(hidden_size)
            self.dropout2 = nn.Dropout(dropout)

            ## Self attention for decoder without masking (?)
            self.cross_attention = AttentionBlock(
                hidden_size=hidden_size, num_heads=num_heads, masking=False
            )

        ## at the end, both will have a feed foward at the end, mlp.

        self.norm_mlp = nn.LayerNorm(hidden_size)
        self.dropout_mlp = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    ## input_key is key padding for this block
    ## kv, is the keyvalue inputs
    ## cross key mask is masking when there is a cross attention.

    def forward(self, x, input_key=None, cross_key_mask=None, kv_cross=None):

        ##first, self attention plus additionlike attention all you need
        ## then norm and dropout like choir.
        ## input key is if we
        x = self.attn1(x, x, key_mask=input_key) + x
        x = self.norm1(x)
        x = self.dropout1(x)

        if self.is_decoder:
            ## this case we cross attend,
            x = self.cross_attention(x, kv_cross, key_mask=cross_key_mask) + x
            x = self.norm2(x)
            x = self.dropout2(x)

        ## mlp
        x = self.mlp(x) + x

        ## lets see logic.... Al final ellos tenían un
        ## something something un linear regressor al final y softmax.
        x = self.norm_mlp(x)
        x = self.dropout_mlp(x)

        return x

class MusicalEmbeddings(nn.Module):
    def __init__(self,num_dif_tokens, num_features_per_token, hidden_size=128):
        super(MusicalEmbeddings, self).__init__()
        self.feature_extractor_embeddings = nn.Embedding(num_dif_tokens, hidden_size)

        self.linear_reg_features = nn.Linear(in_features=num_features_per_token, out_features=hidden_size)

        self.concat_layer = nn.Linear(in_features=hidden_size*2, out_features=hidden_size)

    def forward(self, input_sequence):


        #extrating feature_arr 
        print(input_sequence.shape)
        num_dif_tokens_arr = input_sequence[:,:,0].int()
        print(num_dif_tokens_arr.unique())
        feature_arr = input_sequence[:,:,1:7].float()
        print(feature_arr.shape)
        # print(num_dif_tokens_arr.type())
        # print(feature_arr.type())
        feature_emb = self.linear_reg_features(feature_arr)
        token_emb = self.feature_extractor_embeddings(num_dif_tokens_arr)


        concat_emb = torch.cat([feature_emb,token_emb],dim=-1)

        merged_emb = self.concat_layer(concat_emb)

        return merged_emb




## ENcoder
class Encoder(nn.Module):
    def __init__(
        self,
        number_of_tokens,
        feature_size,
        hidden_size=128,
        num_layers=3,
        num_heads=4,
        encoding_mech="sinusoidal",
    ):
        super(Encoder, self).__init__()
        self.num_layers = num_layers

        ## Creating an embedding layer for token embeddings,
        # num_embeddings (int) – size of the dictionary of embeddings, or how big will the vocabulary be
        ## hidden_size (int) – the size of each embedding vector

        self.musical_embedding = MusicalEmbeddings(num_dif_tokens=number_of_tokens, num_features_per_token=feature_size, hidden_size=hidden_size)

        ## position embedding there are papers that say there is really no difference.
        if encoding_mech == "sinusoidal":
            self.pos_embedding = SinusoidalPosEmbedding(hidden_size)
        elif "libro":
            self.pos_embedding = PositionalEncoding(hidden_size, 0.2)

        else:
            self.pos_embedding = None

        # self.pos_embedding = SinusoidalPosEmbedding(hidden_size)

        ## the encoder will not have masking during the self attention
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size, num_heads, is_decoder=False, masking=False
                )
                for _ in range(self.num_layers)
            ]
        )

        # self.layers = nn.ModuleList([TransformerBlock(hidden_size, num_heads) for _ in range(num_layers)])

    def forward(self, input_sequence, padding_mask=None):

        ##input sequence is going to be introducedas embeddings
        input_emb = self.musical_embedding(input_sequence)

        # print("first_foward", input_emb.shape)
        print(input_emb.shape)
        batch_size, seq_length,hidden = input_emb.shape

        if self.pos_embedding is not None and self.pos_embedding !='libro':
            ## Creating a seuence of integers from 0 to l
            ## What this does is that it ads the positional information to the embedding sequence,
            ## by creating a positional embedding that we ADD to the input embedding.
            ## the input is inside a sinusoidal space or any other space.

            if(self.pos_embedding == 'libro' ):
                pass
                sequence_idx = torch.arange(seq_length, device=input_sequence.device).expand(batch_size,seq_length,hidden).float()
                pos_emb = (
                self.pos_embedding(sequence_idx)
                
            )
            else:
                sequence_idx  = torch.arange(seq_length,  device=input_sequence.device).to(torch.float32)
            
                pos_emb = (
                    self.pos_embedding(sequence_idx)
                    .reshape(1, seq_length, hidden)
                    .expand(batch_size, seq_length, hidden)
                )

                
            

                input_emb = input_emb + pos_emb

        for block in self.blocks:
            input_emb = block(input_emb, input_key=padding_mask)

        return input_emb


class Decoder(nn.Module):
    def __init__(
        self,
        number_of_tokens,
        feature_size,
        hidden_size=128,
        num_layers=3,
        num_heads=4,
        positional_encoding="sinusoidal",
    ):
        super(Decoder, self).__init__()

        ## num_emb is vocabulary size.
        self.musical_embedding = MusicalEmbeddings(num_dif_tokens=number_of_tokens, num_features_per_token=feature_size, hidden_size=hidden_size)

        if positional_encoding == "sinusoidal":
            self.pos_embedding = SinusoidalPosEmbedding(hidden_size)
        elif positional_encoding == "libro":
            self.pos_embedding = PositionalEncoding(hidden_size, 0.2)
        else:
            self.pos_embedding = None
        # self.pos_embedding = SinusoidalPosEmbedding(hidden_size)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, num_heads, is_decoder=True, masking=True)
                for _ in range(num_layers)
            ]
        )

        ## should it be a fully connected layer for output?
        self.fc_out = nn.Linear(hidden_size, number_of_tokens)

    def forward(
        self,
        input_sequence,
        encoder_output,
        input_padding_mask=None,
        encoder_padding_mask=None,
    ):
        input_embeddings = self.musical_embedding(input_sequence)

        batch_size, seq_length, h = input_embeddings.shape

        ## ADD positional information to embeddings.
        if self.pos_embedding is not None and self.pos_embedding !='libro':
            if(self.pos_embedding == 'libro'):
                pass
                sequence_idx = torch.arange(seq_length, device=input_sequence.device).expand(batch_size,seq_length,h).float()
                pos_emb = (
                    self.pos_embedding(sequence_idx)
                    # .reshape(1, seq_length, h)
                    # .expand(batch_size, seq_length, h)
                )
            else:

                sequence_idx = torch.arange(seq_length, device=input_sequence.device).to(torch.float32)
            
            
                pos_emb = (
                    self.pos_embedding(sequence_idx)
                    .reshape(1, seq_length, h)
                    .expand(batch_size, seq_length, h)
                )
            
                input_embeddings = input_embeddings + pos_emb

        for block in self.blocks:
            input_embeddings = block(
                input_embeddings,
                input_key=input_padding_mask,
                cross_key_mask=encoder_padding_mask,
                kv_cross=encoder_output,
            )

        return self.fc_out(input_embeddings)


class EncoderDecoderv1(nn.Module):
    def __init__(
        self,
        number_of_tokens_input,
        number_of_tokens_output,
        feature_size,
        hidden_size=128,
        num_layers=(3, 3),
        num_heads=4,
        positional_encoding="sinusoidal",
    ):
        super(EncoderDecoderv1, self).__init__()

        self.encoder = Encoder(
           number_of_tokens_input,feature_size, hidden_size, num_layers[0], num_heads, positional_encoding
        )
        self.decoder = Decoder(
            number_of_tokens_output,feature_size, hidden_size, num_layers[1], num_heads, positional_encoding
        )

    def forward(self, source_sequence, target_sequence):
        ## padding mask for encoder

        ## first check if the first elements are 6s

        source_seq_first = source_sequence[:,:,0]
        target_seq_first = target_sequence[:,:,0]

        input_key_mask = source_seq_first == 6
        output_key_mask = target_seq_first == 6

        print('input_key_mask')
        print(input_key_mask.shape)
        ## Encode the Input Sequence

        encoded_sequence = self.encoder(source_sequence, padding_mask=input_key_mask)

        ## pass to decoder
        decoded_seq = self.decoder(
            input_sequence=target_sequence,
            encoder_output=encoded_sequence,
            input_padding_mask=output_key_mask,
            encoder_padding_mask=input_key_mask,
        )

        return decoded_seq


class EncoderDecoderv2(nn.Module):
    def __init__(
        self,
        num_embeddings,
        hidden_size=128,
        num_layers=(3, 3),
        num_heads=4,
        positional_encoding="sinusoidal",
    ):
        super(EncoderDecoderv2, self).__init__()

        self.soprano_encoder = Encoder(
            num_embeddings, hidden_size, num_layers[0], num_heads, positional_encoding
        )
        self.chord_encoder = Encoder(
            num_embeddings, hidden_size, num_layers[0], num_heads, positional_encoding
        )

        self.alto_decoder = Decoder(
            num_embeddings, hidden_size, num_layers[1], num_heads, positional_encoding
        )
        self.tenor_decoder = Decoder(
            num_embeddings, hidden_size, num_layers[1], num_heads, positional_encoding
        )
        self.bass_decoder = Decoder(
            num_embeddings, hidden_size, num_layers[1], num_heads, positional_encoding
        )

        def forward(
            self,
            soprano_sequence,
            chord_sequence,
            alto_sequence,
            tenor_sequence,
            bass_sequence,
        ):
            ## padding mask for encoder
            soprano_key_mask = soprano_sequence == 0
            chord_key_mask = chord_sequence == 0
            alto_key_mask = alto_sequence == 0
            tenor_key_mask = tenor_sequence == 0
            bass_key_mask = bass_sequence == 0

            ## Encode the Input Sequence

            encoded_soprano_emb = self.soprano_encoder(
                soprano_sequence, padding_mask=soprano_key_mask
            )
            encoded_chord_emb = self.chord_encoder(
                chord_sequence, padding_mask=chord_key_mask
            )

            added_emb = encoded_soprano_emb + encoded_chord_emb

            # encoded_alto = self.chord_encoder(alto_sequence, padding_mask=alto_key_mask)
            # encoded_tenor = self.chord_encoder(tenor_sequence, padding_mask=tenor_key_mask)
            # encoded_bass = self.chord_encoder(bass_sequence, padding_mask=bass_key_mask)

            ## pass to decoder
            decoded_alto = self.alto_decoder(
                input_sequence=alto_sequence,
                encoder_output=added_emb,
                input_padding_mask=alto_key_mask,
                encoder_padding_mask=soprano_key_mask,
            )

            decoded_tenor = self.tenor_decoder(
                input_sequence=tenor_sequence,
                encoder_output=added_emb,
                input_padding_mask=tenor_key_mask,
                encoder_padding_mask=soprano_key_mask,
            )

            decoded_bass = self.bass_decoder(
                input_sequence=bass_sequence,
                encoder_output=added_emb,
                input_padding_mask=bass_key_mask,
                encoder_padding_mask=soprano_key_mask,
            )

            return decoded_alto, decoded_tenor, decoded_bass

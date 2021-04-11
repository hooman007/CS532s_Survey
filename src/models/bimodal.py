import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_CNN_LSTM_bimodal(args):
    return CNN_LSTM(args)


def get_CNN_self_attention_LSTM(args):
    return CNN_self_attention_LSTM(args)


def get_CNN_AttentionLSTM(args):
    return CNN_AttentionLSTM(args)


def get_CNN_self_attention_AttentionLSTM(args):
    return CNN_self_attention_AttentionLSTM(args)


def get_CNN_transformer(args):
    return CNN_transformer(args)


def get_CNN_self_attention_transformer(args):
    return CNN_self_attention_transformer(args)


class CNN_LSTM(nn.Module):
    """
    An audio-visual speech transcription model based on the Transformer architecture.
    Architecture:
    Character Set: 26 alphabets (A-Z), 10 numbers (0-9), apostrophe ('), space ( ), blank (-), end-of-sequence (<EOS>)
    Audio Input: 321-dim STFT feature vectors with 100 vectors per second. Each group of 4 consecutive feature vectors
                 is linearly transformeinto a single 512-dim feature vector giving 25 vectors per second.
    Video Input: 512-dim feature vector corresponding to each video frame giving 25 vectors per second.
    Output: Log probabilities over the character set at each time step.
    """

    def __init__(self, args):
        super(CNN_LSTM, self).__init__()
        self.audioConv = nn.Conv1d(321, args.d_model, kernel_size=4, stride=4, padding=0)
        self.jointConv = nn.Conv1d(2 * args.d_model, args.d_model, kernel_size=1, stride=1, padding=0)
        self.jointDecoder = nn.LSTM(input_size=args.d_model, hidden_size=args.d_model,
                                    num_layers=args.num_layers, dropout=args.dropout)
        self.outputConv = nn.Conv1d(args.d_model, args.num_classes, kernel_size=1, stride=1, padding=0)
        return

    def forward(self, inputBatch):
        audioInputBatch, videoInputBatch = inputBatch

        if audioInputBatch is not None:
            audioInputBatch = audioInputBatch.transpose(0, 1).transpose(1, 2)
            audioBatch = self.audioConv(audioInputBatch)
            audioBatch = audioBatch.transpose(1, 2).transpose(0, 1)
        else:
            audioBatch = None

        videoBatch = videoInputBatch

        if (audioBatch is not None) and (videoBatch is not None):
            jointBatch = torch.cat([audioBatch, videoBatch], dim=2)
            jointBatch = jointBatch.transpose(0, 1).transpose(1, 2)
            jointBatch = self.jointConv(jointBatch)
            jointBatch = jointBatch.transpose(1, 2).transpose(0, 1)
        elif (audioBatch is None) and (videoBatch is not None):
            jointBatch = videoBatch
        elif (audioBatch is not None) and (videoBatch is None):
            jointBatch = audioBatch
        else:
            print("Both audio and visual inputs missing.")
            exit()

        jointBatch, _ = self.jointDecoder(jointBatch)
        jointBatch = jointBatch.transpose(0, 1).transpose(1, 2)
        jointBatch = self.outputConv(jointBatch)
        jointBatch = jointBatch.transpose(1, 2).transpose(0, 1)
        outputBatch = F.log_softmax(jointBatch, dim=2)
        return outputBatch


class CNN_transformer(nn.Module):
    """
    An audio-visual speech transcription model based on the Transformer architecture.
    Architecture:
    Character Set: 26 alphabets (A-Z), 10 numbers (0-9), apostrophe ('), space ( ), blank (-), end-of-sequence (<EOS>)
    Audio Input: 321-dim STFT feature vectors with 100 vectors per second. Each group of 4 consecutive feature vectors
                 is linearly transformed into a single 512-dim feature vector giving 25 vectors per second.
    Video Input: 512-dim feature vector corresponding to each video frame giving 25 vectors per second.
    Output: Log probabilities over the character set at each time step.
    """

    def __init__(self, args):
        super(CNN_transformer, self).__init__()
        self.audioConv = nn.Conv1d(321, args.d_model, kernel_size=4, stride=4, padding=0)
        self.positionalEncoding = PositionalEncoding(dModel=args.d_model, maxLen=args.peMaxLen)
        self.jointConv = nn.Conv1d(2 * args.d_model, args.d_model, kernel_size=1, stride=1, padding=0)
        encoderLayer = nn.TransformerEncoderLayer(d_model=args.d_model, nhead=args.num_heads,
                                                  dim_feedforward=args.hidden_dim,
                                                  dropout=args.dropout)
        self.jointDecoder = nn.TransformerEncoder(encoderLayer, num_layers=args.num_layers)
        self.outputConv = nn.Conv1d(args.d_model, args.num_classes, kernel_size=1, stride=1, padding=0)
        return

    def forward(self, inputBatch):
        audioInputBatch, videoInputBatch = inputBatch

        if audioInputBatch is not None:  # only the fusion is applied
            audioInputBatch = audioInputBatch.transpose(0, 1).transpose(1, 2)
            audioBatch = self.audioConv(audioInputBatch)
            audioBatch = audioBatch.transpose(1, 2).transpose(0, 1)
        else:
            audioBatch = None

        videoBatch = videoInputBatch

        if (audioBatch is not None) and (videoBatch is not None):
            jointBatch = torch.cat([audioBatch, videoBatch], dim=2)
            jointBatch = jointBatch.transpose(0, 1).transpose(1, 2)
            jointBatch = self.jointConv(jointBatch)
            jointBatch = jointBatch.transpose(1, 2).transpose(0, 1)
        elif (audioBatch is None) and (videoBatch is not None):
            jointBatch = videoBatch
        elif (audioBatch is not None) and (videoBatch is None):
            jointBatch = audioBatch
        else:
            print("Both audio and visual inputs missing.")
            exit()

        jointBatch = self.jointDecoder(jointBatch)
        jointBatch = jointBatch.transpose(0, 1).transpose(1, 2)
        jointBatch = self.outputConv(jointBatch)
        jointBatch = jointBatch.transpose(1, 2).transpose(0, 1)
        outputBatch = F.log_softmax(jointBatch, dim=2)
        return outputBatch


class CNN_self_attention_LSTM(nn.Module):
    """
    An audio-visual speech transcription model based on the Transformer architecture.
    Architecture: Two stacks of 6 Transformer encoder layers form the Encoder (one for each modality),
                  A single stack of 6 Transformer encoder layers form the joint Decoder. The encoded feature vectors
                  from both the modalities are concatenated and linearly transformed into 512-dim vectors.
    Character Set: 26 alphabets (A-Z), 10 numbers (0-9), apostrophe ('), space ( ), blank (-), end-of-sequence (<EOS>)
    Audio Input: 321-dim STFT feature vectors with 100 vectors per second. Each group of 4 consecutive feature vectors
                 is linearly transformed into a single 512-dim feature vector giving 25 vectors per second.
    Video Input: 512-dim feature vector corresponding to each video frame giving 25 vectors per second.
    Output: Log probabilities over the character set at each time step.
    """

    def __init__(self, args):
        super(CNN_self_attention_LSTM, self).__init__()
        self.audioConv = nn.Conv1d(321, args.d_model, kernel_size=4, stride=4, padding=0)
        self.positionalEncoding = PositionalEncoding(dModel=args.d_model, maxLen=args.peMaxLen)
        encoderLayer = nn.TransformerEncoderLayer(d_model=args.d_model, nhead=args.num_heads,
                                                  dim_feedforward=args.hidden_dim,
                                                  dropout=args.dropout)
        self.audioEncoder = nn.TransformerEncoder(encoderLayer, num_layers=args.num_layers)
        self.videoEncoder = nn.TransformerEncoder(encoderLayer, num_layers=args.num_layers)
        self.jointConv = nn.Conv1d(2 * args.d_model, args.d_model, kernel_size=1, stride=1, padding=0)
        self.jointDecoder = nn.LSTM(input_size=args.d_model, hidden_size=args.d_model,
                                    num_layers=args.num_layers, dropout=args.dropout)
        self.outputConv = nn.Conv1d(args.d_model, args.num_classes, kernel_size=1, stride=1, padding=0)
        return

    def forward(self, inputBatch):
        audioInputBatch, videoInputBatch = inputBatch

        if audioInputBatch is not None:
            audioInputBatch = audioInputBatch.transpose(0, 1).transpose(1, 2)
            audioBatch = self.audioConv(audioInputBatch)
            audioBatch = audioBatch.transpose(1, 2).transpose(0, 1)
            audioBatch = self.positionalEncoding(audioBatch)
            audioBatch = self.audioEncoder(audioBatch)
        else:
            audioBatch = None

        if videoInputBatch is not None:
            videoBatch = self.positionalEncoding(videoInputBatch)
            videoBatch = self.videoEncoder(videoBatch)
        else:
            videoBatch = None

        if (audioBatch is not None) and (videoBatch is not None):
            jointBatch = torch.cat([audioBatch, videoBatch], dim=2)
            jointBatch = jointBatch.transpose(0, 1).transpose(1, 2)
            jointBatch = self.jointConv(jointBatch)
            jointBatch = jointBatch.transpose(1, 2).transpose(0, 1)
        elif (audioBatch is None) and (videoBatch is not None):
            jointBatch = videoBatch
        elif (audioBatch is not None) and (videoBatch is None):
            jointBatch = audioBatch
        else:
            print("Both audio and visual inputs missing.")
            exit()

        jointBatch, _ = self.jointDecoder(jointBatch)
        jointBatch = jointBatch.transpose(0, 1).transpose(1, 2)
        jointBatch = self.outputConv(jointBatch)
        jointBatch = jointBatch.transpose(1, 2).transpose(0, 1)
        outputBatch = F.log_softmax(jointBatch, dim=2)
        return outputBatch


class CNN_self_attention_transformer(nn.Module):
    """
    An audio-visual speech transcription model based on the Transformer architecture.
    Architecture: Two stacks of 6 Transformer encoder layers form the Encoder (one for each modality),
                  A single stack of 6 Transformer encoder layers form the joint Decoder. The encoded feature vectors
                  from both the modalities are concatenated and linearly transformed into 512-dim vectors.
    Character Set: 26 alphabets (A-Z), 10 numbers (0-9), apostrophe ('), space ( ), blank (-), end-of-sequence (<EOS>)
    Audio Input: 321-dim STFT feature vectors with 100 vectors per second. Each group of 4 consecutive feature vectors
                 is linearly transformed into a single 512-dim feature vector giving 25 vectors per second.
    Video Input: 512-dim feature vector corresponding to each video frame giving 25 vectors per second.
    Output: Log probabilities over the character set at each time step.
    """

    def __init__(self, args):
        super(CNN_self_attention_transformer, self).__init__()
        self.audioConv = nn.Conv1d(321, args.d_model, kernel_size=4, stride=4, padding=0)
        self.positionalEncoding = PositionalEncoding(dModel=args.d_model, maxLen=args.peMaxLen)
        encoderLayer = nn.TransformerEncoderLayer(d_model=args.d_model, nhead=args.num_heads,
                                                  dim_feedforward=args.hidden_dim,
                                                  dropout=args.dropout)
        self.audioEncoder = nn.TransformerEncoder(encoderLayer, num_layers=args.num_layers)
        self.videoEncoder = nn.TransformerEncoder(encoderLayer, num_layers=args.num_layers)
        self.jointConv = nn.Conv1d(2 * args.d_model, args.d_model, kernel_size=1, stride=1, padding=0)
        self.jointDecoder = nn.TransformerEncoder(encoderLayer, num_layers=args.num_layers)
        self.outputConv = nn.Conv1d(args.d_model, args.num_classes, kernel_size=1, stride=1, padding=0)
        return

    def forward(self, inputBatch):
        audioInputBatch, videoInputBatch = inputBatch

        if audioInputBatch is not None:
            audioInputBatch = audioInputBatch.transpose(0, 1).transpose(1, 2)
            audioBatch = self.audioConv(audioInputBatch)
            audioBatch = audioBatch.transpose(1, 2).transpose(0, 1)
            audioBatch = self.positionalEncoding(audioBatch)
            audioBatch = self.audioEncoder(audioBatch)
        else:
            audioBatch = None

        if videoInputBatch is not None:
            videoBatch = self.positionalEncoding(videoInputBatch)
            videoBatch = self.videoEncoder(videoBatch)
        else:
            videoBatch = None

        if (audioBatch is not None) and (videoBatch is not None):
            jointBatch = torch.cat([audioBatch, videoBatch], dim=2)
            jointBatch = jointBatch.transpose(0, 1).transpose(1, 2)
            jointBatch = self.jointConv(jointBatch)
            jointBatch = jointBatch.transpose(1, 2).transpose(0, 1)
        elif (audioBatch is None) and (videoBatch is not None):
            jointBatch = videoBatch
        elif (audioBatch is not None) and (videoBatch is None):
            jointBatch = audioBatch
        else:
            print("Both audio and visual inputs missing.")
            exit()

        jointBatch = self.jointDecoder(jointBatch)
        jointBatch = jointBatch.transpose(0, 1).transpose(1, 2)
        jointBatch = self.outputConv(jointBatch)
        jointBatch = jointBatch.transpose(1, 2).transpose(0, 1)
        outputBatch = F.log_softmax(jointBatch, dim=2)
        return outputBatch


class PositionalEncoding(nn.Module):
    """
    A layer to add positional encodings to the inputs of a Transformer model.
    Formula:
    PE(pos,2i) = sin(pos/10000^(2i/d_model))
    PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
    """

    def __init__(self, dModel, maxLen):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(maxLen, dModel)
        position = torch.arange(0, maxLen, dtype=torch.float).unsqueeze(dim=-1)
        denominator = torch.exp(torch.arange(0, dModel, 2).float() * (math.log(10000.0) / dModel))
        pe[:, 0::2] = torch.sin(position / denominator)
        pe[:, 1::2] = torch.cos(position / denominator)
        pe = pe.unsqueeze(dim=0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, inputBatch):
        outputBatch = inputBatch + self.pe[:inputBatch.shape[0], :, :]
        return outputBatch


class Attention(nn.Module):
    '''
    from https://arxiv.org/pdf/1409.0473.pdf
    Bahdanau attention
    https://machinelearningmastery.com/how-does-attention-work-in-encoder-decoder-recurrent-neural-networks/
    simple image
    https://images2015.cnblogs.com/blog/670089/201610/670089-20161012111504671-910168246.png
    '''

    def __init__(self, hidden_size, annotation_size):
        super(Attention, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(hidden_size + annotation_size, hidden_size),
            nn.Relu(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, prev_hidden_state, annotations):
        '''
        1. expand prev_hidden_state dimension and transpose
            prev_hidden_state : (batch_size, sequence_length, feature dimension(512))

        2. concatenate
            concatenated : size (batch_size, sequence_length, encoder_hidden_size + hidden_size)

        3. dense and squeeze
            energy : size (batch_size, sequence_length)

        4. softmax to compute weight alpha
            alpha : (batch_size, 1, sequence_length)

        5. weighting annotations
            context : (batch_size, 1, encoder_hidden_size(256))

        Parameters
        ----------
        prev_hidden_state : 3-D torch Tensor
            (batch_size, 1, hidden_size(default 512))

        annotations : 3-D torch Tensor
            (batch_size, sequence_length, encoder_hidden_size(256))

        Returns
        -------
        context : 3-D torch Tensor
            (batch_size, 1, encoder_hidden_size(256))
        '''
        batch_size, sequence_length, _ = annotations.size()

        prev_hidden_state = prev_hidden_state.repeat(sequence_length, 1, 1).transpose(0, 1)

        concatenated = torch.cat([prev_hidden_state, annotations], dim=2)
        attn_energies = self.dense(concatenated).squeeze(2)
        alpha = F.softmax(attn_energies).unsqueeze(1)
        context = alpha.bmm(annotations)

        return context


class CNN_self_attention_AttentionLSTM(nn.Module):
    """
    An audio-visual speech transcription model based on the Transformer architecture.
    Architecture: Two stacks of 6 Transformer encoder layers form the Encoder (one for each modality),
                  A single stack of 6 Transformer encoder layers form the joint Decoder. The encoded feature vectors
                  from both the modalities are concatenated and linearly transformed into 512-dim vectors.
    Character Set: 26 alphabets (A-Z), 10 numbers (0-9), apostrophe ('), space ( ), blank (-), end-of-sequence (<EOS>)
    Audio Input: 321-dim STFT feature vectors with 100 vectors per second. Each group of 4 consecutive feature vectors
                 is linearly transformed into a single 512-dim feature vector giving 25 vectors per second.
    Video Input: 512-dim feature vector corresponding to each video frame giving 25 vectors per second.
    Output: Log probabilities over the character set at each time step.
    """

    def __init__(self, args):
        super(CNN_self_attention_AttentionLSTM, self).__init__()
        self.audioConv = nn.Conv1d(321, args.d_model, kernel_size=4, stride=4, padding=0)
        self.positionalEncoding = PositionalEncoding(dModel=args.d_model, maxLen=args.peMaxLen)
        encoderLayer = nn.TransformerEncoderLayer(d_model=args.d_model, nhead=args.num_heads,
                                                  dim_feedforward=args.hidden_dim,
                                                  dropout=args.dropout)
        self.audioEncoder = nn.TransformerEncoder(encoderLayer, num_layers=args.num_layers)
        self.videoEncoder = nn.TransformerEncoder(encoderLayer, num_layers=args.num_layers)
        self.jointConv = nn.Conv1d(2 * args.d_model, args.d_model, kernel_size=1, stride=1, padding=0)
        self.jointDecoder = nn.LSTM(input_size=args.d_model, hidden_size=args.d_model,
                                    num_layers=args.num_layers, dropout=args.dropout)
        self.outputConv = nn.Conv1d(args.d_model, args.num_classes, kernel_size=1, stride=1, padding=0)
        self.attention = Attention(args.hidden_dim, args.hidden_dim)
        self.contextFusion = nn.Linear(2*args.hidden_dim, args.hidden_dim)
        self.outMLP = nn.Linear(2*args.hidden_dim, args.num_classes)
        return

    def forward(self, inputBatch):
        audioInputBatch, videoInputBatch = inputBatch

        if audioInputBatch is not None:
            audioInputBatch = audioInputBatch.transpose(0, 1).transpose(1, 2)
            audioBatch = self.audioConv(audioInputBatch)
            audioBatch = audioBatch.transpose(1, 2).transpose(0, 1)
            audioBatch = self.positionalEncoding(audioBatch)
            audioBatch = self.audioEncoder(audioBatch)
        else:
            audioBatch = None

        if videoInputBatch is not None:
            videoBatch = self.positionalEncoding(videoInputBatch)
            videoBatch = self.videoEncoder(videoBatch)
        else:
            videoBatch = None

        if (audioBatch is not None) and (videoBatch is not None):
            jointBatch = torch.cat([audioBatch, videoBatch], dim=2)
            jointBatch = jointBatch.transpose(0, 1).transpose(1, 2)
            jointBatch = self.jointConv(jointBatch)
            jointBatch = jointBatch.transpose(1, 2).transpose(0, 1)
        elif (audioBatch is None) and (videoBatch is not None):
            jointBatch = videoBatch
        elif (audioBatch is not None) and (videoBatch is None):
            jointBatch = audioBatch
        else:
            print("Both audio and visual inputs missing.")
            exit()

        all_context = torch.zeros_like(jointBatch.shape) # S, B, dim
        all_out = torch.zeros_like(jointBatch.shape)
        # h = (num_layers, B, dim)
        for i in range(jointBatch.shape[1]):
            jointEmbed = jointBatch[:, i:i+1, :]
            if i == 0:
                decoderOut, h, c = self.jointDecoder(jointEmbed)
            else:
                decoderOut, h, c = self.jointDecoder(jointEmbed, (h, c))
            context = self.attention(h[-1].unsqueeze(1), jointBatch.transpose(0, 1))  # b, 1, dim
            all_context[:, i, :] = context.transpose(0, 1)
            all_out[:, i, :] = decoderOut
            h[0] = self.contextFusion(torch.cat([h[0], context[:, 0]], dim=1))

        stuff = torch.cat([all_context, all_out], dim=2)  # S, b, 2dim
        jointBatch = self.outMLP(stuff.transpose(0, 1))  # b, S, num_classes
        jointBatch = jointBatch.transpose(0, 1)  # s, b, num_classes
        outputBatch = F.log_softmax(jointBatch, dim=2)
        return outputBatch


class CNN_AttentionLSTM(nn.Module):
    """
    An audio-visual speech transcription model based on the Transformer architecture.
    Architecture:
    Character Set: 26 alphabets (A-Z), 10 numbers (0-9), apostrophe ('), space ( ), blank (-), end-of-sequence (<EOS>)
    Audio Input: 321-dim STFT feature vectors with 100 vectors per second. Each group of 4 consecutive feature vectors
                 is linearly transformed into a single 512-dim feature vector giving 25 vectors per second.
    Video Input: 512-dim feature vector corresponding to each video frame giving 25 vectors per second.
    Output: Log probabilities over the character set at each time step.
    """

    def __init__(self, args):
        super(CNN_AttentionLSTM, self).__init__()
        self.audioConv = nn.Conv1d(321, args.d_model, kernel_size=4, stride=4, padding=0)
        self.jointConv = nn.Conv1d(2 * args.d_model, args.d_model, kernel_size=1, stride=1, padding=0)
        self.jointDecoder = nn.LSTM(input_size=args.d_model, hidden_size=args.d_model,
                                    num_layers=args.num_layers, dropout=args.dropout)
        self.outputConv = nn.Conv1d(args.d_model, args.num_classes, kernel_size=1, stride=1, padding=0)
        self.attention = Attention(args.hidden_dim, args.hidden_dim)
        self.contextFusion = nn.Linear(2 * args.hidden_dim, args.hidden_dim)
        self.outMLP = nn.Linear(2 * args.hidden_dim, args.num_classes)
        return

    def forward(self, inputBatch):
        audioInputBatch, videoInputBatch = inputBatch

        if audioInputBatch is not None:
            audioInputBatch = audioInputBatch.transpose(0, 1).transpose(1, 2)
            audioBatch = self.audioConv(audioInputBatch)
            audioBatch = audioBatch.transpose(1, 2).transpose(0, 1)
        else:
            audioBatch = None

        videoBatch = videoInputBatch

        if (audioBatch is not None) and (videoBatch is not None):
            jointBatch = torch.cat([audioBatch, videoBatch], dim=2)
            jointBatch = jointBatch.transpose(0, 1).transpose(1, 2)
            jointBatch = self.jointConv(jointBatch)
            jointBatch = jointBatch.transpose(1, 2).transpose(0, 1)
        elif (audioBatch is None) and (videoBatch is not None):
            jointBatch = videoBatch
        elif (audioBatch is not None) and (videoBatch is None):
            jointBatch = audioBatch
        else:
            print("Both audio and visual inputs missing.")
            exit()

        all_context = torch.zeros_like(jointBatch.shape)  # S, B, dim
        all_out = torch.zeros_like(jointBatch.shape)
        # h = (num_layers, B, dim)
        for i in range(jointBatch.shape[1]):
            jointEmbed = jointBatch[:, i:i + 1, :]
            if i == 0:
                decoderOut, h, c = self.jointDecoder(jointEmbed)
            else:
                decoderOut, h, c = self.jointDecoder(jointEmbed, (h, c))
            context = self.attention(h[-1].unsqueeze(1), jointBatch.transpose(0, 1))  # b, 1, dim
            all_context[:, i, :] = context.transpose(0, 1)
            all_out[:, i, :] = decoderOut
            h[0] = self.contextFusion(torch.cat([h[0], context[:, 0]], dim=1))

        stuff = torch.cat([all_context, all_out], dim=2)  # S, b, 2dim
        jointBatch = self.outMLP(stuff.transpose(0, 1))  # b, S, num_classes
        jointBatch = jointBatch.transpose(0, 1)  # s, b, num_classes
        outputBatch = F.log_softmax(jointBatch, dim=2)
        return outputBatch
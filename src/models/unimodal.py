import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_FC_LSTM():
    pass

def get_CNN_LSTM(args):
    return Video_CNN_LSTM(args)


class Video_CNN_LSTM(nn.Module):
    """
    A video-only speech transcription model based on the Transformer architecture.
    Architecture: CNN on the video followed by an LSTM? the CNN has been applied before so we just add an LSTM
    Character Set: 26 alphabets (A-Z), 10 numbers (0-9), apostrophe ('), space ( ), blank (-), end-of-sequence (<EOS>)
    Input: 512-dim feature vector corresponding to each video frame giving 25 vectors per second.
    Output: Log probabilities over the character set at each time step.
    """

    def __init__(self, args):
        super(Video_CNN_LSTM, self).__init__()
        # self.videoEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        # self.videoDecoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        # self.outputConv = nn.Conv1d(dModel, numClasses, kernel_size=1, stride=1, padding=0)
        self.LSTM = nn.LSTM(input_size=512, hidden_size=args.d_model,
                            num_layers=args.num_layers, dropout=args.dropout)
        self.outputConv = nn.Conv1d(args.d_model, args.num_classes, kernel_size=1, stride=1, padding=0)

        return

    def forward(self, video_inputBatch):
        # input batch is (S, B, 512) video
        batch, _ = self.LSTM(video_inputBatch)  # batch (S, B, HiddenSize)
        batch = batch.transpose(0, 1).transpose(1, 2)
        batch = self.outputConv(batch)  # (S, B, num_classes)
        batch = batch.transpose(1, 2).transpose(0, 1)
        return F.log_softmax(batch, dim=2)

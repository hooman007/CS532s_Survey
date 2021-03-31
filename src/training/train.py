from src.models import BIMODAL_MODEL_FACTORY, UNIMODAL_MODEL_FACTORY
from src.data import DATA_FACTORY
from src.utils import LOSS_FACTORY
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='CNN_transformer',
                        help='BIMODAL: CNN_LSTM, CNNSelfAttention_LSTM, CNN_AttentionLSTM, '
                             'CNNSelfAttention_AttentionLSTM, CNN_transformer, CNNSelfAttention_transformer,'
                             ' \n, UNIMODAL: FC_LSTM, CNN_LSTM')
    parser.add_argument('--modality', type=str, default='bimodal', help='unimodal, bimodal')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=60, help='number of epochs for training')
    parser.add_argument('--optimizer', type=str, default="adam", help='optimizer for training')
    parser.add_argument('--loss', type=str, default='seq2seq', help='seq2seq or CTC')
    parser.add_argument('--data', type=str, default='grid', help='grid or?')

    args = parser.parse_args()
    print(f"raw args = {args}")

    if args.modality == 'unimodal':
        model = UNIMODAL_MODEL_FACTORY[args.model]
    else:
        model = BIMODAL_MODEL_FACTORY[args.model]
    loss = LOSS_FACTORY[args.loss]
    data = DATA_FACTORY[args.data]

    train(model, data, loss)


def train(model, data, loss):
    pass



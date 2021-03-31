from src.utils.loss import get_CTC, get_seq2seq

LOSS_FACTORY = {
    'CTC': get_CTC,
    'seq2seq': get_seq2seq
}
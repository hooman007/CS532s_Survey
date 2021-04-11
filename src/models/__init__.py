from .bimodal import get_CNN_LSTM_bimodal, get_CNN_AttentionLSTM, get_CNN_self_attention_LSTM,\
    get_CNN_self_attention_AttentionLSTM, get_CNN_self_attention_transformer, get_CNN_transformer
from .unimodal import get_FC_LSTM, get_CNN_LSTM

UNIMODAL_MODEL_FACTORY = {
    'FC_LSTM': get_FC_LSTM,
    'CNN_LSTM': get_CNN_LSTM,
}

BIMODAL_MODEL_FACTORY = {
    'CNN_LSTM': get_CNN_LSTM,
    'CNNSelfAttention_LSTM': get_CNN_self_attention_LSTM,
    'CNN_AttentionLSTM': get_CNN_AttentionLSTM,
    'CNNSelfAttention_AttentionLSTM': get_CNN_self_attention_AttentionLSTM,
    'CNN_transformer': get_CNN_transformer,
    'CNNSelfAttention_transformer': get_CNN_self_attention_transformer,
}
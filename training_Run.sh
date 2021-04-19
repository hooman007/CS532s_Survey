#python src/training/train.py --modality bimodal --data LRS2 --batch_size 32 --epochs 400 --model CNNSelfAttention_transformer  --run_name AVSR_schedulerCER
#python src/training/train.py --modality bimodal --data LRS2 --batch_size 32 --epochs 400 --model CNN_transformer
#python src/training/train.py --modality bimodal --data LRS2 --batch_size 32 --epochs 400 --model CNNSelfAttention_AttentionLSTM
#python src/training/train.py --modality bimodal --data LRS2 --batch_size 32 --epochs 400 --model CNNSelfAttention_LSTM
#python src/training/train.py --modality bimodal --data LRS2 --batch_size 32 --epochs 400 --model CNN_AttentionLSTM
#python src/training/train.py --modality bimodal --data LRS2 --batch_size 32 --epochs 400 --model CNN_LSTM
python src/training/train.py --modality_opmode AO --data LRS2 --batch_size 32 --epochs 400 --model CNN_LSTM  --run_name CNN_LSTM_AudioOnly
python src/training/train.py --modality_opmode VO --data LRS2 --batch_size 32 --epochs 400 --model CNN_LSTM  --run_name CNN_LSTM_VideoOnly

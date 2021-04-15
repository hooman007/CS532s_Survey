python src/training/train.py --modality bimodal --data LRS2 --batch_size 32 --epochs 400 --model CNNSelfAttention_transformer  --run_name AVSR_schedulerCER
python src/training/train.py --modality bimodal --data LRS2 --batch_size 32 --epochs 600 --model CNN_transformer
python src/training/train.py --modality bimodal --data LRS2 --batch_size 32 --epochs 600 --model CNNSelfAttention_AttentionLSTM
python src/training/train.py --modality bimodal --data LRS2 --batch_size 32 --epochs 600 --model CNNSelfAttention_LSTM
python src/training/train.py --modality bimodal --data LRS2 --batch_size 32 --epochs 600 --model CNN_AttentionLSTM
python src/training/train.py --modality bimodal --data LRS2 --batch_size 32 --epochs 600 --model CNN_LSTM

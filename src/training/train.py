from src.models import BIMODAL_MODEL_FACTORY, UNIMODAL_MODEL_FACTORY
from src.data import DATA_FACTORY
from src.utils import LOSS_FACTORY
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.data.lrs2_utils import collate_fn
from src.data.lrs2_config import get_LRS2_Cfg
import matplotlib
from tqdm import tqdm
from src.utils.ctc_utils import ctc_greedy_decode, ctc_search_decode
from src.utils.metrics import compute_cer, compute_wer



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='CNN_LSTM',
                        help='BIMODAL: CNN_LSTM, CNNSelfAttention_LSTM, CNN_AttentionLSTM, '
                             'CNNSelfAttention_AttentionLSTM, CNN_transformer, CNNSelfAttention_transformer,'
                             ' \n, UNIMODAL: FC_LSTM, CNN_LSTM')
    parser.add_argument('--modality', type=str, default='unimodal', help='unimodal, bimodal')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=60, help='number of epochs for training')
    # parser.add_argument('--optimizer', type=str, default="adam", help='optimizer for training')
    parser.add_argument('--loss', type=str, default='seq2seq', help='seq2seq or CTC')
    parser.add_argument('--data', type=str, default='LRS2', help='grid or LRS2')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--num_workers', type=int, default=1, help='num_workers')
    parser.add_argument('--d_model', type=int, default=512, help='transformer feature size')
    parser.add_argument('--peMaxLen', type=int, default=2500, help='max len for positional encoding')
    parser.add_argument('--num_heads', type=int, default=8, help='num of attention heads')
    parser.add_argument('--hidden_dim', type=int, default=2048,
                        help='hidden dim size of FFNs and FC layers and LSTM even')
    parser.add_argument('--dropout', type=float, default=0.1, help='drpout rate')
    parser.add_argument('--num_layers', type=int, default=6, help='num of layers')
    parser.add_argument('--num_classes', type=int, default=40, help='num of classes!')
    parser.add_argument('--decode_scheme', type=str, default='greedy', help='CTC decoding scheme, greedy or search')
        # for lrs2 it's 40, no idea for gird and have to basically change this

    args = parser.parse_args()
    print(f"raw args = {args}")

    train(args)


def train(args):
    matplotlib.use("Agg")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    gpuAvailable = torch.cuda.is_available()
    device = torch.device("cuda" if gpuAvailable else "cpu")
    print(f"device:{device}")
    kwargs = {"num_workers": args.num_workers, "pin_memory": True} if gpuAvailable else {}
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    trainData = DATA_FACTORY[args.data + '_train']
    trainLoader = DataLoader(trainData, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, **kwargs)
    valData = DATA_FACTORY[args.data + '_val']
    valLoader = DataLoader(valData, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, **kwargs)

    if args.modality == 'unimodal':
        model = UNIMODAL_MODEL_FACTORY[args.model](args)
    else:
        model = BIMODAL_MODEL_FACTORY[args.model](args)
    model.to(device)
    # print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # they also had an scheduler that I skipped
    loss_function = nn.CTCLoss(blank=0, zero_infinity=False)

    trainingLossCurve = list()
    validationLossCurve = list()
    trainingWERCurve = list()
    validationWERCurve = list()

    for epoch in range(args.epochs):
        trainingLoss = 0
        trainingCER = 0
        trainingWER = 0

        for batch, (inputBatch, targetBatch, inputLenBatch, targetLenBatch) in enumerate(
                tqdm(trainLoader, leave=False, desc="Train", ncols=75)):

            inputBatch, targetBatch = ((inputBatch[0].float()).to(device), (inputBatch[1].float()).to(device)), (
                targetBatch.int()).to(device)
            inputLenBatch, targetLenBatch = (inputLenBatch.int()).to(device), (targetLenBatch.int()).to(device)

            opmode = np.random.choice(["AO", "VO", "AV"],
                                      p=[0.2, 0.2, 0.6])
            if opmode == "AO":
                inputBatch = (inputBatch[0], None)
            elif opmode == "VO":
                inputBatch = (None, inputBatch[1])
            else:
                pass

            if args.modality == 'unimodal':
                inputBatch = inputBatch[1]
                if inputBatch is None:
                    continue

            optimizer.zero_grad()
            model.train()
            outputBatch = model(inputBatch)
            with torch.backends.cudnn.flags(enabled=False):
                loss = loss_function(outputBatch, targetBatch, inputLenBatch, targetLenBatch)
            loss.backward()
            optimizer.step()

            trainingLoss = trainingLoss + loss.item()
            eosIx = 39 # index of EOS
            spaceIx = 1
            predictionBatch, predictionLenBatch = ctc_greedy_decode(outputBatch.detach(), inputLenBatch, eosIx)
            trainingCER = trainingCER + compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch)
            trainingWER = trainingWER + compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch,
                                                    spaceIx)

        trainingLoss = trainingLoss / len(trainLoader)
        trainingCER = trainingCER / len(trainLoader)
        trainingWER = trainingWER / len(trainLoader)
        trainingLossCurve.append(trainingLoss)
        trainingWERCurve.append(trainingWER)

        evalLoss = 0
        evalCER = 0
        evalWER = 0

        for batch, (inputBatch, targetBatch, inputLenBatch, targetLenBatch) in enumerate(
                tqdm(valLoader, leave=False, desc="Eval",
                     ncols=75)):

            inputBatch, targetBatch = ((inputBatch[0].float()).to(device), (inputBatch[1].float()).to(device)), (
                targetBatch.int()).to(device)
            inputLenBatch, targetLenBatch = (inputLenBatch.int()).to(device), (targetLenBatch.int()).to(device)


            opmode = np.random.choice(["AO", "VO", "AV"], p=[0, 0, 1])
            if opmode == "AO":
                inputBatch = (inputBatch[0], None)
            elif opmode == "VO":
                inputBatch = (None, inputBatch[1])
            else:
                pass

            if args.modality == 'unimodal': # video only
                inputBatch = inputBatch[1]
                if inputBatch is None:
                    continue

            model.eval()
            with torch.no_grad():
                outputBatch = model(inputBatch)
                with torch.backends.cudnn.flags(enabled=False):
                    loss = loss_function(outputBatch, targetBatch, inputLenBatch, targetLenBatch)

            evalLoss = evalLoss + loss.item()
            if args.decode_scheme == "greedy":
                predictionBatch, predictionLenBatch = ctc_greedy_decode(outputBatch, inputLenBatch, eosIx)
            # elif args.decode_scheme == "search":
            #     predictionBatch, predictionLenBatch = ctc_search_decode(outputBatch, inputLenBatch,
            #                                                             evalParams["beamSearchParams"],
            #                                                             evalParams["spaceIx"], evalParams["eosIx"],
            #                                                             evalParams["lm"])
            else:
                print("Invalid Decode Scheme")
                exit()

            evalCER = evalCER + compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch)
            evalWER = evalWER + compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch,
                                            spaceIx)

        validationLoss = evalLoss / len(valLoader)
        validationCER = evalCER / len(valLoader)
        validationWER = evalWER / len(valLoader)
        validationLossCurve.append(validationLoss)
        validationWERCurve.append(validationWER)
        # printing the stats after each step
        print("Step: %03d || Tr.Loss: %.6f  Val.Loss: %.6f || Tr.CER: %.3f  Val.CER: %.3f || Tr.WER: %.3f  Val.WER: %.3f"
            % (epoch, trainingLoss, validationLoss, trainingCER, validationCER, trainingWER, validationWER))


if __name__ == '__main__':
    main()
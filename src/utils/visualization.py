import torch
import wandb
import pandas as pd

INDEX_TO_CHAR = {1:" ", 22:"'", 30:"1", 29:"0", 37:"3", 32:"2", 34:"5", 38:"4", 36:"7", 35:"6", 31:"9", 33:"8",
		                          5:"A", 17:"C", 20:"B", 2:"E", 12:"D", 16:"G", 19:"F", 6:"I", 9:"H", 24:"K", 25:"J", 18:"M",
		                          11:"L", 4:"O", 7:"N", 27:"Q", 21:"P", 8:"S", 10:"R", 13:"U", 3:"T", 15:"W", 23:"V", 14:"Y",
		                          26:"X", 28:"Z", 39:"<EOS>"}    #index to character reverse mapping

def visualize_sentences(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch, mode, epochID):

    targetBatch = targetBatch.cpu()
    targetLenBatch = targetLenBatch.cpu()

    preds = list(torch.split(predictionBatch, predictionLenBatch.tolist()))
    trgts = list(torch.split(targetBatch, targetLenBatch.tolist()))

    pred_sentences = []
    trgt_sentences = []

    # table = wandb.Table(columns=["Target", "Prediction"])

    for n in range(5):
        pred = preds[n].numpy()[:-1]
        pred_chars = [INDEX_TO_CHAR[indx] for indx in pred]
        str = ""
        # pred_sentence = str.join(pred_chars)
        # pred_sentences.append(str.join(pred_chars))
        pred_sentences.append(pred_chars)

        trgt = trgts[n].numpy()[:-1]
        trgt_chars = [INDEX_TO_CHAR[indx] for indx in trgt]
        str = ""
        # trgt_sentence = str.join(trgt_chars)
        trgt_sentences.append(str.join(trgt_chars))

        # table.add_data(trgt_sentence, pred_sentence)

    print(f"predictionBatch is {predictionBatch}")
    transcription = {'Target': trgt_sentences, 'Prediction': pred_sentences}
    df = pd.DataFrame(data=transcription)
    print(df)
    # wandb.log({f"Visualization/{mode}_sentences": wandb.Table(dataframe=df)}, step=epochID)

    # transcription = [trgt_sentences, pred_sentences]
    # wandb.log({f"{mode}/transcribtion": wandb.Table(data=transcription, columns=["Target", "Prediction"])}, step=epochID)

    # wandb.log({f"{mode}/transcribtion": table})
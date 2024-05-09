import torch
from transformers import BertTokenizerFast, BertConfig, BertForTokenClassification
device = 'cuda'
labels_to_ids = {'O': 0, 'B-geo': 1, 'B-gpe': 2, 'B-per': 3, 'I-geo': 4, 'B-org': 5, 'I-org': 6, 'B-tim': 7, 'I-per': 8, 'I-gpe': 9, 'I-tim': 10}
ids_to_labels = {0: 'O', 1: 'B-geo', 2: 'B-gpe', 3: 'B-per', 4: 'I-geo', 5: 'B-org', 6: 'I-org', 7: 'B-tim', 8: 'I-per', 9: 'I-gpe', 10: 'I-tim'}
new_model = BertForTokenClassification.from_pretrained('./model_1', num_labels=len(labels_to_ids))
tokenizer = BertTokenizerFast.from_pretrained('./model_1')
new_model.to(device)
MAX_LEN = 128

def get_prediction_bert(sentence):
    inputs = tokenizer(sentence.split(),
                        # is_pretokenized=True, 
                        is_split_into_words = True,
                        return_offsets_mapping=True, 
                        padding='max_length', 
                        truncation=True, 
                        max_length=MAX_LEN,
                        return_tensors="pt")
    
    # move to gpu
    ids = inputs["input_ids"].to(device)
    mask = inputs["attention_mask"].to(device)
    # forward pass
    outputs = new_model(ids, attention_mask=mask)
    logits = outputs[0]
    
    active_logits = logits.view(-1, new_model.num_labels) # shape (batch_size * seq_len, num_labels)
    flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size*seq_len,) - predictions at the token level
    
    tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
    token_predictions = [ids_to_labels[i] for i in flattened_predictions.cpu().numpy()]
    wp_preds = list(zip(tokens, token_predictions)) # list of tuples. Each tuple = (wordpiece, prediction)
    
    prediction = []
    for token_pred, mapping in zip(wp_preds, inputs["offset_mapping"].squeeze().tolist()):
      #only predictions on first word pieces are important
      if mapping[0] == 0 and mapping[1] != 0:
        prediction.append(token_pred[1])
      else:
        continue
    return prediction

def main(sentence):
    prediction = get_prediction_bert(sentence)
    out = []
    for i, j in zip(prediction,sentence.split()):
        out.append(i+' - '+j)
    return out
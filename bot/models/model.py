from transformers import DistilBertTokenizer
import torch


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = torch.load('bot/models/distilbert3eps.pth', map_location='cpu')


def classify(sentence):
    print('Отзыв получен ...')
    enc = tokenizer.encode_plus(sentence, max_length=512,
                                pad_to_max_length=True,
                                return_attention_mask=True,
                                return_tensors='pt',
                                truncation=True,
                                )

    input_id, attention_mask = enc['input_ids'], enc['attention_mask']
    print('Отзыв отправлен на оценку ...')
    cls = model(
        input_ids=input_id,
        attention_mask=attention_mask
    ).logits
    rating = torch.argmax(cls, dim=1).cpu().detach().numpy() + 1
    return f'Оценка отзыва составляет {rating.item()} звезд'

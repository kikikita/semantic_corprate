"""Training module"""
import json
import torch
import pandas as pd
from transformers import DistilBertTokenizer, AdamW, \
    DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
from loguru import logger
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, Sampler


def convert_to_dataset_torch(data: pd.DataFrame, model_config=None):
    """Method for conversion from pandas DataFrame to torch DataSet"""
    input_ids = []
    attention_masks = []
    tokenizer_config = model_config.get("tokenizer_config", {
        "max_length": 512,
        "pad_to_max_length": True,
        "return_attention_mask": True,
        "return_tensors": 'pt',
        "truncation": True
    })
    model_name = model_config.get("model_name", "distilbert-base-uncased")
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    for row in tqdm(data, total=data.shape[0]):
        encoded_dict = tokenizer.encode_plus(row, **tokenizer_config)
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    input_ids.to(dtype=torch.long)
    attention_masks.to(dtype=torch.long)
    return input_ids, attention_masks


def dataprep(data: pd.DataFrame, sampler: Sampler, labels=None,
             dataloader_config=None):
    """Method for data preparation"""
    dataloader_config = dataloader_config.get("dataloader_config", {
        "batch_size": 16,
        "num_workers": 0,
        "drop_last": True})

    inps, masks = convert_to_dataset_torch(data)
    if labels:
        labels = torch.tensor(labels)
        encoded = TensorDataset(inps, masks, labels)
    else:
        encoded = TensorDataset(inps, masks)

    dataloader = DataLoader(
        encoded,
        sampler=sampler(encoded),
        **dataloader_config
    )

    return dataloader


def trainloop(model, optimizer, dataloader, device, epochs):
    """Training loop method"""
    model.to(device)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    total_train_loss = 0
    for epoch_i in tqdm(range(0, epochs)):
        print("")
        print(f'======== Epoch {epoch_i + 1} / {epochs} ========')
        print('Training...')

        losses = []
        model.train()

        for _, batch in enumerate(dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            optimizer.zero_grad()

            loss = model(input_ids=b_input_ids,
                         attention_mask=b_input_mask,
                         labels=b_labels).loss

            total_train_loss += loss.item()
            losses.append(loss.item())
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            scheduler.step()

            if len(losses) == 20:
                print(f"Loss: {sum(losses) / len(losses)}")
                losses = []

    print("")
    print(f"  Average training loss: {total_train_loss / len(dataloader):.2f}")

    return model


def get_data(ml_config):
    """Method for data preparing"""
    logger.info("Preparing data")

    data_config = ml_config.get("data_config", None)
    train_df_csv = data_config.get("train_data", None)
    target_column = data_config.get("target_column", "Rating")
    data_column = data_config.get("data_column", "Review")

    train_df = pd.read_csv(train_df_csv)

    y_train = train_df[target_column]
    train = train_df[data_column]

    logger.info("Data prepared")
    return train, y_train


def run_train(train_config_name):
    """Training method"""
    with open(train_config_name, encoding="utf-8") as file:
        ml_config = json.load(file)

    logger.info("Config received")
    train, y_train = get_data(ml_config)
    train_dataloader = dataprep(
        data=train,
        labels=y_train - 1,
        sampler=RandomSampler,
        dataloader_config=ml_config.get("dataloader_config", None)
    )

    logger.info("Preparing model")

    model_config = ml_config.get("model_config", None)
    model_name = model_config.get("model_type", "distilbert-base-uncased")
    model_params = model_config.get("model_params", {
        "num_labels": 5
    })

    model_to_train = DistilBertForSequenceClassification.from_pretrained(
        model_name, **model_params)

    optimizer_params = model_config.get("optimizer_params", {
        "lr": 2e-5,
        "eps": 1e-8
    })
    optimizer = AdamW(model_to_train.parameters(), **optimizer_params)

    logger.info("Model prepared")

    logger.info("Model training")

    trained_model = trainloop(
        model=model_to_train,
        optimizer=optimizer,
        dataloader=train_dataloader,
        device=model_config.get("device", "cpu"),
        epochs=model_config.get("num_epochs")
    )
    save_name = model_config.get("save_name", "distilbert.pth")
    torch.save(trained_model, save_name)

    logger.info("Model trained")


if __name__ == '__main__':
    import argparse

    parser_train = argparse.ArgumentParser()
    parser_train.add_argument("-ml_config", type=str, help="config")
    args = parser_train.parse_args()
    run_train(args.ml_config)

"""Testing module"""
import json
import torch
import pandas as pd
import numpy as np

from tqdm import tqdm
from loguru import logger
from torch.utils.data import SequentialSampler

from src.models import dataprep


def get_data(ml_config):
    """Method for data reading"""
    logger.info("Preparing data")

    data_config = ml_config.get("data_config", None)
    test_df_csv = data_config.get("test_data", None)
    data_column = data_config.get("data_column", "Review")

    test_df = pd.read_csv(test_df_csv)

    test = test_df[data_column]

    logger.info("Data prepared")
    return test


def evaluation(model, dataloader, device):
    """Method for model evaluation"""
    preds = []
    for _, batch in enumerate(tqdm(dataloader)):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)

        predictions = model(input_ids=b_input_ids,
                            attention_mask=b_input_mask).logits
        preds.append(torch.argmax(predictions.cpu().detach().numpy()))
    return np.array(preds).flatten()


def run_inference(ml_config_name):
    """Inference method"""
    with open(ml_config_name, encoding="utf-8") as file:
        ml_config = json.load(file)
    logger.info("Config received")

    logger.info("Data preparing")

    test = get_data(ml_config)

    test_dataloader = dataprep(
        data=test,
        sampler=SequentialSampler,
        dataloader_config=ml_config.get("dataloader_config", None)
    )

    logger.info("Data prepared")

    logger.info("Model preparing")

    model_name = ml_config.get("model_name", "distilbert.pth")
    map_location = ml_config.get("device", "cpu")
    model = torch.load(model_name, map_location=map_location)

    logger.info("Model prepared")

    logger.info("Inference started")

    predictions = evaluation(model, test_dataloader, map_location)

    logger.info("Inference ended")

    sub_file = ml_config.get("submission_file", "sample_submission.csv")
    sub_name = ml_config.get("submission_name", "submission.csv")

    sub_file['preds'] = np.concatenate(predictions)
    sub = sub_file[['id', 'preds']]
    sub.columns = ['id', 'target']
    sub.to_csv(sub_name, index=False)


if __name__ == '__main__':
    import argparse

    parser_test = argparse.ArgumentParser()
    parser_test.add_argument("-ml_config", type=str, help="config")
    args = parser_test.parse_args()
    run_inference(args.ml_config)

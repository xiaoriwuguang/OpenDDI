import torch
import numpy as np
import random
import os
from utils.logger import create_logger
from parms_setting import settings
from parms_setting import set_random_seed
from data.dataset_manager import dataset_manager
from utils.logger import Logger
from utils.config import *
from models.model_manager import model_manager
from pipeline import Pipeline


def main():
    set_random_seed(1, deterministic=True)
    args = settings()
    logger = create_logger(args)
    
    args.cuda = (args.device == 'cuda')

    Dataset_manager = dataset_manager(args)
    ddi_dataset = Dataset_manager.load_dataset()
    ddi_dataset.load_data()

    Model_manager = model_manager(args)
    if args.origin:
        model = Model_manager.load_origin_model(ddi_dataset)
    else:
        model = Model_manager.load_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    ddi_pipeline = Pipeline(args, logger, ddi_dataset, model, optimizer)
    ddi_pipeline.run()

if __name__ == "__main__":
    main()

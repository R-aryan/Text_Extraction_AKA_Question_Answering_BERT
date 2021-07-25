import numpy as np
import pandas as pd
from sklearn import model_selection

from services.text_extraction.application.ai.model import TextExtractionModel
from services.text_extraction.application.ai.training.src import utils
from services.text_extraction.application.ai.training.src.dataset import TextExtractionDataset
from services.text_extraction.application.ai.training.src.engine import Engine
from services.text_extraction.application.ai.training.src.preprocess import Preprocess
from services.text_extraction.settings import Settings

from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader


class Train:
    def __init__(self):
        # initialize required class
        self.settings = Settings
        self.engine = Engine()
        self.preprocess = Preprocess()

        # initialize required variables
        self.bert_text_model = None
        self.optimizer = None
        self.scheduler = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.total_steps = None
        self.best_jaccard = 0
        self.param_optimizer = None
        self.optimizer_parameters = None
        self.total_steps = None
        self.train_data_loader = None
        self.validation_data_loader = None

    def optimizer_params(self):
        self.param_optimizer = list(self.bert_text_model.named_parameters())
        self.optimizer_parameters = [
            {
                "params": [
                    p for n, p in self.param_optimizer if not any(nd in n for nd in self.settings.no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in self.param_optimizer if any(nd in n for nd in self.settings.no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

    def __initialize(self):
        # Instantiate Bert Classifier
        self.bert_text_model = TextExtractionModel()
        self.bert_text_model.to(self.settings.DEVICE)
        self.optimizer_params()

        # Create the optimizer
        self.optimizer = AdamW(self.optimizer_parameters,
                               lr=5e-5,  # Default learning rate
                               eps=1e-8  # Default epsilon value
                               )

        # Set up the learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=0,  # Default value
                                                         num_training_steps=self.total_steps)

    def create_data_loaders(self, tweet, sentiment, selected_text, batch_size, num_workers):
        dataset = TextExtractionDataset(tweet=tweet, sentiment=sentiment, selected_text=selected_text)
        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        return data_loader

    def load_data(self, csv_data_path):
        df = pd.read_csv(csv_data_path).dropna().reset_index(drop=True)
        df['text'] = df['text'].apply(lambda x: self.preprocess.clean_text(x))
        df['selected_text'] = df['selected_text'].apply(lambda x: self.preprocess.clean_text(x))

        df_train, df_valid = model_selection.train_test_split(
            df,
            random_state=self.settings.seed_value,
            test_size=self.settings.test_size,
            stratify=df.sentiment.values

        )

        df_train = df_train.reset_index(drop=True)
        df_valid = df_valid.reset_index(drop=True)

        # creating Data Loaders
        # train data loader
        self.train_data_loader = self.create_data_loaders(tweet=df_train.tweet.values,
                                                          sentiment=df_train.sentiment.values,
                                                          selected_text=df_train.selected_text.values,
                                                          batch_size=self.settings.TRAIN_BATCH_SIZE,
                                                          num_workers=self.settings.TRAIN_NUM_WORKERS)
        # validation data loader
        self.validation_data_loader = self.create_data_loaders(tweet=df_valid.tweet.values,
                                                               sentiment=df_valid.sentiment.values,
                                                               selected_text=df_valid.selected_text.values,
                                                               batch_size=self.settings.TRAIN_BATCH_SIZE,
                                                               num_workers=self.settings.TRAIN_NUM_WORKERS)
        # validation data loader

        self.total_steps = int(len(df_train) / self.settings.TRAIN_BATCH_SIZE * self.settings.EPOCHS)

    def train(self):
        early_stopping = utils.EarlyStopping(patience=5, mode="max")
        for epochs in range(self.settings.EPOCHS):
            self.engine.train_fn(data_loader=self.train_data_loader,
                                 model=self.bert_text_model,
                                 optimizer=self.optimizer,
                                 device=self.settings.DEVICE,
                                 schedular=self.scheduler)

            self.best_jaccard = self.engine.eval_fn(data_loader=self.validation_data_loader,
                                                    model=self.bert_text_model,
                                                    device=self.settings.DEVICE)

            print(f"Jaccard Score = {self.best_jaccard}")
            early_stopping(epoch_score=self.best_jaccard,
                           model=self.bert_text_model,
                           model_path=self.settings.WEIGHTS_PATH)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    def run(self):
        try:
            print("Loading and Preparing the Dataset-----!! ")
            self.load_data(csv_data_path=self.settings.TRAIN_DATA)
            print("Dataset Successfully Loaded and Prepared-----!! ")
            print()
            print("-" * 70)
            print("Loading and Initializing the Bert Model -----!! ")
            self.__initialize()
            print("Model Successfully Loaded and Initialized-----!! ")
            print()
            print("-" * 70)
            print("------------------Starting Training-----------!!")
            self.engine.set_seed()
            self.train()
            print("Training complete-----!!!")

        except BaseException as ex:
            print("Following Exception Occurred---!! ", str(ex))

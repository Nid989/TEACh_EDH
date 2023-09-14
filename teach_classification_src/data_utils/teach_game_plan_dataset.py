from typing import Literal, Optional
import torch
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import gc
from utils import load_from_pickle, pad_seq

data_types_ = Literal["train", "validation"]

class TEACh_GamePlan_Dataset:
    def __init__(self, config, tokenizer):
        super(TEACh_GamePlan_Dataset, self).__init__()

        self.config = config
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_setting = config["MODEL_SETTING"] # `unimodal (lang & actions only)` or `multimodal`
        # self.train_dataset = load_from_pickle(self.config["PATH_TO_TRAIN_DATA"])
        # self.validation_dataset = load_from_pickle(self.config["PATH_TO_VALIDATION_DATA"])
        self.train_dataset = pd.read_pickle(self.config["PATH_TO_TRAIN_DATA"])
        self.validation_dataset = pd.read_pickle(self.config["PATH_TO_VALIDATION_DATA"])
        self.train_data_loader = self.set_up_data_loader(data_type="train") # convert to tensors & batch dataset items
        self.validation_data_loader = self.set_up_data_loader(data_type="validation") # convert to tensors & batch dataset items

    def convert_actions_to_onehot(self,
                                  sequence_actions: torch.Tensor,
                                  num_classes: Optional[int]=None) -> Optional[torch.Tensor]:

        """
        Args:
            sequence_actions: multi-dimensional torch.Tensor w/ dim=0 `batch_size`,
                        & dim=1 `max_sequence_length`.
            num_classes: output dimensionality of one-hot encoded representation,
                        if num_classes=`None`, then F.one_hot wil automatically decide
                        max_num_classes possible.
        Output:
            One-hot encoded torch.Tensor w/ dim=0 `batch_size`, dim=1 `max_sequence_length`,
                        and dim=2 `num_classes`.
        """
        return F.one_hot(sequence_actions, num_classes=num_classes)

    def preprocess_dataset(self, dataset):
        source_dialogs = [item for item in dataset[self.config["SOURCE_DIALOG_COLUMN"]]]
        model_inputs = self.tokenizer(source_dialogs,
                                      max_length=self.config["SOURCE_DIALOG_MAX_LEN"],
                                      padding="max_length",
                                      truncation=True)
        model_inputs['input_ids'] = torch.tensor([item for item in model_inputs['input_ids']], dtype=torch.long, device=self.device)
        model_inputs['attention_mask'] = torch.tensor([item for item in model_inputs['attention_mask']], dtype=torch.long, device=self.device)

        source_game_plans = [item for item in dataset[self.config["SOURCE_GAME_PLAN_COLUMN"]]]
        game_plan_encodings = self.tokenizer(source_game_plans,
                                             max_length=self.config["SOURCE_GAME_PLAN_MAX_LEN"],
                                             padding="max_length",
                                             truncation=True,
                                             is_split_into_words=True)
        model_inputs["game_plan_input_ids"] = torch.tensor([item for item in game_plan_encodings['input_ids']], dtype=torch.long, device=self.device)
        model_inputs["game_plan_attention_mask"] = torch.tensor([item for item in game_plan_encodings['attention_mask']], dtype=torch.long, device=self.device)

        # target_actions (`labels`)
        model_inputs["labels"] = torch.stack([pad_seq(torch.tensor(item, dtype=torch.int64).unsqueeze(dim=1),
                                                      dim=1,
                                                      max_len=self.config["TARGET_GAME_PLAN_MAX_LEN"],
                                                      pad_token_id=self.config["TARGET_GAME_PLAN_PADDING_IDX"])
                                              for item in dataset[self.config["TARGET_GAME_PLAN_COLUMN"]].values.tolist()], 0).to(self.device)

        if self.model_setting == "multimodal":
            # pad visual feats i.e. driver_images_history_feats & driver_images_future_feats
            # driver_images_history_feats
            model_inputs["visual_input"] = torch.stack([pad_seq(torch.tensor(item, dtype=torch.float),
                                                                dim=self.config["SOURCE_VISUAL_DIM"],
                                                                max_len=self.config["SOURCE_VISUAL_MAX_LEN"])
                                                        for item in dataset[self.config["SOURCE_VISUAL_COLUMN"]].values.tolist()], 0).to(self.device)

        del source_dialogs
        del source_game_plans
        del game_plan_encodings
        gc.collect()
        return model_inputs

    def formulate_object_labels(self,
                                target_obj_interaction_actions: torch.Tensor,
                                target_objects: torch.Tensor):
        object_labels = torch.zeros_like(target_obj_interaction_actions)
        object_labels[target_obj_interaction_actions.nonzero(as_tuple=True)] = target_objects
        return object_labels

    def set_up_data_loader(self, data_type: data_types_ = "train"):

        if data_type == "train":
            dataset = self.preprocess_dataset(self.train_dataset)
        elif data_type == "validation":
            dataset = self.preprocess_dataset(self.validation_dataset)
        else:
            raise ValueError(f"provide an appropriate data-split to exctract features; found {data_type}, instead of `train` or `validation`")

        print(dataset.keys())

        if self.model_setting == "unimodal":
            dataset = TensorDataset(
                dataset["input_ids"],
                dataset["attention_mask"],
                dataset["game_plan_input_ids"],
                dataset["game_plan_attention_mask"],
                dataset["labels"].squeeze(-1).to(torch.int64)
            )
        elif self.model_setting == "multimodal":
            dataset = TensorDataset(
                dataset["input_ids"],
                dataset["attention_mask"],
                dataset["game_plan_input_ids"],
                dataset["game_plan_attention_mask"],
                dataset["visual_input"],
                dataset["labels"].squeeze(-1).to(torch.int64)
            )
        else:
            raise ValueError(f"provide an appropriate `model-setting` for further processing; found {self.config['MODEL_SETTING']}, instead of `unimodal` or `multimodal`")

        data_loader = DataLoader(
            dataset,
            batch_size=self.config["BATCH_SIZE"],
            shuffle=True
        )

        del dataset
        gc.collect()
        return data_loader

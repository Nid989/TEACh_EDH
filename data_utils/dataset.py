import torch
from torch.utils.data import DataLoader, TensorDataset
import gc
from utils import load_from_pickle, pad_seq

class TEACh_EDH_Dataset:
    def __init__(self, config, tokenizer):
        super(TEACh_EDH_Dataset, self).__init__()

        self.config = config
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset = load_from_pickle(self.config["PATH_TO_DATA"])
        self.set_up_data_loader() # convert to tensors & batch dataset items

    def preprocess_dataset(self):
        source_dialogs = [item for item in self.dataset[self.config["SOURCE_DIALOG_COLUMN"]]]
        model_inputs = self.tokenizer(source_dialogs,
                                max_length=self.config["SOURCE_DIALOG_MAX_LEN"],
                                padding="max_length",
                                truncation=True)
        model_inputs['input_ids'] = torch.tensor([item for item in model_inputs['input_ids']], dtype=torch.long, device=self.device)
        model_inputs['attention_mask'] = torch.tensor([item for item in model_inputs['attention_mask']], dtype=torch.long, device=self.device)
        # pad action sequences i.e. source_actions & target_actions
        # NOTE: avoiding the processing of obj_interaction_actions and objects
        # source_actions
        model_inputs["action_input"] = torch.stack([pad_seq(torch.tensor(item, dtype=torch.int64).unsqueeze(dim=1),
                                                            dim=1,
                                                            max_len=self.config["SOURCE_ACTION_MAX_LEN"])
                                                    for item in self.dataset[self.config["SOURCE_ACTION_COLUMN"]].values.tolist()], 0).to(self.device)
        # target_actions (`labels`)
        model_inputs["labels"] = torch.stack([pad_seq(torch.tensor(item, dtype=torch.int64).unsqueeze(dim=1),
                                                      dim=1,
                                                      max_len=self.config["TARGET_ACTION_MAX_LEN"],
                                                      pad_token_id=self.config["ACTION_PADDING_IDX"])
                                              for item in self.dataset[self.config["TARGET_ACTION_COLUMN"]].values.tolist()], 0).to(self.device)
        print(model_inputs["labels"].shape)

        # pad visual feats i.e. driver_images_history_feats & driver_images_future_feats
        # driver_images_history_feats
        model_inputs["visual_input"] = torch.stack([pad_seq(torch.tensor(item, dtype=torch.float),
                                                            dim=self.config["VISUAL_DIM"],
                                                            max_len=self.config["SOURCE_VISUAL_MAX_LEN"])
                                                    for item in self.dataset[self.config["SOURCE_VISUAL_COLUMN"]].values.tolist()], 0).to(self.device)
        # driver_images_future_feats
        model_inputs["decoder_visual_input"] = torch.stack([pad_seq(torch.tensor(item, dtype=torch.float),
                                                            dim=self.config["VISUAL_DIM"],
                                                            max_len=self.config["TARGET_VISUAL_MAX_LEN"])
                                                    for item in self.dataset[self.config["TARGET_VISUAL_COLUMN"]].values.tolist()], 0).to(self.device)

        # ================================ Modifications ================================ #
        # TODO: utilize the `target_obj_interaction_actions` and formulate `object_labels`
        model_inputs["object_labels"] = torch.stack([pad_seq((self.formulate_object_labels(torch.tensor(item[0], dtype=torch.int64), torch.tensor(item[1], dtype=torch.int64))).unsqueeze(dim=1),
                                                             dim=1,
                                                             max_len=self.config["TARGET_ACTION_MAX_LEN"]) 
                                                     for item in self.data[[self.config["TARGET_OBJ_INTERACTION_ACTION_COLUMN"], self.config["TARGET_OBJECT_COLUMN"]]].values.tolist()], 0)
        # =============================================================================== #

        del source_dialogs
        gc.collect()
        return model_inputs

    def formulate_object_labels(self,
                                target_obj_interaction_actions: torch.Tensor, 
                                target_objects: torch.Tensor):
        object_labels = torch.zeros_like(target_obj_interaction_actions)
        object_labels[target_obj_interaction_actions.nonzero(as_tuple=True)] = target_objects
        return object_labels

    def set_up_data_loader(self):
        dataset = self.preprocess_dataset()
        print(dataset.keys())
        dataset = TensorDataset(
            dataset["input_ids"],
            dataset["attention_mask"],
            dataset["action_input"].squeeze(-1),
            dataset["visual_input"],
            dataset["decoder_visual_input"],
            dataset["labels"].squeeze(-1),
            dataset["object_labels"].squeeze(-1),
        )
        self.data_loader = DataLoader(
            dataset,
            batch_size=self.config["BATCH_SIZE"],
            shuffle=True
        )
        del dataset
        gc.collect()
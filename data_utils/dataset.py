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
        model_inputs["action_input"] = torch.stack([pad_seq(torch.tensor(item, dtype=torch.long).unsqueeze(dim=1),
                                                            dim=1,
                                                            max_len=self.config["SOURCE_ACTION_MAX_LEN"]) 
                                                    for item in self.dataset[self.config["SOURCE_ACTION_COLUMN"]].values.tolist()], 0).to(self.device)
        # target_actions
        model_inputs["action_output"] = torch.stack([pad_seq(torch.tensor(item, dtype=torch.long).unsqueeze(dim=1),
                                                            dim=1,
                                                            max_len=self.config["TARGET_ACTION_MAX_LEN"]) 
                                                    for item in self.dataset[self.config["TARGET_ACTION_COLUMN"]].values.tolist()], 0).to(self.device)
        # pad visual feats i.e. driver_images_history_feats & driver_images_future_feats
        # driver_images_history_feats
        model_inputs["visual_input"] = torch.stack([pad_seq(torch.tensor(item, dtype=torch.float),
                                                            dim=self.config["VISUAL_DIM"],
                                                            max_len=self.config["SOURCE_VISUAL_MAX_LEN"]) 
                                                    for item in self.dataset[self.config["SOURCE_VISUAL_COLUMN"]].values.tolist()], 0).to(self.device)
        # driver_images_future_feats
        model_inputs["visual_output"] = torch.stack([pad_seq(torch.tensor(item, dtype=torch.float),
                                                            dim=self.config["VISUAL_DIM"],
                                                            max_len=self.config["TARGET_VISUAL_MAX_LEN"]) 
                                                    for item in self.dataset[self.config["TARGET_VISUAL_COLUMN"]].values.tolist()], 0).to(self.device)
        
        del source_dialogs
        gc.collect()
        return model_inputs    

    def set_up_data_loader(self):
        dataset = self.preprocess_dataset()
        print(dataset.keys())
        dataset = TensorDataset(
            dataset["input_ids"],
            dataset["attention_mask"],
            dataset["action_input"],
            dataset["action_output"],
            dataset["visual_input"],
            dataset["visual_output"]
        )
        self.data_loader = DataLoader(
            dataset,
            batch_size=self.config["BATCH_SIZE"],
            shuffle=True
        )
        del dataset
        gc.collect()
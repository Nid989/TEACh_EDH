import os
import gc
import re
import copy
import pandas as pd
import numpy as np
import PIL
from tqdm import tqdm
from typing import Optional, Tuple, Literal
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor, ViTForImageClassification
from vocab import Vocab

from utils import load_from_json, load_image, get_gpu_memory_usage

# TODO: load preprocessor specific config file and define dynamic variable paths

tqdm.pandas()

dialog_history_column_types_ = Literal["dialog_history", "dialog_history_cleaned"]
driver_images_column_types_ = Literal["driver_image_history", "driver_images_future"]
driver_actions_column_types_ = Literal["driver_action_history", "driver_actions_future"]

class TEACh_Data_Preprocessor:
    def __init__(self, config):
        super(TEACh_Data_Preprocessor, self).__init__()
        
        self.config = config
        self.dataset_type = config["dataset_type"]
        self.path_to_edh_instances_dir = os.path.join(config["path_to_edh_instances_dir"], config["dataset_type"])
        self.path_to_edh_images_dir = os.path.join(config["path_to_edh_images_dir"], config["dataset_type"])
        # load edh instances of self.dataset_type i.e. `train`, `valid_seen`, or `valid_unseen`
        self.data = self.load_edh_instance_data(truncate=False)
        
        # setup device `cuda` or `cpu`
        self.device_setup()
        # initialize ViT image-processor & model
        self.load_ViT(model_checkpoint=config["vit_model_checkpoint"])
        # load action_class & obj_class vocab
        self.action_vocab = torch.load(config["path_to_action_class_vocab"])
        self.obj_vocab = torch.load(config["path_to_obj_class_vocab"])
        
    def load_edh_instance_data(self, truncate: Optional[bool]=False) -> pd.DataFrame:
        data = defaultdict(list)
        _ = [data[key].append(value) for instance in os.scandir(self.path_to_edh_instances_dir) \
             for key, value in load_from_json(instance.path).items()]
        data = pd.DataFrame(data)
        if truncate:
            keep_columns = ["dialog_history", "driver_action_history", "driver_image_history", "driver_actions_future", \
                            "driver_images_future", "interactions", "game_id"]
            data = data[keep_columns]
        return data
    
    def derive_image_file_path(self, filename: str, game_id: str) -> str:
        path_to_game_dir = os.path.join(self.path_to_edh_images_dir, game_id)
        assert filename in [item.name for item in  os.scandir(path_to_game_dir)], f"file not found @ game-directory {game_id}"
        return os.path.join(path_to_game_dir, filename)
    
    def device_setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.device_count() and torch.cuda.is_available():
            gpu_memory_usage = get_gpu_memory_usage()
            gpu_id = np.argmin(np.asarray(gpu_memory_usage))
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            torch.cuda.set_device(int(gpu_id))
    
    def process_dialogs(self, index: int, column: dialog_history_column_types_="dialog_history") -> Optional[str]:
        example_dialog_history = self.data.iloc[index][column]
        roles, utterances = list(zip(*example_dialog_history))
        example_dialog = " ".join([f"[{role.upper()}]: {utterance.lower()} | " for role, utterance in zip(roles, utterances)])
        example_dialog = re.sub(" +", " ", example_dialog)
        return example_dialog
    
    def prepare_action_vocabulary(self, init_words: list=None) -> Vocab:
        init_words = init_words if init_words is None else ["<pad>", "<stop>", "<seg>", "<mask>"]
        self.action_vocab = Vocab(init_words)
        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            driver_actions_history = row["driver_action_history"]
            driver_actions_future = row["driver_actions_future"]
            _ = [self.action_vocab.word2index(action["action_name"], train=True) for action in driver_actions_history]
            _ = [self.action_vocab.word2index(action["action_name"], train=True) for action in driver_actions_future]
        torch.save(self.action_vocab, "./action_cls.vocab") # save vocabulary
    
    def process_actions(self, index: int) -> defaultdict: 
        example = self.data.iloc[index]
        example_driver_actions_history = example["driver_action_history"]
        example_driver_actions_future = example["driver_actions_future"]
        driver_actions = defaultdict(list)
        for action in example_driver_actions_history:
            action_dict_with_idx = copy.deepcopy(action)
            action_dict_with_idx["action"] = self.action_vocab.word2index(action["action_name"])
            driver_actions["actions_history"].append(action_dict_with_idx)
            driver_actions["actions_history_pred_mask"].append(0)
        for action in example_driver_actions_future:
            action_dict_with_idx = copy.deepcopy(action)
            action_dict_with_idx["action"] = self.action_vocab.word2index(action["action_name"])
            driver_actions["actions_future"].append(action_dict_with_idx)
            driver_actions["actions_future_pred_mask"].append(1)
        return driver_actions
    
    def prepare_actions_features(self):
        def _load_actions(data: dict) -> Optional[list]:
            actions_list = [[action["action"] for action in data]]
            actions_list = sum(actions_list, [])
            return actions_list

        def _load_obj_interaction_actions(data: dict) -> Optional[list]:
            obj_interaction_list = [[action["obj_interaction_action"] for action in data]]
            obj_interaction_list = sum(obj_interaction_list, [])
            return obj_interaction_list

        def _load_object_classes(data: dict, vocab: Vocab) -> Optional[list]:
            object_classes = []
            for index, action in enumerate(data):
                if action["oid"] is not None:
                    object_class = action["oid"].split("|")[0]
                    object_classes.append(vocab.word2index(object_class))
            return object_classes

        source_actions, source_obj_interaction_actions, source_objects = [], [], []
        target_actions, target_obj_interaction_actions, target_objects = [], [], []
        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            actions_data = row["driver_actions_proc"]
            # actions
            source_actions.append(_load_actions(actions_data["actions_history"]))
            target_actions.append(_load_actions(actions_data["actions_future"]))
            # obj_interaction_actions
            source_obj_interaction_actions.append(_load_obj_interaction_actions(actions_data["actions_history"]))
            target_obj_interaction_actions.append(_load_obj_interaction_actions(actions_data["actions_future"]))
            # object class
            source_objects.append(_load_object_classes(actions_data["actions_history"], self.obj_vocab))
            target_objects.append(_load_object_classes(actions_data["actions_future"], self.obj_vocab))

        action_feats_df = pd.DataFrame({
            "source_actions": source_actions,
            "target_actions": target_actions,
            "target_obj_interaction_actions": target_obj_interaction_actions,
            "target_objects": target_objects
        })
        return action_feats_df
    
    def load_ViT(self, model_checkpoint: str='google/vit-base-patch16-224') -> Optional[Tuple[ViTImageProcessor, ViTForImageClassification]]:
        self.processor = ViTImageProcessor.from_pretrained(model_checkpoint)
        self.model = ViTForImageClassification.from_pretrained(model_checkpoint).to(self.device)
        
    def generate_image_features(self, image: Optional[PIL.Image.Image]=None, 
                                batch_input: Optional[torch.Tensor]=None) -> torch.Tensor: 
        if image is not None and batch_input is not None:
            raise ValueError("You cannot specify both image and batch_input at the same time")
        elif image is not None:
            inputs = self.processor(image, return_tensors="pt")
        elif batch_input is not None:
            inputs = self.processor(batch_input, return_tensors="pt")
        else:
            raise ValueError("You have to specify either image or batch_input")
        with torch.no_grad():
            outputs = self.model(inputs['pixel_values'].to(self.device), output_hidden_states=True)
        # NOTE: somehow the `.last_hidden_state` function isn't working, hence extracting manually
        image_feats = outputs.hidden_states[-1][:, 0, :] 
        del inputs
        del outputs
        gc.collect()
        return image_feats.detach().cpu()
    
    def derive_instance_image_features(self, index: int, apply_batch: Optional[bool]=False,
                                       column: driver_images_column_types_=None) -> Optional[np.ndarray]:
        
        instance = self.data.iloc[index]
        game_id = instance["game_id"]

        if apply_batch:
            extracted_images_as_array = [] # store scene images -> List[torch.Tensor]
            for image_filename in instance[column]:
                extracted_images_as_array.append(
                    torch.tensor(np.asarray(
                        load_image(
                            self.derive_image_file_path(image_filename, game_id)
                        )
                    ))
                )
            instance_dataloader = DataLoader(extracted_images_as_array, batch_size=2)
            image_features = [self.generate_image_features(batch_input=batch) for batch in instance_dataloader]
            del extracted_images_as_array
            del instance_dataloader
            del image_filename
        else:
            image_features = []
            for image_filename in instance[column]:
                path_to_image_file = self.derive_image_file_path(image_filename, game_id) 
                image = load_image(path_to_image_file)
                image_features.append(self.generate_image_features(image=image).unsqueeze(dim=0))
            del image_filename
            del image
        del instance
        del game_id
        gc.collect()
        return torch.cat(image_features).numpy()
    
    def batch_processing_image_features(self, apply_batch: Optional[bool]=False, \
                                        column: driver_images_column_types_=None) -> None:
        if column is None:
            raise ValueError(f"You have to define driver_images_column_types:\
            {', '.join(dialog_history_column_types_.__args__)}")
           
        feature_column_name_map = {
            "driver_image_history": "driver_images_history_feats",
            "driver_images_future": "driver_images_future_feats"
        }
        self.data[feature_column_name_map[column]] = self.data.progress_apply(lambda row: \
                                                                              self.derive_instance_image_features(row.name, apply_batch, column), \
                                                                              axis=1)

    def prepare_dataset(self, sample: Optional[bool]=False, \
                     image_feature_extraction: Optional[bool]=False, \
                     train_action_vocab: Optional[bool]=False) -> None: 
        if sample:
            self.data = self.data.sample(n=5)
            self.data.reset_index(drop=True, inplace=True)
            
        # concatenate commander and driver dialog using pre-defined tags
        print("processing dialogs-history")
        self.data["dialog_history_proc"] = self.data.progress_apply(lambda row: \
                                                                    self.process_dialogs(row.name, "dialog_history_cleaned"), \
                                                                    axis=1)
        # train action vocabulary w/ predefined tags & action-names
        if train_action_vocab:
            print("processing action class vocabulary")
            self.prepare_action_vocabulary()
        # merge driver_action_history & driver_actions_future into a dict
        print("processing driver actions")
        self.data["driver_actions_proc"] = self.data.progress_apply(lambda row: \
                                                                    self.process_actions(row.name), \
                                                                    axis=1)
        self.data = pd.concat([self.data, self.prepare_actions_features()], axis=1)
        
        if image_feature_extraction:
            # generate image features for driver_image_history & driver_images_future
            print("processing driver images files")
            self.batch_processing_image_features(apply_batch=True, column="driver_image_history")
            self.batch_processing_image_features(apply_batch=True, column="driver_images_future")
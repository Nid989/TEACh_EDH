import os
import gc
import re
import copy
import pandas as pd
import numpy as np
import PIL
from tqdm import tqdm
from typing import Optional, List, Literal
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor, ViTForImageClassification
from vocab import Vocab

from utils import load_from_json, load_img_db, get_gpu_memory_usage

tqdm.pandas()

dialog_history_column_types_ = Literal["dialog_history", "dialog_history_cleaned"]
driver_images_column_types_ = Literal["driver_image_history", "driver_images_future"]
driver_actions_column_types_ = Literal["driver_action_history", "driver_actions_future"]

class TEACh_Data_Preprocessor:
    def __init__(self, config: dict):
        super(TEACh_Data_Preprocessor, self).__init__()
        
        self.config = config
        self.dataset_type = config["dataset_type"]
        self.path_to_edh_instances_dir = os.path.join(config["path_to_edh_instances_dir"], config["dataset_type"])
        self.path_to_edh_images_dir = os.path.join(config["path_to_edh_images_dir"], config["dataset_type"])
        # load edh instances of self.dataset_type i.e. `train`, `valid_seen`, or `valid_unseen`
        self.data = self.load_edh_instance_data(truncate=False)
        
        # load action_class & obj_class vocab
        self.action_vocab = torch.load(config["path_to_action_class_vocab"])
        self.obj_vocab = torch.load(config["path_to_obj_class_vocab"])
        
        # load image database proportional to `dataset_type`
        self.img_db = load_img_db(path_to_file=config["path_to_img_db"])
        
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
    
    # TODO: make sure | doesnot appear at last!
    def process_dialogs(self, index: int, column: dialog_history_column_types_="dialog_history") -> Optional[str]:
        example_dialog_history = self.data.iloc[index][column]
        roles, utterances = list(zip(*example_dialog_history))
        example_dialog = " ".join([f"[{role.upper()}]: {utterance.lower()} | " for role, utterance in zip(roles, utterances)])
        example_dialog = re.sub(" +", " ", example_dialog)
        return example_dialog
    
    def prepare_action_vocabulary(self, init_words: list=None) -> Vocab:
        init_words = init_words if init_words is not None else ["<pad>", "<sos>", "<eos>"]
        print(init_words)
        self.action_vocab = Vocab(init_words)
        print(self.action_vocab)
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
                    object_classes.append(vocab.word2index(object_class, train=True))
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
        
        # IMPORTANT: need to append the <eos>{end-of-sentence} tag to the target_actions to make sure
        # that when employing decoder it attends towards terminating the sequence. As well as 
        # target_obj_interaction_actions.
        action_feats_df["target_actions"] = action_feats_df.progress_apply(
            lambda row: row["target_actions"] + [self.action_vocab.word2index("<eos>")], 
            axis=1
        )
        action_feats_df["target_obj_interaction_actions"] = action_feats_df.progress_apply(
            lambda row: row["target_obj_interaction_actions"] + [0], 
            axis=1
        )
            
        return action_feats_df
    
    def shift_images_to_right_w_history(self, history_images: list, future_images: list):
        return [history_images[-1]] + future_images
    
    def batch_processing_image_features(self, column: driver_images_column_types_=None):
        if column is None:
            raise ValueError("please provide a appropriate column of type `driver_images_column_types`.")

        def batch_forumlate_db_keys(img_file_names: List[str], game_id: str) -> List[str]:

            def formulate_db_key(filename: str, game_id: str):
                    return f"{game_id}/{filename}"
            return [formulate_db_key(filename, game_id) for filename in img_file_names]

        def batch_search_and_aggregate_features(key_indices: List[str]) -> torch.Tensor:
            output_features = []
            for key in key_indices:

                filtered_data = self.img_db[self.img_db["path_to_image_file"] == key]["feature_representation"]
                if filtered_data.empty:
                    raise ValueError("key not found error, please make sure the provided key_indices constitute of appropriate keys.")
                output_features.append(torch.tensor(filtered_data.item(), dtype=torch.float))
            return torch.stack(output_features, dim=0)

        feature_column_name_map = {
            "driver_image_history": "driver_images_history_feats",
            "driver_images_future": "driver_images_future_feats"
        }

        # formulate key-indices used to search across image-database for feature extraction
        self.data[column] = self.data.apply(
            lambda row: batch_forumlate_db_keys(row[column], row["game_id"]),
            axis=1
        )

        self.data[feature_column_name_map[column]] = self.data.progress_apply(
            lambda row: batch_search_and_aggregate_features(row[column]),
            axis=1
        )

    def prepare_dataset(self, sample: Optional[bool]=False, \
                        image_feature_extraction: Optional[bool]=False, \
                        train_action_vocab: Optional[bool]=False, \
                        shift_images_to_right: Optional[bool]=True) -> None: 
    
        # drop rows with `0` history or future images
        self.data.drop(
            self.data.apply(lambda row: len(row["driver_image_history"]), axis=1)[lambda x: x==0].index, 
            inplace=True
        )
        self.data.drop(
            self.data.apply(lambda row: len(row["driver_images_future"]), axis=1)[lambda x: x==0].index, 
            inplace=True
        )
        self.data.reset_index(drop=True, inplace=True)
        
        # normalize `driver_image_future` w/ decoder_action_inputs
        # `driver_image_history`[-1] + shift_image_to_right 
        # accomodation for <start-of-action> tokens in decoder_action_inputs
        if shift_images_to_right:
            print("shifting images to right by 1 position")
            # shift target images to right by 1 position, and 
            # append the last input images 
            # (correspondance to start-of-action-sequence)
            self.data["driver_images_future"] = self.data.progress_apply(
                lambda row: self.shift_images_to_right_w_history(
                    row["driver_image_history"], 
                    row["driver_images_future"]
            ), axis=1)
        
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
            self.batch_processing_image_features(column="driver_image_history")
            self.batch_processing_image_features(column="driver_images_future")


if __name__ == "__main__":
    
    # change root directory to "./data"
    path_to_data_dir = os.path.join(os.getcwd(), "./data")
    if os.path.exists(path_to_data_dir):
        os.chdir(path_to_data_dir)
    else:
        raise ValueError("please make sure that the file is being executed under the root directory, and \"./data\" directory if accessible.")

    # Possible `dataset_type`: `train`, `valid_seen`, or `valid_unseen`

    # train
    teach_data_preprocessor_train = TEACh_Data_Preprocessor(config={
        "dataset_type": "train",
        "path_to_edh_instances_dir": "./edh_instances/",
        "path_to_edh_images_dir": "./images/",
        "path_to_obj_class_vocab": "./obj_cls.vocab",
        "path_to_action_class_vocab": "./action_cls.vocab",
        "path_to_img_db": "./TEACh_image_database_train.pkl"
    })

    teach_data_preprocessor_train.prepare_dataset(sample=True, image_feature_extraction=True, train_action_vocab=False, shift_images_to_right=True)
    teach_data_preprocessor_train.data.to_pickle("./TEACh_dataset_train.pkl")

    # valid_seen
    teach_data_preprocessor_valid_seen = TEACh_Data_Preprocessor(config={
        "dataset_type": "valid_seen",
        "path_to_edh_instances_dir": "./edh_instances/",
        "path_to_edh_images_dir": "./images/",
        "path_to_obj_class_vocab": "./obj_cls.vocab",
        "path_to_action_class_vocab": "./action_cls.vocab",
        "path_to_img_db": "./TEACh_image_database_valid_seen.pkl"
    })

    teach_data_preprocessor_valid_seen.prepare_dataset(sample=True, image_feature_extraction=True, train_action_vocab=False, shift_images_to_right=True)
    teach_data_preprocessor_valid_seen.data.to_pickle("./TEACh_dataset_valid_seen.pkl")
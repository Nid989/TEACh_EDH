import os
import gc
import pandas as pd
from typing import Optional
from collections import defaultdict
from tqdm import tqdm

import torch
import torchvision
from torchvision import transforms
from transformers import ViTImageProcessor, ViTForImageClassification
from utils import load_from_json

class TEACh_Image_Database:
    def __init__(self, config: dict):
        super(TEACh_Image_Database, self).__init__()
        
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_type = config["dataset_type"]
        self.path_to_edh_instances_dir = os.path.join(config["path_to_edh_instances_dir"], config["dataset_type"])
        self.path_to_edh_images_dir = os.path.join(config["path_to_edh_images_dir"], config["dataset_type"])
        # load edh instances of self.dataset_type i.e. `train`, `valid_seen`, or `valid_unseen`
        self.data = self.load_edh_instance_data(truncate=False)
        # formulate total unique image-files across dataset_type 
        self.derive_unique_file_paths()
        # load ViT model and feature-extractor/processor
        self.load_ViT(config["vit_model_checkpoint"])
        self.batch_size = config["batch_size"]
        
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
    
    def derive_unique_file_paths(self):
        future_image = [f"./images/{self.dataset_type}/{row['game_id']}/{image_file_name}" for index, row in self.data.iterrows() for image_file_name in row["driver_images_future"]]
        history_image = [f"./images/{self.dataset_type}/{row['game_id']}/{image_file_name}" for index, row in self.data.iterrows() for image_file_name in row["driver_image_history"]]
        self.total_image_file = list(set(history_image + future_image))
        del future_image
        del history_image
        gc.collect()
    
    def load_ViT(self, model_checkpoint: str="google/vit-base-patch16-224"):
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_checkpoint)
        self.model = ViTForImageClassification.from_pretrained(model_checkpoint).to(self.device)
        self.model.eval()
        
    def prepare_image_database(self):
        
        # data-loader; yields batches of file-paths
        def batch_generator(file_list, batch_size):
            for i in range(0, len(file_list), batch_size):
                yield file_list[i:i+batch_size]

        # iterate over set of batch of file-paths and extract image-features
        def extract_features_batch(file_paths):
            batch_images = [transforms.ToPILImage()(torchvision.io.read_image(file_path)) for file_path in file_paths]
            inputs = self.feature_extractor(batch_images, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs.to("cuda"), output_hidden_states=True)
            features = outputs.hidden_states[-1][:, 0, :].detach().clone().cpu().numpy()  # Use the mean of hidden states as features
            
            del batch_images
            del inputs
            del outputs
            gc.collect()
            return features
        
        batches = list(batch_generator(self.total_image_file, self.batch_size))
        feature_data = []
        for batch_files in tqdm(batches):
            batch_features = extract_features_batch(batch_files)
            for i in range(len(batch_files)):
                feature_data.append((batch_files[i], batch_features[i]))
                
        df = pd.DataFrame(feature_data, columns=["path_to_image_file", "feature_representation"])
        
        del batches
        del feature_data
        gc.collect()
        return df
    

if __name__ == "__main__":
    
    # change root directory to "./data"
    path_to_data_dir = os.path.join(os.getcwd(), "./data")
    if os.path.exists(path_to_data_dir):
        os.chdir(path_to_data_dir)
    else:
        raise ValueError("please make sure that the file is being executed under the root directory, and \"./data\" directory if accessible.")

    # Possible `dataset_type`: `train`, `valid_seen`, or `valid_unseen`

    # train
    teach_image_db_train = TEACh_Image_Database(config = {
        "dataset_type": "train",
        "path_to_edh_instances_dir": "./edh_instances/",
        "path_to_edh_images_dir": "./images/",
        "vit_model_checkpoint": "google/vit-base-patch16-224",
        "batch_size": 128
    })

    df_train = teach_image_db_train.prepare_image_database()
    df_train.to_pickle("./TEACh_image_database_train.pkl")

    # valid_seen
    teach_image_db_valid_seen = TEACh_Image_Database(config = {
        "dataset_type": "valid_seen",
        "path_to_edh_instances_dir": "./edh_instances/",
        "path_to_edh_images_dir": "./images/",
        "vit_model_checkpoint": "google/vit-base-patch16-224",
        "batch_size": 128
    })

    df_valid_seen = teach_image_db_valid_seen.prepare_image_database()
    df_valid_seen.to_pickle("./TEACh_image_database_valid_seen.pkl")
import gc
from tqdm import tqdm
from typing import Optional
import pandas as pd
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
from transformers.models.bart.configuration_bart import BartConfig
from transformers import BartTokenizerFast

from modeling.multimodal.multimodal_TEACh_model_for_action_generation import MultimodalTEAChModelForActionGeneration
from modeling.unimodal.TEACh_model_for_action_generation import TEAChModelForActionGeneration
from data_utils.dataset import TEACh_EDH_Dataset
from modeling.generation_utils import Generation
from utils import prepare_for_training, pad_seq, check_and_create_directory

def get_scores(pred_action_seq: list,
               gt_action_seq: list) -> Optional[dict]:

    """
    Calculate `Success Rate`, `Trajectory Length Weighted {Succcess Rate, Goal Condictiona}`
    """
    # Success = 1 if all the ground truth actions are present in the predictions
    if (pred_action_seq == gt_action_seq):
        success = 1
    else:
        success = 0

    gt_path_len = len(gt_action_seq)
    traj_len = len(pred_action_seq)
    max_val = max(gt_path_len, traj_len)

    # Expected state changes in E_hat
    exp = len([x for x in pred_action_seq if x in gt_action_seq])
    gc_success_rate = exp/gt_path_len

    # Trajectory length weighted metrics
    tlw_success = success*gt_path_len/max_val
    tlw_gc_success_rate = gc_success_rate*gt_path_len/max_val

    return {
        "success_rate": success,
        "tlw_success_rate": tlw_success,
        "gc_success_rate": gc_success_rate,
        "tlw_gc_success_rate": tlw_gc_success_rate
    }

class TEAChTrainer:
    def __init__(self, config: dict):

        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.bart_config = BartConfig.from_pretrained(config["MODEL_CHECKPOINT"])

        if config["MODEL_SETTING"] == "unimodal":
            self.model_setting = "unimodal"
            self.model = TEAChModelForActionGeneration(self.bart_config, config)
            self.model.to(self.device)
        elif config["MODEL_SETTING"] == "multimodal":
            self.model_setting = "multimodal"
            self.model = MultimodalTEAChModelForActionGeneration(self.bart_config, config)
            self.model.to(self.device)
        else:
            raise ValueError(f"provide an appropriate `model-setting` for further processing; found {config['MODEL_SETTING']}, instead of `unimodal` or `multimodal`")

        self.optimizer = prepare_for_training(model=self.model,
                                              base_learning_rate=float(config["BASE_LEARNING_RATE"]),
                                              new_learning_rate=float(config["NEW_LEARNING_RATE"]),
                                              weight_decay=float(config["WEIGHT_DECAY"]))

        self.tokenizer = BartTokenizerFast.from_pretrained(config["MODEL_CHECKPOINT"])

        self.dataset = TEACh_EDH_Dataset(config, self.tokenizer)
        self.train_data_loader = self.dataset.train_data_loader # NOTE to implement in TEACh_EDH_Dataset
        self.validation_data_loader = self.dataset.validation_data_loader # NOTE to implement in TEACh_EDH_Dataset

        self.action_generation = Generation()

    def train(self, **gen_kwargs):

        train_losses = []
        val_losses = []

        for epoch in range(self.config["MAX_EPOCHS"]):
            train_loss = self.train_epoch(model=self.model,
                                          data_loader=self.train_data_loader,
                                          optimizer=self.optimizer)
            train_losses.append(train_loss)

            val_loss = self.val_epoch(model=self.model,
                                      data_loader=self.validation_data_loader,
                                      optimizer=self.optimizer)
            val_losses.append(val_loss)

            val_results = self.get_val_scores(self.model,
                                              self.tokenizer,
                                              self.dataset.validation_dataset,
                                              desc="Validation Generation Iteration",
                                              epoch=epoch,
                                              **gen_kwargs)

            print(val_results)

            print("Epoch: {}\ttrain_loss: {}\tval_loss: {}\tmin_validation_loss: {}".format(epoch+1, train_loss, val_loss, min(val_losses)))

            print("\nval_success_rate: {}, \tval_tlw_success_rate: {}, \tval_gc_success_rate: {}, \tval_tlw_gc_success_rate: {}".format(
                val_results["success_rate"], val_results["tlw_success_rate"], val_results["gc_success_rate"], val_results["tlw_gc_success_rate"]
            ))


    def train_epoch(self,
                    model,
                    data_loader,
                    optimizer):

        model.train()
        epoch_train_loss = 0.0
        for step, batch in enumerate(tqdm(data_loader, desc="Training Iteration")):
            batch = tuple(t.to(self.device) for t in batch)

            if self.model_setting == "unimodal":

                input_ids, attention_mask, action_input, labels = batch
                optimizer.zero_grad()
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    action_input=action_input,
                    labels=labels,
                )

            elif self.model_setting == "multimodal":

                input_ids, attention_mask, action_input, visual_input, decoder_visual_input, labels = batch
                optimizer.zero_grad()
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    action_input=action_input,
                    visual_input=visual_input,
                    decoder_visual_input=decoder_visual_input,
                    labels=labels,
                )

            loss = outputs["loss"]
            epoch_train_loss += loss.item()

            loss.backward()
            optimizer.step()

        del batch
        del input_ids
        del attention_mask
        del action_input
        del labels
        del outputs
        del loss
        if self.model_setting == "multimodal":
            del visual_input
            del decoder_visual_input
        gc.collect()
        torch.cuda.empty_cache()

        return epoch_train_loss / step

    def val_epoch(self,
                  model,
                  data_loader,
                  optimizer):

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for step, batch in enumerate(tqdm(data_loader, desc="Validation Loss Iteration")):
                batch = tuple(t.to(self.device) for t in batch)

                if self.model_setting == "unimodal":

                    input_ids, attention_mask, action_input, labels = batch
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        action_input=action_input,
                        labels=labels,
                    )

                elif self.model_setting == "multimodal":

                    input_ids, attention_mask, action_input, visual_input, decoder_visual_input, labels = batch
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        action_input=action_input,
                        visual_input=visual_input,
                        decoder_visual_input=decoder_visual_input,
                        labels=labels,
                    )

                loss = outputs["loss"]
                epoch_val_loss += loss.item()

        del batch
        del input_ids
        del attention_mask
        del action_input
        del labels
        del outputs
        del loss
        if self.model_setting == "multimodal":
            del visual_input
            del decoder_visual_input
        gc.collect()
        torch.cuda.empty_cache()

        return epoch_val_loss / step

    def test_epoch(self,
                   model,
                   tokenizer: BartTokenizerFast,
                   data: pd.DataFrame,
                   desc,
                   **gen_kwargs):

        model.eval()
        predictions = []
        gold = []
        with torch.no_grad():
            for index, row in tqdm(data.iterrows(), desc=desc, total=data.shape[0]):
                encodings = tokenizer(row[self.config["SOURCE_DIALOG_COLUMN"]], max_length=self.config["SOURCE_DIALOG_MAX_LEN"], padding="max_length", return_tensors="pt")
                action_input = pad_seq(torch.tensor(row[self.config["SOURCE_ACTION_COLUMN"]], dtype=torch.int64).unsqueeze(dim=-1),
                                       dim=1,
                                       max_len=self.config["SOURCE_ACTION_MAX_LEN"]).squeeze(dim=-1).unsqueeze(dim=0)
                if self.model_setting == "unimodal":
                    visual_input=None
                    decoder_visual_input=None
                elif self.model_setting == "multimodal":
                    visual_input = pad_seq(torch.tensor(row[self.config["SOURCE_VISUAL_COLUMN"]], dtype=torch.float),
                                           dim=self.config["VISUAL_DIM"],
                                           max_len=self.config["SOURCE_VISUAL_MAX_LEN"]).unsqueeze(dim=0)
                    decoder_visual_input = None # formulated during function call `prepare_inputs_for_generation()`

                pred_trajectory = self.action_generation.generate_no_beam_search(
                    model=model,
                    encodings=encodings,
                    action_input=action_input,
                    labels=None, # labels are automatically initialized `[<sos>]`
                    cur_len=1,
                    sos_token_id=1,
                    eos_token_id=2,
                    max_decoding_length=80,
                    min_decoding_length=12,
                    do_sample=True,
                    temperature=0.01,
                    top_k=5,
                    top_p=1.0,
                    visual_input=visual_input,
                    decoder_visual_input=decoder_visual_input
                )
                predictions.append(pred_trajectory.detach().tolist())
                gold.append(row[self.config["TARGET_ACTION_COLUMN"]])

        del index
        del row
        del encodings
        del action_input
        del pred_trajectory
        if self.model_setting == "multimodal":
            del visual_input
            del decoder_visual_input
        gc.collect()
        torch.cuda.empty_cache()

        return predictions, gold

    def get_val_scores(self,
                       model,
                       tokenizer: BartTokenizerFast,
                       data: pd.DataFrame,
                       desc,
                       epoch,
                       **gen_kwargs):

        predictions, gold = self.test_epoch(model,
                                            tokenizer,
                                            data,
                                            desc=desc,
                                            **gen_kwargs)

        # `scores`; list(dict), `results`; dict(list) -> pd.DataFrame
        scores = [get_scores(prediction[0], gt) for prediction, gt in zip(predictions, gold)]
        results = defaultdict(list)
        _ = [results[key].append(value) for scores_dict in scores for key, value in scores_dict.items()]
        results = pd.DataFrame(results)
        aggregated_results = results.apply(lambda col: np.mean(col), axis=0).to_dict() # column-wise `mean/avg`

        if "Validation" in desc:
            val_df = pd.DataFrame(list(zip(gold, predictions)), columns=['target_actions', 'predicted_actions'])
            RESULT_OUTPUT_DIR = self.config["RESULT_OUTPUT_DIR"]
            file_name = check_and_create_directory(RESULT_OUTPUT_DIR + "./val/") + "./TEACh_epoch_" + str(epoch+1) + "_val_results.csv"
            val_df.to_csv(file_name, index=False)
            print("Validation File saved")

        del predictions
        del gold
        gc.collect()
        torch.cuda.empty_cache()

        return aggregated_results
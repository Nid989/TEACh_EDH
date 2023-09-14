import gc
import pandas as pd
from tqdm import tqdm
from typing import Optional

import torch
import torch.nn as nn
import nn.functional as F
from torch.optim import AdamW

from transformers.models.bart.configuration_bart import BartConfig
from transformers import BartTokenizerFast

from evaluate import load
seqeval = load("seqeval")

from utils import check_and_create_directory
from teach_classification_src.data_utils.teach_game_plan_dataset import TEACh_GamePlan_Dataset
from teach_classification_src.modeling.multimodal.multimodal_TEACh_model_for_gameplan_classification import MultimodalTEAChModelforGamePlanClassification

def get_scores(p, labels_list, full_rep: bool=False):
    predictions, labels = p

    ignore_idx_list = [-100] # pad token

    true_predictions = [
        [labels_list[p] for (p, l) in zip(prediction, label) if l not in ignore_idx_list]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [labels_list[l] for (p, l) in zip(prediction, label) if l not in ignore_idx_list]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels, zero_division=False)

    if full_rep:
        return results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

def evaluate_teach_data_scores(pred_action_seq: list,
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

        self.bart_config = BartConfig.from_pretrained(config["MODEL_CHECKPOINT"], num_labels=config["UNQ_GAME_PLAN_INDEX_COUNT"])

        if config["MODEL_SETTING"] == "unimodal":
            pass
        elif config["MODEL_SETTING"] == "multimodal":
            self.model_setting = "multimodal"
            self.model = MultimodalTEAChModelforGamePlanClassification(self.bart_config, config)
            self.model.to(self.device)
        else:
            raise ValueError(f"provide an appropriate `model-setting` for further processing; found {config['MODEL_SETTING']}, instead of `unimodal` or `multimodal`")

        self.optimizer = AdamW(self.model.parameters(),
                               lr=config["LEARNING_RATE"],
                               weight_decay=config["WEIGHT_DECAY"])

        self.tokenizer = BartTokenizerFast.from_pretrained(config["MODEL_CHECKPOINT"], add_prefix_space=True)

        self.action_object_tuple_vocab = torch.load("./action_object_tuple_vocab.pt").to_dict()["index2word"]

        self.dataset = TEACh_GamePlan_Dataset(config, self.tokenizer)
        self.train_data_loader = self.dataset.train_data_loader
        self.validation_data_loader = self.dataset.validation_data_loader

    def train(self, **gen_kwargs):

        train_losses = []
        val_losses = []
        val_f1 = []

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
                                              self.validation_data_loader,
                                              desc="Validation Generation Iteration",
                                              epoch=epoch,
                                              **gen_kwargs)

            val_f1.append(val_results["f1"])

            print("Epoch: {}\ttrain_loss: {}\tval_loss: {}\tmin_validation_loss: {}".format(epoch+1, train_loss, val_loss, min(val_losses)))

            print("val_precision: {:0.2f}\tval_recall: {:0.2f}\tval_f1: {:0.2f}\tval_accuracy: {:0.2f}".format(
                val_results["precision"], val_results["recall"], val_results["f1"], val_results["accuracy"]))

    def train_epoch(self,
                    model,
                    data_loader,
                    optimizer):

        model.train()
        epoch_train_loss = 0.0
        for step, batch in enumerate(tqdm(data_loader, desc="Training Iteration")):
            batch = tuple(t.to(self.device) for t in batch)

            if self.model_setting == "unimodal":
                pass

            elif self.model_setting == "multimodal":
                input_ids, attention_mask, game_plan_input_ids, game_plan_attention_mask, visual_input, labels = batch
                optimizer.zero_grad()
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    game_plan_input_ids=game_plan_input_ids,
                    game_plan_attention_mask=game_plan_attention_mask,
                    visual_input=visual_input,
                    labels=labels,
                )

            loss = outputs["loss"]
            epoch_train_loss += loss.item()

            loss.backward()
            optimizer.step()

        del batch
        del input_ids
        del attention_mask
        del game_plan_input_ids
        del game_plan_attention_mask
        del labels
        del outputs
        del loss
        if self.model_setting == "multimodal":
            del visual_input
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
                    pass

                elif self.model_setting == "multimodal":
                    input_ids, attention_mask, game_plan_input_ids, game_plan_attention_mask, visual_input, labels = batch
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        game_plan_input_ids=game_plan_input_ids,
                        game_plan_attention_mask=game_plan_attention_mask,
                        visual_input=visual_input,
                        labels=labels,
                    )

                loss = outputs["loss"]
                epoch_val_loss += loss.item()

        del batch
        del input_ids
        del attention_mask
        del game_plan_input_ids
        del game_plan_attention_mask
        del labels
        del outputs
        del loss
        if self.model_setting == "multimodal":
            del visual_input
        gc.collect()
        torch.cuda.empty_cache()

        return epoch_val_loss / step

    def test_epoch(self,
                   model,
                   data_loader,
                   desc,
                   **gen_kwargs):

        model.eval()
        out_predictions = []
        gold = []
        with torch.no_grad():
            pbar = tqdm(data_loader, desc=desc)
            for step, batch in enumerate(pbar):
                batch = tuple(t.to(self.device) for t in batch)
                outputs = None
                if self.model_setting == "unimodal":
                    pass

                elif self.model_setting == "multimodal":
                    input_ids, attention_mask, game_plan_input_ids, game_plan_attention_mask, visual_input, labels = batch
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        game_plan_input_ids=game_plan_input_ids,
                        game_plan_attention_mask=game_plan_attention_mask,
                        visual_input=visual_input,
                        labels=labels,
                    )

                if outputs:
                    probabilities = F.softmax(outputs.logits, dim=-1)
                    predictions = probabilities.argmax(dim=-1).cpu().tolist()

                    out_predictions.extend(predictions)
                    gold.extend(labels.cpu().tolist())

        del batch
        del input_ids
        del attention_mask
        del game_plan_input_ids
        del game_plan_attention_mask
        del labels
        del outputs
        del probabilities
        del predictions
        if self.model_setting == "multimodal":
            del visual_input
        gc.collect()
        torch.cuda.empty_cache()

        return out_predictions, gold

    def get_val_scores(self,
                       model,
                       data_loader,
                       desc,
                       epoch,
                       **gen_kwargs):

        predictions, gold = self.test_epoch(model,
                                            data_loader,
                                            desc=desc,
                                            **gen_kwargs)
        result = get_scores(p=(predictions, gold),
                            labels_list=self.action_object_tuple_vocab)

        if "Validation" in desc:
            val_df = pd.DataFrame(list(zip(gold, predictions)), columns=['target_actions', 'predicted_actions'])
            PATH_TO_RESULT_OUTPUT_DIR = self.config["PATH_TO_RESULT_OUTPUT_DIR"]
            file_name = check_and_create_directory(PATH_TO_RESULT_OUTPUT_DIR + "./val/") + "./TEACh_epoch_" + str(epoch+1) + "_val_results.csv"
            val_df.to_csv(file_name, index=False)
            print("Validation File saved")

        del predictions
        del gold
        gc.collect()
        torch.cuda.empty_cache()

        return result
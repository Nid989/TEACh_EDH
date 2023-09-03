import gc
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers.models.bart.configuration_bart import BartConfig
from transformers import BartTokenizerFast

from modeling.multimodal.multimodal_TEACh_model_for_action_generation import MultimodalTEAChModelForActionGeneration
from data_utils.dataset import TEACh_EDH_Dataset
from utils import prepare_for_training

class TEAChTrainer:
    def __init__(self, config: dict):

        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.bart_config = BartConfig.from_pretrained(config["MODEL_CHECKPOINT"])

        self.model = MultimodalTEAChModelForActionGeneration(self.bart_config, config)
        self.model.to(self.device)

        self.optimizer = prepare_for_training(model=self.model,
                                              base_learning_rate=config["BASE_LEARNING_RATE"],
                                              new_learning_rate=config["NEW_LEARNING_RATE"],
                                              weight_decay=config["WEIGHT_DECAY"])
        
        self.tokenizer = BartTokenizerFast.from_pretrained(config["MODEL_CHECKPOINT"])

        self.dataset = TEACh_EDH_Dataset(config, self.tokenizer)
        self.data_loader = self.dataset.data_loader

    def train(self):

        train_losses = []
        val_losses = []

        for epoch in range(self.config["MAX_EPOCHS"]):
            train_loss = self.train_epoch(model=self.model,
                                          data_loader=self.data_loader,
                                          optimizer=self.optimizer)
            train_losses.append(train_loss)

            val_loss = self.val_epoch(model=self.model,
                                      data_loader=self.data_loader,
                                      optimizer=self.optimizer)
            val_losses.append(val_loss)

            # ================================ evaluation ================================ #
            # add evaluation steps and further processing steps
            # ============================================================================ #

            # TOOD: complete this!

    def train_epoch(self,
                    model,
                    data_loader,
                    optimizer):

        model.train()
        epoch_train_loss = 0.0
        for step, batch in enumerate(tqdm(data_loader, desc="Training Iteration")):
            batch = tuple(t.to(self.device) for t in batch)
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
        del visual_input
        del decoder_visual_input
        del labels
        del outputs
        del loss
        gc.collect()

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
        del visual_input
        del decoder_visual_input
        del labels
        del outputs
        del loss
        gc.collect()
        torch.cuda.empty_cache()

        return epoch_val_loss / step
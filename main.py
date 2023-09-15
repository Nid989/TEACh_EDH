from teach_classification_src.legacy.TEACh_trainer import TEAChTrainer as TEAChTrainer_Cls
from teach_encoder_decoder_src.TEACh_trainer import TEAChTrainer as TEAChTrainer_EncDec
from utils import load_from_yaml

if __name__ == "__main__":

    path_to_configuration_file = "./teach_classification_src/config.yaml"
    config = load_from_yaml(path_to_configuration_file) # load `config.yaml` file

    trainer = TEAChTrainer_Cls(config=config)

    trainer.train()
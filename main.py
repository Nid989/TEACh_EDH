from TEACh_trainer import TEAChTrainer
from utils import config

if __name__ == "__main__":

    # intialize the TEACh-Trainer using pre-defined configurations
    trainer = TEAChTrainer(config=config)

    # train model on the defined architecutre
    trainer.train()
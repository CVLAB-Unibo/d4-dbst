from hesiod import hmain

from trainers.d4dep import D4DepthTrainer
from trainers.d4sem import D4SemanticsTrainer
from trainers.d4transfer import D4TransferTrainer


@hmain(base_cfg_dir="cfg/bases", template_cfg_file="cfg/d4.yaml")
def main() -> None:
    print("Training D4 depth network...")
    dep_trainer = D4DepthTrainer()
    dep_trainer.train()

    print("Training D4 semantics network...")
    sem_trainer = D4SemanticsTrainer()
    sem_trainer.train()

    print("Training D4 transfer network...")
    transfer_trainer = D4TransferTrainer()
    transfer_trainer.train()


if __name__ == "__main__":
    main()

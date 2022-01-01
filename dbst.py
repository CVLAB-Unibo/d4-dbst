from hesiod import hmain

from trainers.dbst import DBSTTrainer


@hmain(base_cfg_dir="cfg/bases", template_cfg_file="cfg/dbst.yaml")
def main() -> None:
    print("Depth-Based Self-Training...")
    dbst_trainer = DBSTTrainer()
    dbst_trainer.train()


if __name__ == "__main__":
    main()

import sys

import torch
from hesiod import get_out_dir, hmain

from trainers.d4transfer import D4TransferTrainer

if len(sys.argv) != 2:
    print("Usage: py eval_d4.py RUN_FILE")
    exit(-1)


@hmain(base_cfg_dir="cfg/bases", run_cfg_file=sys.argv[1], parse_cmd_line=False)
def main() -> None:
    trainer = D4TransferTrainer()

    ckpt_path = get_out_dir() / "d4transfer/ckpt.pt"
    ckpt = torch.load(ckpt_path)
    trainer.transfer.load_state_dict(ckpt["transfer"])

    val_source_miou = trainer.val("source")
    print(f"Source dataset mIoU: {val_source_miou:.4f}")

    val_target_miou = trainer.val("target")
    print(f"Target dataset mIoU: {val_target_miou:.4f}")


if __name__ == "__main__":
    main()

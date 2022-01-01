from hesiod import hmain

from trainers.d4uda import D4UDAMerger


@hmain(base_cfg_dir="cfg/bases", template_cfg_file="cfg/d4uda.yaml")
def main() -> None:
    print("Merging predictions from D4 and selected UDA method...")
    d4uda_merger = D4UDAMerger()
    d4uda_merger.merge()


if __name__ == "__main__":
    main()

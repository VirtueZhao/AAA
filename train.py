import argparse
import torch
from DGB.utils import get_cfg_default, set_random_seed, setup_logger
from DGB.engine import build_trainer


def setup_cfg(args):
    cfg = get_cfg_default()

    # Load config from the datasets config file
    if args.config_path_dataset:
        cfg.merge_from_file(args.config_path_dataset)
    # Load config from the baseline config file
    if args.config_path_trainer:
        cfg.merge_from_file(args.config_path_trainer)
    # Reset config from input arguments
    reset_cfg_from_args(cfg, args)

    clean_cfg(cfg, args.trainer)

    cfg.freeze()

    return cfg


def reset_cfg_from_args(cfg, args):
    if args.gpu:
        cfg.GPU = args.gpu

    if args.seed:
        cfg.SEED = args.seed

    if args.dataset_path:
        cfg.DATASET.PATH = args.dataset_path

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.max_epoch:
        cfg.OPTIM.MAX_EPOCH = args.max_epoch

    if args.batch_size:
        cfg.DATALOADER.TRAIN.BATCH_SIZE = args.batch_size

    if args.lr:
        cfg.OPTIM.LR = args.lr

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domain:
        cfg.DATASET.TARGET_DOMAIN = args.target_domain

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.op:
        cfg.TRAIN.OP = args.op
    else:
        cfg.TRAIN.OP = "None"


def clean_cfg(cfg, trainer):
    """Remove unused trainers (configs).

    Aim: Only show relevant information when calling print(cfg).

    Args:
        cfg (_C): cfg instance.
        trainer (str): baseline name.
    """
    keys = list(cfg.TRAINER.keys())
    for key in keys:
        if key == "NAME" or key == trainer.upper():
            continue
        cfg.TRAINER.pop(key, None)


def main(args):
    torch.set_num_threads(1)

    cfg = setup_cfg(args)

    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print("** Config **")
    print(cfg)

    trainer = build_trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0", help="specify GPU")
    parser.add_argument(
        "--seed", type=int, default=42, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source_domains",
        type=str,
        nargs="+",
        help="source domain for domain generalization",
    )
    parser.add_argument(
        "--target_domain",
        type=str,
        nargs="+",
        help="target domain for domain generalization",
    )
    parser.add_argument("--dataset_path", type=str, help="path to datasets")
    parser.add_argument("--output_dir", type=str, help="output directory")

    parser.add_argument("--max_epoch", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config_path_trainer", type=str, help="baseline config file path"
    )
    parser.add_argument(
        "--config_path_dataset", type=str, help="datasets config file path"
    )
    parser.add_argument("--trainer", type=str, help="name of trainers")
    parser.add_argument("--op", type=str)

    args = parser.parse_args()
    main(args)

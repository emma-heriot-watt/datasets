import os
from argparse import ArgumentParser, Namespace
from typing import Iterable

from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import ByteLevel, Whitespace
from tokenizers.trainers import BpeTrainer, WordPieceTrainer

from emma_datasets.api.storage import DatasetDB
from emma_datasets.common.logger import get_logger
from emma_datasets.datamodels.instance import Instance


logger = get_logger(__name__)


def create_data_iterator(db_path: str) -> Iterable[str]:
    """Opens the dataset and create an iterator over all the language annotations."""
    with DatasetDB(db_path) as db:
        for _, _, raw_data in db:
            data = Instance.parse_raw(raw_data)

            yield from data.language_annotations


def main(args: Namespace) -> None:
    """Trains a new tokenizer."""
    logger.info(f"Loading data from dataset {args.db_path}")
    data_iterator = create_data_iterator(args.db_path)
    # The order of the special tokens matters!
    special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

    if args.tokenizer == "bpe":
        tokenizer = Tokenizer(BPE(unk_token=special_tokens[0]))
        # After we instantiate the tokenizer, we need to setup a pre-tokenizer
        # to find correct word boundaries
        tokenizer.pre_tokenizer = ByteLevel()
        trainer = BpeTrainer(
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            special_tokens=special_tokens,
        )
    elif args.tokenizer == "wordpieces":
        tokenizer = Tokenizer(WordPiece(unk_token=special_tokens[0]))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordPieceTrainer(
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            special_tokens=special_tokens,
        )
    else:
        raise ValueError(f"Wrong choice of Tokenizer: {args.tokenizer}")

    tokenizer.normalizer = BertNormalizer(lowercase=args.lowercase)
    logger.info(f"Created {type(tokenizer)} tokenizer")

    tokenizer.train_from_iterator(data_iterator, trainer)

    logger.info(f"Saving tokenizer to path {args.output_path}")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    tokenizer.save(args.output_path)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--db_path", type=str, help="Path to a DatasetDB file", default="storage/db/instances.db"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="bpe",
        choices=("wordpieces", "bpe"),
        help="The type of tokenizer to train",
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Whether to lowercase the tokens",
    )
    parser.add_argument("--vocab_size", type=int, default=10000)  # noqa: WPS432
    parser.add_argument("--min_frequency", type=int, default=0)
    parser.add_argument(
        "--output_path",
        type=str,
        help="Tokenizer output path",
        default="storage/tokenizer/emma.json",
    )

    args = parser.parse_args()
    main(args)

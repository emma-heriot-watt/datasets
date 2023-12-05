import logging
from argparse import ArgumentParser

from emma_datasets.common import use_rich_for_logging
from emma_datasets.datamodels.datasets.utils.simbot_utils.high_level_key_processor import (
    HighLevelKeyProcessor,
)


use_rich_for_logging()
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--high_level_key",
        default="action--timemachine_target-object--bowl_target-object-color--red_converted-object--bowl-4L3ie",
    )

    parser.add_argument("--paraphrases_per_template", default=1, type=int)
    parser.add_argument("--prefix_inclusion_probability", default=0.2, type=float)  # noqa: WPS432

    args = parser.parse_args()

    high_level_key_processor = HighLevelKeyProcessor(
        prefix_inclusion_probability=args.prefix_inclusion_probability,
        paraphrases_per_template=args.paraphrases_per_template,
    )

    data = high_level_key_processor(highlevel_key=args.high_level_key)
    logger.info(f"{data}")

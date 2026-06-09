from __future__ import annotations

import sys
from typing import Optional, Sequence, Tuple

from .config import LOGGER, configure_logging, parse_args, validate_config
from .oof import run_oof_experiment
from .training import (
    run_final,
    run_multi_split_validation,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_logging()
    config = parse_args(argv)
    validate_config(config)

    LOGGER.info("Starting Expedia Learning-to-Rank with config: %s", config)

    if config.skip_validation and config.no_final:
        raise ValueError(
            "Both --skip-validation and --no-final were set; nothing to do."
        )

    if config.workflow == "oof":
        run_oof_experiment(config)
        LOGGER.info("Expedia OOF Learning-to-Rank process completed successfully.")
        return 0

    best_iteration: Optional[int] = None
    ranker_blend_weights: Tuple[float, ...] = ()
    use_ranker_blend_for_final = False

    if not config.skip_validation:
        (
            best_iteration,
            _,
            use_ranker_blend_for_final,
            ranker_blend_weights,
        ) = run_multi_split_validation(config)

    if not config.no_final:
        run_final(
            config,
            best_iteration,
            use_ranker_blend_for_final,
            ranker_blend_weights,
        )

    LOGGER.info("Expedia Learning-to-Rank process completed successfully.")
    return 0


if __name__ == "__main__":
    raise sys.exit(main(sys.argv[1:]))

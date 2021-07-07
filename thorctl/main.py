import logging
import sys
from . import commands


logger = logging.getLogger("thorctl")
logger.setLevel(logging.DEBUG)

if len(logger.handlers) == 0:
    h = logging.StreamHandler(stream=sys.stderr)
    h.setLevel(logging.DEBUG)
    h.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s - %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(h)


def main():
    parser, args = commands.parse_args()
    commands.dispatch(parser, args)

#!/usr/bin/env python
"""Export SVIRO into COCO + YOLO layouts.

Usage:
    python scripts/prepare_sviro.py --config configs/data/sviro.yaml [--copy] [data.root=/path]
"""

import argparse
import logging

from cabinsense.config import load_config
from cabinsense.data.prepare import prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data/sviro.yaml")
    parser.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    parser.add_argument("overrides", nargs="*", help="key=value overrides")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config, args.overrides)
    out = prepare(cfg, link_mode="copy" if args.copy else "symlink")
    print(f"Export written to {out}")


if __name__ == "__main__":
    main()

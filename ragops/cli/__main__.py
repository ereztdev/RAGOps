"""Entrypoint for python -m ragops.cli <command>."""

from ragops.cli.main import main
import sys

if __name__ == "__main__":
    sys.exit(main())

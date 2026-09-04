#!/usr/bin/env python3
"""kexnav -- BSP -> bspc -> .aas -> .nav

Commands:
    generate    build a .nav file for one or more maps
    check       compare a generated file against Nightdive's, where one exists

    python3 kexnav.py generate q2dm1 -o out/
    python3 kexnav.py <command> -h      for a command's own options
"""

import sys

from kexnav.cli import check, generate

COMMANDS = {
    "generate": generate.main,
    "check": check.main,
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print(f"kexnav: unknown command {command!r}\n")
        print(__doc__)
        return 2
    return COMMANDS[command](rest)


if __name__ == "__main__":
    sys.exit(main())

import argparse
from pathlib import Path
import sys

from uncertainties import ufloat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("s0", type=float)
    parser.add_argument("s0_err", type=float)
    parser.add_argument("s1", type=float)
    parser.add_argument("s1_err", type=float)
    parser.add_argument("--out", "-o", type=Path)
    args = parser.parse_args()

    s0 = ufloat(args.s0, args.s0_err)
    s1 = ufloat(args.s1, args.s1_err)
    rho = s1 / s0

    print(f"{rho=}")

    if args.out:
        with open(args.out, "w") as f:
            f.write(f"Command: {sys.argv}\nrho: {rho}")


if __name__ == "__main__":
    main()

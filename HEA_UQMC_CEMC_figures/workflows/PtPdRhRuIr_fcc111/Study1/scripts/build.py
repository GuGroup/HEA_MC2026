#!/usr/bin/env python3
"""Build the C++ UQ/CEMC executable and the random-slab reconstruction helper."""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def compile_one(cxx: str, flags: list[str], src: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [cxx, *flags, "-o", str(out), str(src)]
    print("Compiling:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Wrote", out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="src/uq_cemc_mpi.cpp")
    parser.add_argument("--output", default="build/uq_cemc_mpi")
    parser.add_argument("--cxx", default=None, help="Override compiler for main executable, e.g. mpicxx or g++")
    parser.add_argument("--no-mpi", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--skip-helper", action="store_true", help="Do not build build/reconstruct_random_slab")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    src = (root / args.source).resolve()
    out = (root / args.output).resolve()

    if args.cxx:
        cxx = args.cxx
    elif not args.no_mpi and shutil.which("mpiicpc"):
        cxx = "mpiicpc"
    elif not args.no_mpi and shutil.which("mpicxx"):
        cxx = "mpicxx"
    else:
        cxx = "g++"

    flags = ["-std=c++17", "-O0" if args.debug else "-O3", "-Wall", "-Wextra"]
    mpi_wrappers = {"mpicxx", "mpic++", "mpiicpc", "mpiicpc"}
    if Path(cxx).name in mpi_wrappers:
        flags.append("-DUSE_MPI")
    if not args.debug:
        flags += ["-march=native"]

    compile_one(cxx, flags, src, out)

    if not args.skip_helper:
        helper_src = (root / "src/reconstruct_random_slab.cpp").resolve()
        helper_out = (root / "build/reconstruct_random_slab").resolve()
        helper_cxx = cxx
        helper_flags = ["-std=c++17", "-O0" if args.debug else "-O3", "-Wall", "-Wextra"]
        if not args.debug:
            helper_flags += ["-march=native"]
        compile_one(helper_cxx, helper_flags, helper_src, helper_out)


if __name__ == "__main__":
    main()

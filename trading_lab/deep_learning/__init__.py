"""Deep Learning helpers for V10.

The data/sequence utilities intentionally do not require PyTorch so the core
project and its test suite can run without the optional DL dependency.
Install ``requirements_dl.txt`` before importing models/train components.
"""
from .dataset import SequenceBundle, build_sequence_bundle

__all__ = ["SequenceBundle", "build_sequence_bundle"]

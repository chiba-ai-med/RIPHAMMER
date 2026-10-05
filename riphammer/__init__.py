"""RIPHAMMER: Multi-band NMF for cell-type-specific TAD extraction from bulk Hi-C."""

__version__ = "0.1.0"

from riphammer.core import MultibandNMF
from riphammer.io import mcool_to_features, w_to_cool
from riphammer.tad import call_tads, compare_boundaries

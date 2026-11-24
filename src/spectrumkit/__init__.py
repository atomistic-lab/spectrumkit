"""spectrumkit: Compute spectra from MD simulation data."""

__authors__ = "MAICoS Developer Team"

from ._version import __version__  # noqa: F401
from .auxiliary import XYZReader
from .dielectricspectrum import (
    DielectricSpectrum,
    WienerKhinchin,
    calculate_spectrum_from_current,
    calculate_spectrum_from_dipole,
)

__all__ = [
    "DielectricSpectrum",
    "WienerKhinchin",
    "XYZReader",
    "calculate_spectrum_from_current",
    "calculate_spectrum_from_dipole",
]

# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Auxiliary readers attaching non-coordinate data to MDAnalysis trajectories.

Importing this module registers the readers with
:func:`MDAnalysis.auxiliary.core.auxreader`, so that files are recognised by
their extension in :meth:`~MDAnalysis.coordinates.base.ReaderBase.add_auxiliary`.
"""

from .xyz import XYZReader

__all__ = ["XYZReader"]

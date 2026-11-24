# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Auxiliary reader for extended XYZ files.

Background
----------

Extended XYZ (extxyz) files, as written by ASE_ and i-PI_, store arbitrary data next to
the coordinates:

* per-atom properties as extra columns, declared by the ``Properties`` key of the
  comment line, e.g. ``Properties=species:S:1:pos:R:3:forces:R:3``,
* per-frame properties as ``key=value`` pairs on the comment line, e.g.
  ``energy=-1.2 polarization="0.1 0.2 0.3"``.

The coordinate reader :class:`MDAnalysis.coordinates.XYZ.XYZReader` only reads the
positions. The :class:`XYZReader` of this module reads **everything else**: every
per-atom property except ``species`` and ``pos`` and every per-frame key. It makes them
available as MDAnalysis auxiliary data, for example Born effective charges, forces,
energies, dipoles or polarizabilities.

The comment line is parsed with the rules of ASE's
:func:`ase.io.extxyz.key_val_str_to_dict`: values become integers, floats or booleans
(or arrays of them) where possible and stay strings otherwise. ``virial`` and ``stress``
are reshaped to 3x3 matrices in Fortran order, and ``Lattice`` to a 3x3 matrix whose
rows are the cell vectors (like ``atoms.cell`` in ASE). Per-atom properties keep the
type declared in ``Properties`` (``R``: float, ``I``: int, ``L``: bool, ``S``: str). A
property with one column has shape ``(n_atoms,)``, one with ``n`` columns
``(n_atoms, n)``.

A file without a ``Properties`` key is read as plain XYZ: like for the coordinate
reader, its comment line is a free-text title and ignored, and all columns after the
positions are read as one per-atom term ``data``.

Standalone usage
----------------

All data is read when the reader is created and stored in :attr:`XYZReader.data_dict`,
a dictionary that maps each term to an array with the frames along the first axis::

    from spectrumkit.auxiliary import XYZReader

    aux = XYZReader("ipi.becx_0.extxyz", dt=0.0005)
    aux.terms  # ['becx', 'Lattice', 'ipi_comment', 'pbc']
    aux.data_dict["becx"].shape  # (n_frames, n_atoms, 3)

Adding data to trajectories
---------------------------

Like other auxiliary readers, :class:`XYZReader` attaches its data to the trajectory
frames with matching time. The ``aux_spec`` dictionary maps the name used in ``ts.aux``
to a term of the file::

    import MDAnalysis as mda
    import spectrumkit  # registers the EXTXYZ and XYZ auxiliary formats

    u = mda.Universe("ipi.positions_0.extxyz", format="XYZ", dt=0.002)
    u.trajectory.add_auxiliary({"becx": "becx"}, "ipi.becx_0.extxyz", dt=0.0005)
    u.trajectory.ts.aux.becx  # (n_atoms, 3)

Without ``aux_spec`` all terms are attached under their own names::

    u.trajectory.add_auxiliary(auxdata=XYZReader("forces.extxyz", dt=0.0005))

.. note::
    Extended XYZ has no standard time key. Unless a ``time_selector`` names a per-frame
    key holding the time in ps, the time of step ``i`` is ``initial_time + i * dt``,
    and ``dt`` defaults to 1 ps. Pass the actual ``dt`` of the file, here 0.5 fs for
    i-PI output written every step. The coordinate reader needs its own ``dt`` as well.

.. note::
    Atoms are matched by index: the file MUST list the atoms in the same order as the
    topology.

.. note::
    Units are not converted. ``represent_ts_as="average"`` works only for numeric
    terms.

.. _ASE: https://wiki.fysik.dtu.dk/ase/ase/io/formatoptions.html#extxyz
.. _i-PI: https://docs.ipi-code.org
"""

import itertools
import re
import weakref
from pathlib import Path

import numpy as np
from MDAnalysis.auxiliary import _AUXREADERS, base
from MDAnalysis.lib.util import anyopen

# ``key``, ``key=value``, ``key="quoted value"``, ``key={...}``, ``key=[...]``
_KEY_VALUE = re.compile(
    r"""([^\s="'{}\[\]]+)
        (?:\s*=\s*(?:"([^"]*)"|'([^']*)'|\{([^}]*)\}|\[([^\]]*)\]|([^\s"'{}\[\]]+)))?
    """,
    re.VERBOSE,
)
_TOKEN = re.compile(r"[^\s,]+")
_STR_TO_BOOL = {
    "T": True,
    "F": False,
    "true": True,
    "false": False,
    "True": True,
    "False": False,
    "TRUE": True,
    "FALSE": False,
}
# Same sets as ``ase.io.extxyz``
_UNPROCESSED_KEYS = {"uid"}
_SPECIAL_3_3_KEYS = {"Lattice", "virial", "stress"}

# Read by MDAnalysis' XYZ topology parser and coordinate reader
_COORDINATE_FIELDS = ("species", "pos")
_PLAIN_XYZ_PROPERTIES = "species:S:1:pos:R:3"
# ``int`` and ``float`` values of a per-frame key may mix, other types may not
_KINDS = {"b": "bool", "i": "number", "u": "number", "f": "number"}


def _convert_tokens(tokens: list[str]) -> list | None:
    # Python's int/float per token is ~5x faster than np.array(tokens, dtype)
    for convert in (int, float):
        try:
            return [convert(token) for token in tokens]
        except ValueError:
            continue
    if all(token in _STR_TO_BOOL for token in tokens):
        return [_STR_TO_BOOL[token] for token in tokens]
    return None


def _convert_value(key: str, value: str) -> object:
    if key.lower() in _UNPROCESSED_KEYS:
        return value
    converted: object = value
    values = _convert_tokens(_TOKEN.findall(value))
    if values is not None:
        converted = values[0] if len(values) == 1 else np.array(values)

    if key in _SPECIAL_3_3_KEYS:
        if not isinstance(converted, np.ndarray) or converted.shape != (9,):
            raise ValueError(
                f"Expected 9 numbers for the 3x3 matrix '{key}', got {value!r}"
            )
        converted = converted.reshape((3, 3), order="F")
    return converted


def parse_comment_line(line: str) -> dict:
    """Parse the ``key=value`` pairs of an extended XYZ comment line.

    Follows :func:`ase.io.extxyz.key_val_str_to_dict`: values may be delimited by
    quotes, curly or square brackets. Values are converted to int, float or bool, or to
    arrays of them, and stay strings otherwise. A key without value is ``True``.
    ``Lattice``, ``virial`` and ``stress`` are reshaped to 3x3 matrices in Fortran
    order. Escaped characters, malformed quoting and JSON values are not supported.

    Parameters
    ----------
    line : str
        Comment (second) line of an extended XYZ frame.

    Returns
    -------
    dict
        Parsed values by key, in the order of the line.

    Raises
    ------
    ValueError
        If ``Lattice``, ``virial`` or ``stress`` is not a vector of 9 numbers.
    """
    info = {}
    for match in _KEY_VALUE.finditer(line):
        key, *values = match.groups()
        value = next((v for v in values if v is not None), "T")
        info[key] = _convert_value(key, value)
    return info


def parse_properties(spec: str) -> list[tuple[str, str, int]]:
    """Parse the ``Properties`` value of an extended XYZ comment line.

    Parameters
    ----------
    spec : str
        Colon-separated ``name:type:ncols`` triplets, e.g. ``"species:S:1:pos:R:3"``.
        ``type`` is one of ``R`` (real), ``I`` (integer), ``S`` (string) and ``L``
        (logical).

    Returns
    -------
    list of tuple(str, str, int)
        ``(name, type, ncols)`` of every property, in column order.

    Raises
    ------
    ValueError
        If ``spec`` is not a sequence of valid triplets.
    """
    fields = spec.split(":")
    if len(fields) % 3:
        raise ValueError(f"Properties={spec!r} is not a list of name:type:ncols")
    properties = []
    for name, kind, ncols in zip(fields[::3], fields[1::3], fields[2::3], strict=True):
        if kind not in ("R", "I", "S", "L"):
            raise ValueError(f"Unknown type {kind!r} of property {name!r}")
        if not ncols.isdigit() or int(ncols) < 1:
            raise ValueError(f"Invalid number of columns {ncols!r} of {name!r}")
        properties.append((name, kind, int(ncols)))
    return properties


def _convert_column(block: np.ndarray, name: str, kind: str) -> np.ndarray:
    try:
        if kind == "R":
            return block.astype(np.float64)
        if kind == "I":
            return block.astype(np.int64)
    except ValueError as err:
        raise ValueError(f"Property '{name}': {err}") from None
    if kind == "L":
        is_true = np.isin(block, ("T", "True"))
        if not (is_true | np.isin(block, ("F", "False"))).all():
            raise ValueError(f"Property '{name}' has values that are not T/F")
        return is_true
    return block.copy()


class XYZStep(base.AuxStep):
    """:class:`~MDAnalysis.auxiliary.base.AuxStep` for extended XYZ files.

    The data of each step is a dictionary of all terms of the file; time and data of
    interest are selected by term name.

    Parameters
    ----------
    time_selector : str, optional
        Per-frame key holding the time in ps. Default ``None``: the time is calculated
        from ``dt`` and ``initial_time``.
    data_selector : str, optional
        Term to be stored in ``data``. Default ``None``: ``data`` is the dictionary of
        all terms.
    **kwargs
        Other AuxStep options.

    See Also
    --------
    :class:`MDAnalysis.auxiliary.base.AuxStep`
    """

    def _select_time(self, key: str | None) -> np.float64 | None:
        return self._select_data(key)

    def _select_data(self, key: str | None) -> np.ndarray | None:
        if key is None:
            return None
        try:
            return self._data[key]
        except KeyError:
            raise KeyError(
                f"'{key}' is not a term of the extended XYZ file. "
                f"Available terms: {list(self._data)}"
            ) from None

    def _empty_data(self) -> np.ndarray | dict[str, np.ndarray]:
        if self._data_selector is None:
            return {
                term: np.full(np.shape(value), np.nan)
                for term, value in self._data.items()
            }
        return np.full(np.shape(self.data), np.nan)


class _ExtXYZFile:
    """All terms of an extended XYZ file, parsed once and shared by readers."""

    def __init__(self, filename: Path):
        with anyopen(str(filename)) as xyz_file:
            lines = xyz_file.readlines()
        starts = self._frame_starts(lines)
        self.n_steps = len(starts)

        infos = [parse_comment_line(lines[start + 1]) for start in starts]
        properties = self._properties(infos, lines[starts[0] + 2])

        self.data_dict: dict[str, np.ndarray] = {}
        self._read_atom_columns(lines, starts, properties)
        self._read_frame_keys(infos)
        # read-only, because all readers of the file share these arrays
        for array in self.data_dict.values():
            array.flags.writeable = False

    def _frame_starts(self, lines: list[str]) -> list[int]:
        """Return the index of the first line of every frame."""
        starts: list[int] = []
        position = 0
        # frames end at the first blank line (XYZWriter appends one) or at EOF
        while position < len(lines) and lines[position].strip():
            frame = len(starts)
            try:
                n_atoms = int(lines[position])
            except ValueError:
                raise ValueError(
                    f"Frame {frame}: expected the number of atoms, "
                    f"got {lines[position]!r}"
                ) from None
            if frame == 0:
                self.n_atoms = n_atoms
            elif n_atoms != self.n_atoms:
                raise ValueError(
                    f"Frame {frame} has {n_atoms} atoms, frame 0 has {self.n_atoms}"
                )
            if position + n_atoms + 2 > len(lines):
                raise ValueError(f"Frame {frame} is truncated")
            starts.append(position)
            position += n_atoms + 2
        if not starts:
            raise ValueError("No frames found")
        for line_index in range(position, len(lines)):
            if lines[line_index].strip():
                raise ValueError(
                    f"Unexpected content after a blank line, line {line_index + 1}"
                )
        return starts

    @staticmethod
    def _properties(infos: list[dict], first_atom_line: str) -> list:
        """Return the per-atom properties shared by all frames."""
        specs = [info.pop("Properties", None) for info in infos]
        for frame, spec in enumerate(specs):
            if spec != specs[0]:
                raise ValueError(
                    f"Frame {frame} has Properties={spec}, frame 0 has {specs[0]}"
                )
        if specs[0] is not None:
            return parse_properties(specs[0])
        # plain XYZ: the comment line is a free-text title, not key=value pairs
        for info in infos:
            info.clear()
        properties = parse_properties(_PLAIN_XYZ_PROPERTIES)
        n_extra = len(first_atom_line.split()) - 4
        if n_extra > 0:
            properties.append(("data", "R", n_extra))
        return properties

    def _read_atom_columns(
        self, lines: list[str], starts: list[int], properties: list
    ) -> None:
        n_columns = sum(ncols for _, _, ncols in properties)
        atom_lines = itertools.chain.from_iterable(
            lines[start + 2 : start + 2 + self.n_atoms] for start in starts
        )
        # one split of all atom lines is ~3x faster than splitting line by line
        tokens = np.array("".join(atom_lines).split())
        if tokens.size != len(starts) * self.n_atoms * n_columns:
            raise ValueError(
                f"Atom lines do not have the {n_columns} columns given by "
                f"Properties {properties}"
            )
        tokens = tokens.reshape(len(starts), self.n_atoms, n_columns)

        first_column = 0
        for name, kind, ncols in properties:
            block = tokens[..., first_column : first_column + ncols]
            first_column += ncols
            if name in _COORDINATE_FIELDS:
                continue
            values = _convert_column(block, name, kind)
            self.data_dict[name] = values[..., 0] if ncols == 1 else values

    def _read_frame_keys(self, infos: list[dict]) -> None:
        keys = dict.fromkeys(key for info in infos for key in info)
        for key in keys:
            if key in self.data_dict:
                raise ValueError(f"'{key}' is both a per-atom and a per-frame term")
            values = [info.get(key) for info in infos]
            missing = [frame for frame, value in enumerate(values) if value is None]
            if missing:
                present = next(value for value in values if value is not None)
                if not np.issubdtype(np.asarray(present).dtype, np.number):
                    raise ValueError(
                        f"Non-numeric per-frame key '{key}' is missing in frames "
                        f"{missing}"
                    )
                for frame in missing:
                    values[frame] = np.full(np.shape(present), np.nan)
            kinds = {
                _KINDS.get(np.asarray(value).dtype.kind, "str") for value in values
            }
            if len(kinds) > 1:
                raise ValueError(
                    f"Per-frame key '{key}' changes its type between frames: {kinds}"
                )
            try:
                array = np.array(values)
            except ValueError:
                raise ValueError(
                    f"Per-frame key '{key}' changes its shape between frames"
                ) from None
            if key == "Lattice":
                # rows are the cell vectors, like ASE's ``atoms.cell``
                array = array.transpose(0, 2, 1)
            self.data_dict[key] = array


_PARSED_FILES: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


class XYZReader(base.AuxReader):
    """Auxiliary reader for extended XYZ files.

    All data of the file is read and stored on initialisation. Readers of the same
    unchanged file share one parse. See :mod:`spectrumkit.auxiliary.xyz` for the
    conventions.

    Parameters
    ----------
    filename : str or pathlib.Path
        Location of the extended XYZ file, optionally compressed (``.gz``, ``.bz2``).
    **kwargs
        Other AuxReader options, e.g. ``dt``, ``initial_time`` or ``time_selector``.

    Attributes
    ----------
    data_dict : dict
        Maps each term to a NumPy array with the frames along the first axis: per-atom
        terms have shape ``(n_steps, n_atoms[, ncols])``, per-frame terms
        ``(n_steps, ...)``.
    terms : list of str
        Names of all terms in :attr:`data_dict`: the per-atom terms in column order,
        followed by the per-frame terms in the order of the comment line.

    Raises
    ------
    ValueError
        If the frames differ in the number of atoms or in ``Properties``, if a frame is
        truncated, if the atom columns don't match ``Properties`` or if a per-frame key
        changes its shape or is missing from frames while not being numeric.

    See Also
    --------
    :class:`MDAnalysis.auxiliary.base.AuxReader`
    :meth:`MDAnalysis.coordinates.base.ReaderBase.add_auxiliary`

    Note
    ----
    The file is assumed to be of a size such that reading and storing the full contents
    is practical. The arrays of :attr:`data_dict` are read-only, since readers of the
    same file share them.
    """

    format = "EXTXYZ"
    _Auxstep = XYZStep

    def __init__(self, filename: str | Path, **kwargs):
        self._auxdata = Path(filename).resolve()
        stat = self._auxdata.stat()
        key = (self._auxdata, stat.st_mtime_ns, stat.st_size)
        # ``attach_auxiliary`` creates a new reader for every attached term; they share
        # one parse instead of re-reading the file each time.
        parsed = _PARSED_FILES.get(key)
        if parsed is None:
            parsed = _PARSED_FILES[key] = _ExtXYZFile(self._auxdata)
        self._parsed = parsed  # keeps the weakly cached parse alive

        self.data_dict = dict(parsed.data_dict)
        self.terms = list(self.data_dict)
        self._n_steps = parsed.n_steps
        super().__init__(**kwargs)

    def _memory_usage(self) -> int:
        return sum(array.nbytes for array in self.data_dict.values())

    def _read_next_step(self) -> XYZStep:
        """Read next auxiliary step and update ``auxstep``.

        Returns
        -------
        XYZStep
            Updated with the data for the new step.

        Raises
        ------
        StopIteration
            When end of auxiliary data set is reached.
        """
        auxstep = self.auxstep
        new_step = self.step + 1
        if new_step < self.n_steps:
            auxstep._data = {
                term: values[new_step] for term, values in self.data_dict.items()
            }
            auxstep.step = new_step
            return auxstep
        self.rewind()
        raise StopIteration

    def _go_to_step(self, i: int) -> XYZStep:
        """Move to and read i-th auxiliary step.

        Parameters
        ----------
        i : int
            Step number (0-indexed) to move to

        Returns
        -------
        XYZStep

        Raises
        ------
        ValueError
            If step index not in valid range.
        """
        if i >= self.n_steps or i < 0:
            raise ValueError(
                f"Step index {i} is not valid for auxiliary (num. steps {self.n_steps})"
            )
        self.auxstep.step = i - 1
        self.next()
        return self.auxstep

    def read_all_times(self) -> np.ndarray:
        """Get NumPy array of time at each step.

        Returns
        -------
        numpy.ndarray
            Time at each step.
        """
        if self.time_selector is not None:
            return self.data_dict[self.time_selector]
        return np.arange(self.n_steps) * self.dt + self.initial_time


# ``.xyz`` files are guessed as format "XYZ"
_AUXREADERS["XYZ"] = XYZReader

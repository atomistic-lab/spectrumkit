# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the extended XYZ auxiliary reader."""

import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
from ase.io import read
from ase.io.extxyz import key_val_str_to_dict
from MDAnalysisTests.auxiliary.base import BaseAuxReaderTest, BaseAuxReference
from MDAnalysisTests.datafiles import COORDINATES_TOPOLOGY, COORDINATES_XTC
from numpy.testing import assert_equal

from spectrumkit.auxiliary.xyz import (
    XYZReader,
    XYZStep,
    parse_comment_line,
    parse_properties,
)

sys.path.append(str(Path(__file__).parents[1]))
from data import AUX_EXTXYZ

TERMS = [
    "forces",
    "bec",
    "charges",
    "tag",
    "fixed",
    "site",
    "Lattice",
    "time",
    "energy",
    "polarization",
    "polarizability",
    "stress",
    "pbc",
    "label",
]

PLAIN_XYZ = "2\nfree text title\nO 0 0 0 1.0 2.0\nH 1 0 0 3.0 4.0\n"


class XYZReference(BaseAuxReference):
    """Reference values of ``AUX_EXTXYZ`` for :class:`BaseAuxReaderTest`.

    The per-frame term ``polarization`` holds ``[i, 2*i, 2**i]`` and ``time`` holds
    ``i``, the values the base tests expect.
    """

    def __init__(self):
        super().__init__()
        self.testdata = str(AUX_EXTXYZ)
        self.reader = XYZReader
        self.description["auxdata"] = AUX_EXTXYZ.resolve()
        self.description["format"] = self.reader.format
        self.description["data_selector"] = "polarization"

        def reference_auxstep(i):
            auxstep = XYZStep(
                dt=self.dt, initial_time=self.initial_time, data_selector="polarization"
            )
            auxstep.step = i
            auxstep._data = {"polarization": self.format_data([i, 2 * i, 2**i])}
            return auxstep

        self.auxsteps = [reference_auxstep(i) for i in range(self.n_steps)]
        self.iter_list_auxsteps = [self.auxsteps[1], self.auxsteps[3]]
        self.iter_slice_auxsteps = self.auxsteps[::2]

        self.time_selector = "time"
        self.data_selector = "polarization"
        self.select_data_ref = [
            self.format_data([i, 2 * i, 2**i]) for i in range(self.n_steps)
        ]


# COORDINATES_TOPOLOGY has no elements
@pytest.mark.filterwarnings("ignore:Element information is missing:UserWarning")
class TestXYZReader(BaseAuxReaderTest):
    """Run the generic AuxReader tests of MDAnalysis on :class:`XYZReader`."""

    @pytest.fixture
    def ref(self):
        """Reference values of the test file."""
        return XYZReference()

    @pytest.fixture
    def ref_universe(self, ref):
        """Universe with ``polarization`` attached as ``test``."""
        u = mda.Universe(COORDINATES_TOPOLOGY, COORDINATES_XTC)
        u.trajectory.add_auxiliary({"test": "polarization"}, ref.testdata)
        return u

    @pytest.fixture
    def reader(self, ref):
        """Reader selecting ``polarization``."""
        return ref.reader(
            ref.testdata,
            initial_time=ref.initial_time,
            dt=ref.dt,
            auxname=ref.name,
            data_selector="polarization",
        )

    def test_terms(self, reader):
        """Test that all but ``species`` and ``pos`` are terms, in file order."""
        assert reader.terms == TERMS

    # ase.Atoms sets ndarray.shape, deprecated in NumPy 2.5
    @pytest.mark.filterwarnings("ignore:Setting the shape:DeprecationWarning")
    @pytest.mark.parametrize("term", TERMS)
    def test_term_matches_ase(self, reader, term):
        """Test that every term equals the value ASE reads."""
        for i, atoms in enumerate(read(AUX_EXTXYZ, index=":")):
            ase_terms = {
                **atoms.info,
                **atoms.arrays,
                **atoms.calc.results,
                "Lattice": atoms.cell.array,
                "pbc": atoms.pbc,
                "stress": atoms.get_stress(voigt=False),
            }
            assert_equal(reader.data_dict[term][i], ase_terms[term])

    @pytest.mark.parametrize("time_selector", [None, "time"])
    def test_read_all_times(self, ref, time_selector):
        """Test times from ``dt`` and from a time key."""
        reader = ref.reader(ref.testdata, time_selector=time_selector)
        assert_equal(reader.read_all_times(), np.arange(ref.n_steps))

    def test_invalid_data_selector_raises(self, reader):
        """Test that an unknown term lists the available ones."""
        with pytest.raises(KeyError, match="Available terms"):
            reader.data_selector = "nope"

    def test_data_is_read_only(self, reader):
        """Test that the arrays shared between readers can't be changed."""
        with pytest.raises(ValueError, match="read-only"):
            reader.data_dict["bec"][0, 0, 0] = 1

    def test_attach_all_terms(self, ref):
        """Test attaching all terms without ``aux_spec``."""
        reader = ref.reader(ref.testdata)
        u = mda.Universe(COORDINATES_TOPOLOGY, COORDINATES_XTC)
        u.trajectory.add_auxiliary(auxdata=reader)
        u.trajectory[3]
        for term in TERMS:
            assert_equal(u.trajectory.ts.aux[term], reader.data_dict[term][3])

    def test_empty_data_all_terms(self, ref):
        """Test NaN for every term of a frame without auxiliary step."""
        reader = ref.reader(ref.testdata, auxname=ref.name)
        # no auxiliary step lies within this timestep
        ts = mda.coordinates.base.Timestep(5, dt=0.5)
        ts.frame = 1
        reader.update_ts(ts)
        for term in TERMS:
            assert np.isnan(ts.aux.test[term]).all()

    def test_n_atoms_mismatch_raises(self, tmp_path):
        """Test attaching per-atom data of another system."""
        path = tmp_path / "plain.xyz"
        path.write_text(PLAIN_XYZ)
        u = mda.Universe(COORDINATES_TOPOLOGY, COORDINATES_XTC)
        with pytest.raises(ValueError, match=r"\['data'\] have 2 atoms"):
            u.trajectory.add_auxiliary({"data": "data"}, str(path))

    def test_changed_file_is_read_again(self, ref, tmp_path):
        """Test that the shared parse is renewed for a changed file."""
        path = tmp_path / "aux.extxyz"
        content = AUX_EXTXYZ.read_text()
        path.write_text(content)
        first = ref.reader(path)
        path.write_text(content.replace("energy=-1.5 ", "energy=18.5 "))
        assert_equal(ref.reader(path).data_dict["energy"][1], 18.5)
        assert_equal(first.data_dict["energy"][1], -1.5)


def test_plain_xyz(tmp_path):
    """Test extra columns of a plain XYZ file."""
    path = tmp_path / "plain.xyz"
    path.write_text(PLAIN_XYZ)
    reader = XYZReader(path)
    assert reader.terms == ["data"]
    assert_equal(reader.data_dict["data"], [[[1.0, 2.0], [3.0, 4.0]]])


def test_missing_numeric_key_is_nan(tmp_path):
    """Test NaN for a numeric key missing in a frame."""
    path = tmp_path / "missing.extxyz"
    path.write_text(
        "1\nProperties=species:S:1:pos:R:3 energy=1\nO 0 0 0\n"
        "1\nProperties=species:S:1:pos:R:3\nO 0 0 0\n"
    )
    assert_equal(XYZReader(path).data_dict["energy"], [1.0, np.nan])


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("1\n\nO 0 0 0\n2\n\nO 0 0 0\nH 0 0 0\n", "Frame 1 has 2 atoms"),
        ("\n", "No frames found"),
        ("2\n\nO 0 0 0\n", "Frame 0 is truncated"),
        ("1\n\nO 0 0 0\n\n1\n\nO 0 0 0\n", "after a blank line, line 5"),
        ("one\n\nO 0 0 0\n", "expected the number of atoms"),
        (
            (
                "1\nProperties=species:S:1:pos:R:3\nO 0 0 0\n"
                "1\nProperties=species:S:1:pos:R:3:q:R:1\nO 0 0 0 1\n"
            ),
            "Frame 1 has Properties",
        ),
        ("1\nProperties=species:S:1:pos:R:3:q:R:1\nO 0 0 0\n", "columns given"),
        ("1\nProperties=species:S:1:pos:R:3:q:I:1\nO 0 0 0 1.5\n", "Property 'q'"),
        ("1\nProperties=species:S:1:pos:R:3:q:L:1\nO 0 0 0 1\n", "not T/F"),
        (
            (
                "1\nProperties=species:S:1:pos:R:3 a=1\nO 0 0 0\n"
                '1\nProperties=species:S:1:pos:R:3 a="1 2"\nO 0 0 0\n'
            ),
            "'a' changes its shape",
        ),
        (
            (
                "1\nProperties=species:S:1:pos:R:3 a=x\nO 0 0 0\n"
                "1\nProperties=species:S:1:pos:R:3\nO 0 0 0\n"
            ),
            r"'a' is missing in frames \[1\]",
        ),
        (
            (
                "1\nProperties=species:S:1:pos:R:3 a=1.5\nO 0 0 0\n"
                "1\nProperties=species:S:1:pos:R:3 a=x\nO 0 0 0\n"
            ),
            "'a' changes its type",
        ),
        (
            "1\nProperties=species:S:1:pos:R:3:a:R:1 a=1\nO 0 0 0 1\n",
            "'a' is both",
        ),
    ],
)
def test_invalid_file_raises(tmp_path, content, match):
    """Test errors for inconsistent or malformed files."""
    path = tmp_path / "invalid.extxyz"
    path.write_text(content)
    with pytest.raises(ValueError, match=match):
        XYZReader(path)


@pytest.mark.parametrize(
    "line",
    [
        "a=1",
        "a=-1.5e3",
        'a="1 2 3"',
        "a={1 2 3}",
        "a=[1, 2, 3]",
        "a = 2",
        "a=T",
        'a="T T F"',
        "flag",
        "a=hello",
        'a="hello world"',
        "uid=007",
        'stress="1 2 3 4 5 6 7 8 9"',
        'Lattice="1 2 3 4 5 6 7 8 9" pbc="T T T"',
        'ipi_comment="Step:  0  Bead:  0 becx{ase}  cell{ase}" b=1.5',
    ],
)
def test_parse_comment_line_matches_ase(line):
    """Test comment line parsing against ASE."""
    assert_equal(parse_comment_line(line), key_val_str_to_dict(line))


@pytest.mark.parametrize(
    ("parse", "arg"),
    [
        (parse_comment_line, "stress=1"),
        (parse_properties, "species:S"),
        (parse_properties, "a:X:1"),
        (parse_properties, "a:R:0"),
    ],
)
def test_invalid_header_raises(parse, arg):
    """Test errors for invalid header values."""
    with pytest.raises(ValueError):  # noqa: PT011
        parse(arg)

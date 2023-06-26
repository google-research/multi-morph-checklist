# coding=utf-8
# Copyright 2023 The Google Research authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for utils."""
from absl.testing import absltest
from multi_morph_checklist import core
from multi_morph_checklist import languages


class UtilsTest(absltest.TestCase):

  def test_frozensets_contain(self):
    """Test for template in English."""

    t = languages.EN.template(
        "{pa} {pb.<pa.GENDER>}",
        pa=[{"FEM": "123"}, {"MASC": "456"}],
        pb=[{"FEM": "1", "MASC": "4"}],
        verbose=True,
    )
    self.assertIsInstance(t, core.Template)
    samples = t(2, export=False)
    self.assertLen(samples, 2)
    self.assertCountEqual(samples, ["123 1", "456 4"])
    self.assertEqual(
        str(languages.EN.dimensions._name_to_dim["STARTSWITH"]),
        "'STARTSWITH': ['VOW', 'CONS'] (sampleable: True)",
    )
    self.assertEqual(
        str(languages.EN.dimensions._name_to_dim["ORDER"]),
        "'ORDER': ['GT', 'LT'] (sampleable: False)",
    )
    self.assertEqual(
        languages.SV.dimensions._name_to_dim["ENDSWITH"]("tjena"), "VOW"
    )
    self.assertEqual(
        languages.FR.dimensions._name_to_dim["STARTSWITH"]("air"), "VOW"
    )
    self.assertEqual(
        str(languages.ZH.dimensions._name_to_dim["CLASSIFIER_TYPE"]),
        "'CLASSIFIER_TYPE': ['GENERIC', 'PAIR', 'ANIMAL', 'LONG_NARROW',"
        " 'FLAT', 'SHEET', 'CLOTHING'] (sampleable: False)",
    )
    self.assertEqual(
        str(languages.SV.dimensions._name_to_dim["WITHARTICLE"]),
        "'WITHARTICLE': ['WITH_ART', 'NO_ART'] (sampleable: False)",
    )
    self.assertEqual(
        str(languages.SV.dimensions._name_to_dim["ENDSWITH"]),
        "'ENDSWITH': ['VOW', 'CONS'] (sampleable: False)",
    )
    self.assertEqual(
        str(languages.RU.dimensions._name_to_dim["STARTSWITH2C"]),
        "'STARTSWITH2C': ['NO_CONS2', 'CONS2'] (sampleable: False)",
    )
    self.assertEqual(
        str(languages.RU.dimensions._name_to_dim["STARTSWITH2C"]),
        "'STARTSWITH2C': ['NO_CONS2', 'CONS2'] (sampleable: False)",
    )
    self.assertEqual(
        str(languages.IT.dimensions._name_to_dim["STARTSWITH"]),
        "'STARTSWITH': ['VOW', 'CONS', 'CONS2'] (sampleable: False)",
    )
    self.assertEqual(
        str(languages.SK.dimensions._name_to_dim["CASE"]),
        "'CASE': ['NOM', 'GEN', 'DAT', 'ACC', 'LOK', 'INS'] (sampleable:"
        " False)",
    )
    self.assertEqual(
        str(languages.AR.dimensions._name_to_dim["NBPLACE"]),
        "'NBPLACE': ['BEF', 'AFT'] (sampleable: False)",
    )
    self.assertEqual(
        str(languages.FR.dimensions._name_to_dim["DIRECTIVE"]),
        "'DIRECTIVE': ['DIRECT', 'INDIRECT'] (sampleable: False)",
    )


if __name__ == "__main__":
  absltest.main()

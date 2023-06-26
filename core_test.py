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
"""Tests for core."""
from absl.testing import absltest
from multi_morph_checklist import core


class CoreTest(absltest.TestCase):
  dim1 = core.Dimension("d1", ["A", "B"])
  dim2 = core.Dimension("d2", ["C", "D"])
  dims = core.Dimensions([dim1, dim2])
  r1 = core.Rule("r1", dims, {"A": "1", "B": "2"})

  def test_dimension(self):
    self.assertEqual("d1", self.dim1.name)
    self.assertEqual(["A", "B"], self.dim1.features)
    self.assertTrue(self.dim1.sampleable)
    self.assertEqual("'d1': ['A', 'B'] (sampleable: True)", str(self.dim1))

  def test_dimension_with_call(self):
    dim = core.Dimension(
        "test_call", ["1", "0"], False, lambda s: "1" if s else "0"
    )
    self.assertEqual("test_call", dim.name)
    self.assertEqual(["1", "0"], dim.features)
    self.assertEqual("1", dim("123"))
    self.assertIsNone(dim(""))
    self.assertFalse(dim.sampleable)

  def test_dimensions(self):
    self.assertEqual({"d1": self.dim1, "d2": self.dim2}, self.dims._name_to_dim)
    self.assertEqual(
        {"d1": ["A", "B"], "d2": ["C", "D"]}, self.dims.name_to_features
    )
    self.assertEqual(
        {
            "A": "d1",
            "B": "d1",
            "C": "d2",
            "D": "d2",
        },
        self.dims.feature_to_name,
    )

    self.assertCountEqual(
        [self.dim1, self.dim2],
        self.dims.get_dimensions_from_features(["A", "C"]),
    )
    with self.assertRaises(KeyError):
      self.dims("d3")
    self.assertIn("d1", self.dims)
    self.assertNotIn("d3", self.dims)
    self.assertEqual(
        "'d1': ['A', 'B'] (sampleable: True)\n'd2': ['C', 'D'] (sampleable:"
        " True)\n",
        str(self.dims),
    )

  def test_rules(self):
    rules = {"A.C": "1", "A.D": "2", "B": "3"}
    rule = core.Rule("R.X", self.dims, rules)
    self.assertEqual("R.X", rule.name)
    self.assertEqual("R", rule.code)
    self.assertEqual(
        {
            frozenset(["A", "C"]): "1",
            frozenset(["A", "D"]): "2",
            frozenset(["B"]): "3",
            frozenset(["B", "C"]): "3",
            frozenset(["B", "D"]): "3",
        },
        rule.rules,
    )
    self.assertEqual(frozenset(["A", "B"]), rule._to_set("A.B"))
    self.assertEqual(frozenset(["A", "B"]), rule._to_set(frozenset(["A", "B"])))
    self.assertCountEqual(
        [
            frozenset(["A", "C"]),
            frozenset(["A", "D"]),
            frozenset(["B"]),
            frozenset(["B", "C"]),
            frozenset(["B", "D"]),
        ],
        rule.keys(),
    )
    self.assertIn("B", rule)
    self.assertNotIn("X", rule)
    self.assertEqual(
        {
            frozenset(["A", "C"]): "1",
            frozenset(["A", "D"]): "2",
        },
        rule._get_partial_matches("A"),
    )
    self.assertEqual(
        {"d1": {"B", "A"}, "d2": {"D", "C"}},
        rule.unique_features_per_dimension(),
    )
    self.assertEqual("3", rule("B"))
    self.assertEqual("3", rule(frozenset(["B"])))
    self.assertCountEqual(["1", "3"], rule(["A.C", "B"]))

  def test_vect_rules(self):
    rules_vect = {"A.C": ["1", "11"], "A.D": ["2", "20", "21"], "B": "3"}
    rule_vect = core.Rule("R.X", self.dims, rules_vect)
    self.assertEqual("R", rule_vect.code)
    self.assertEqual(
        {
            frozenset(["A", "C"]): ["1", "11"],
            frozenset(["A", "D"]): ["2", "20", "21"],
        },
        rule_vect._get_partial_matches("A"),
    )
    self.assertCountEqual(["1", "11", "2", "20", "21"], rule_vect("A"))
    self.assertEqual("3", rule_vect(frozenset(["B"])))
    self.assertCountEqual([["1", "11"], "3"], rule_vect(["A.C", "B"]))

  def test_convert_list_to_vector_rule(self):
    rule = core.convert_list_to_vector_rule(
        [{"A": "1", "B": "2"}, {"A": "3", "B": "4"}], "R", self.dims
    )
    self.assertIsInstance(rule, core.Rule)
    self.assertCountEqual(["1", "3"], rule("A"))
    self.assertCountEqual(["2", "4"], rule("B"))
    self.assertEqual("R", rule.name)
    self.assertIn(self.dim1, rule.dimensions.dimensions)
    self.assertEqual(
        "Rule 'R':\n'A': ['1', '3']\n'B': ['2', '4']\nUnique dimensions:\n'd1':"
        " False",
        str(rule).strip(" ").strip("\n"),
    )
    self.assertEqual({"R.d1": "A"}, rule.generate_mapping_feature("A"))

  def test_find_all_sub_dimensions(self):
    sub_dims = core.find_all_sub_dimensions(
        [{"A": "1", "B": "2"}, {"A": "3", "B": "4"}], self.dims
    )
    self.assertLen(sub_dims.dimensions, 1)
    self.assertIn(self.dim1, sub_dims.dimensions)
    self.assertNotIn(self.dim2, sub_dims.dimensions)
    sub_dims = core.find_all_sub_dimensions(
        [{"A": "1", "B": "2"}, {"C": "3", "D": "4"}], self.dims
    )
    self.assertLen(sub_dims.dimensions, 2)
    self.assertIn(self.dim1, sub_dims.dimensions)
    self.assertIn(self.dim2, sub_dims.dimensions)

  def test_find_changing_dimensions(self):
    self.assertEqual(
        {"d1": False, "d2": True},
        core.find_changing_dimensions({"A.C": "x", "A.D": "y"}, self.dims),
    )
    self.assertEqual(
        {"d1": True},
        core.find_changing_dimensions({"A": "x", "B": "x"}, self.dims),
    )

  def test_find_constraints_placeholder(self):
    self.assertEqual(
        ["d1"], core.find_constraints_placeholder("job.A", self.dims)
    )
    self.assertCountEqual(
        ["d1", "d2"],
        core.find_constraints_placeholder("name.<job.d2>.B", self.dims),
    )

  def test_init_template(self):
    t = core.Template(
        "{pa} {pb.<pa.d1>}",
        self.dims,
        [self.r1],
        pa=[{"A": "123"}, {"B": "456"}],
        pb=[{"A": "1", "B": "4"}],
        verbose=True,
    )
    self.assertIsInstance(t, core.Template)
    samples = t(2, export=False)
    self.assertLen(samples, 2)
    self.assertCountEqual(samples, ["123 1", "456 4"])

  def test_duplicate_init_template(self):
    t = core.Template(
        "{pa} {pb.<pa.d1>}",
        self.dims,
        [self.r1],
        pa=[{"A": "123"}, {"B": "456"}],
        pb=[{"A": "1", "B": "4"}],
        verbose=True,
    )
    self.assertIsInstance(t, core.Template)
    samples = t(2, export=False)
    self.assertLen(samples, 2)
    self.assertCountEqual(samples, ["123 1", "456 4"])

  def test_template_compiled_rule(self):
    t = core.Template(
        "{pa} {1:pa.A|2:pa.B}",
        self.dims,
        [self.r1],
        pa=[{"A": "123"}, {"B": "456"}],
        pb=[{"A": "1", "B": "4"}],
        verbose=True,
    )
    self.assertIsInstance(t, core.Template)
    samples = t(2, export=False)
    self.assertLen(samples, 2)
    self.assertCountEqual(samples, ["123 1", "456 2"])

  def test_template_non_attribute(self):
    t = core.Template(
        "{pa} {ent}",
        self.dims,
        [self.r1],
        pa=["123", "456"],
        ent=["abc", "def"],
        verbose=True,
    )
    self.assertIsInstance(t, core.Template)
    samples = t(4, export=False)
    self.assertLen(samples, 4)
    self.assertCountEqual(samples, ["123 abc", "123 def", "456 abc", "456 def"])

  def test_template_duplicate_non_attribute(self):
    t = core.Template(
        "{pa1} {pa2}",
        self.dims,
        [],
        pa=["123", "456"],
        same_keys_cfg={"pa": {"repeat": False, "order": True}},
        verbose=True,
    )
    self.assertIsInstance(t, core.Template)
    samples = t(4, export=False)
    self.assertLen(samples, 1)
    self.assertIn("123", samples[0])
    self.assertIn("456", samples[0])

  def test_template_random(self):
    t = core.Template(
        "{pa} {1:pa.A|2:pa.B}",
        self.dims,
        [self.r1],
        pa=[{"A": "123"}, {"B": "456"}],
        pb=[{"A": "1", "B": "4"}],
        verbose=True,
        randomized=True,
    )
    self.assertIsInstance(t, core.Template)
    samples = t(4, export=False)
    self.assertLen(samples, 2)
    self.assertCountEqual(samples, ["123 1", "456 2"])

  def test_template_fn(self):
    def choice_fn(x):
      return x if x == "123" else x + "s"

    t = core.Template(
        "{pa} {$choice(pa)}",
        self.dims,
        [self.r1],
        pa=["123", "456"],
        choice=choice_fn,
        verbose=True,
        randomized=True,
    )
    self.assertIsInstance(t, core.Template)
    samples = t(4, export=False)
    self.assertLen(samples, 2)
    self.assertCountEqual(samples, ["123 123", "456 456s"])


if __name__ == "__main__":
  absltest.main()

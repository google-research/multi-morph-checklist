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
from absl.testing import parameterized
from multi_morph_checklist import core
from multi_morph_checklist import utils


class UtilsTest(parameterized.TestCase):
  dim1 = core.Dimension("a1", ["v1", "v2"])
  dim2 = core.Dimension("a2", ["v2"])
  dims = core.Dimensions([dim1, dim2])

  def test_frozensets_contain(self):
    """Test for frozensets_contain function."""
    keys = frozenset("A")
    fsets = [
        frozenset(("A", "B")),
        frozenset(("A", "C")),
        frozenset(("B", "C")),
    ]
    self.assertEqual(fsets[:2], utils.frozensets_contain(keys, fsets))

  def test_get_frozen_set_option(self):
    """Test for get_frozen_set_option function."""
    options = [{"r1.a1": "v1", "r1.a2": "v2", "r2.a1": "v3", "r2.a2": "v4"}]
    expected = [{"r1": frozenset(("v1", "v2")), "r2": frozenset(("v3", "v4"))}]
    self.assertEqual(expected, utils.get_frozen_set_option(options))

  @parameterized.named_parameters(
      (
          "without_subset_keys",
          [
              {
                  "r1": frozenset(("v1", "v2")),
                  "r2": frozenset(("v1", "v2")),
                  "r3": frozenset(("v3", "v4")),
                  "r4": frozenset(("v4", "v5")),
                  "r5": frozenset(("v4", "v5")),
              },
              {"r1": frozenset(("v1", "v2"))},
          ],
          None,
          [[["r1", "r2"], ["r4", "r5"]], []],
      ),
      (
          "with_subset_keys",
          [
              {
                  "r1": frozenset(("v1", "v2")),
                  "r2": frozenset(("v1", "v2")),
                  "r3": frozenset(("v3", "v4")),
                  "r4": frozenset(("v4", "v5")),
                  "r5": frozenset(("v4", "v5")),
              },
              {"r1": frozenset(("v1", "v2"))},
          ],
          ["r1", "r2"],
          [[["r1", "r2"]], []],
      ),
  )
  def test_find_duplicate_values(self, options, subset_keys, expected):
    """Test for find_duplicate_values."""
    # order of nested outputs in function does not matter, but here requires
    # order to keep consistent with expected result
    self.assertEqual(
        expected,
        [sorted(e) for e in utils.find_duplicate_values(options, subset_keys)],
    )

  def test_get_dimensions_conditions(self):
    """Test for get_dimensions_conditions function."""
    self.assertCountEqual(
        [
            {"p1.a1": "v1", "p1.a2": "v3", "p2.a1": "v1", "p2.a2": "v3"},
            {"p1.a1": "v1", "p1.a2": "v3", "p2.a1": "v1", "p2.a2": "v4"},
            {"p1.a1": "v1", "p1.a2": "v3", "p2.a1": "v2", "p2.a2": "v3"},
            {"p1.a1": "v1", "p1.a2": "v3", "p2.a1": "v2", "p2.a2": "v4"},
            {"p1.a1": "v1", "p1.a2": "v4", "p2.a1": "v1", "p2.a2": "v4"},
            {"p1.a1": "v1", "p1.a2": "v4", "p2.a1": "v2", "p2.a2": "v3"},
            {"p1.a1": "v1", "p1.a2": "v4", "p2.a1": "v2", "p2.a2": "v4"},
            {"p1.a1": "v2", "p1.a2": "v3", "p2.a1": "v2", "p2.a2": "v3"},
            {"p1.a1": "v2", "p1.a2": "v3", "p2.a1": "v2", "p2.a2": "v4"},
            {"p1.a1": "v2", "p1.a2": "v4", "p2.a1": "v2", "p2.a2": "v4"},
        ],
        utils.get_dimensions_conditions(
            {
                "p1.a1": ["v1", "v2"],
                "p1.a2": ["v3", "v4"],
                "p2.a1": ["v1", "v2"],
                "p2.a2": ["v3", "v4"],
            },
            {"p1": "p", "p2": "p"},
            {"p": {"order": True, "repeat": False}},
            verbose=True,
        ),
    )

  def test_flatten_nested_list(self):
    """Test for flatten_nested_list function."""
    l = [[{"a": 1, "b": 2}, {"c": 1, "d": 2}]]
    expected = [{"a": 1, "b": 2, "c": 1, "d": 2}]
    self.assertEqual(expected, utils.flatten_nested_list_of_dict(l))

  def test_combine_cartesian_products(self):
    """Test for combine_cartesian_products function."""
    products = [[{"a": 1, "b": 2}], [{"c": 3, "d": 4}]]
    expected = [{"a": 1, "b": 2, "c": 3, "d": 4}]
    self.assertEqual(expected, utils.combine_cartesian_products(products))

  @parameterized.named_parameters(
      (
          "full_matrix",
          True,
          False,
          [
              ("1", "1"),
              ("1", "2"),
              ("1", "3"),
              ("2", "1"),
              ("2", "2"),
              ("2", "3"),
              ("3", "1"),
              ("3", "2"),
              ("3", "3"),
          ],
      ),
      (
          "full_no_diagonal",
          False,
          False,
          [
              ("1", "2"),
              ("1", "3"),
              ("2", "1"),
              ("2", "3"),
              ("3", "1"),
              ("3", "2"),
          ],
      ),
      ("lower_no_diagonal", False, True, [("1", "2"), ("1", "3"), ("2", "3")]),
      (
          "lower_matrix",
          True,
          True,
          [
              ("1", "1"),
              ("1", "2"),
              ("1", "3"),
              ("2", "2"),
              ("2", "3"),
              ("3", "3"),
          ],
      ),
  )
  def test_product_order_repeat(self, repeat, order, expected):
    """Test for product_order_repeat function."""
    values = ["1", "2", "3"]
    count = 2
    self.assertEqual(
        expected, utils.product_order_repeat(values, count, repeat, order)
    )

  def test_all_options_same_size(self) -> None:
    self.assertEqual(2, utils.all_options_same_size([{"A": "1", "B": "2"}]))
    self.assertEqual(
        -1, utils.all_options_same_size([{"A": "1", "B": "2"}, {"C": "3"}])
    )

  @parameterized.named_parameters(
      ("full_matrix", True, False, 252),
      ("full_no_diagonal", False, False, 30240),
      ("lower_no_diagonal", False, True, 100000),
      ("lower_matrix", True, True, 2002),
  )
  def test_compute_number_options(self, repeat, order, expected):
    """Test for compute_number_options function."""
    self.assertEqual(
        expected, utils.compute_number_options(10, 5, repeat, order)
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="standard",
          vectors={"first_name.GENDER": ["SG", "PL"]},
          expected=[{"first_name.GENDER": "SG"}, {"first_name.GENDER": "PL"}],
      ),
      dict(
          testcase_name="standard_same_keys",
          vectors={
              "first_name1.GENDER": ["SG", "PL"],
              "first_name2.GENDER": ["SG", "PL"],
          },
          same_keys=[(["first_name1", "first_name2"], True)],
          same_keys_cfg={"first_name": {"order": False, "repeat": True}},
          duplicate_rules_to_root={
              "first_name1": "first_name",
              "first_name2": "first_name",
          },
          short_rule_to_rules={
              "first_name1": ["first_name1.GENDER"],
              "first_name2": ["first_name2.GENDER"],
          },
          expected=[
              {"first_name1.GENDER": "SG", "first_name2.GENDER": "SG"},
              {"first_name1.GENDER": "SG", "first_name2.GENDER": "PL"},
              {"first_name1.GENDER": "PL", "first_name2.GENDER": "SG"},
              {"first_name1.GENDER": "PL", "first_name2.GENDER": "PL"},
          ],
      ),
  )
  def test_cartesian_product_with_keys(
      self,
      vectors,
      expected,
      same_keys=None,
      same_keys_cfg=None,
      duplicate_rules_to_root=None,
      short_rule_to_rules=None,
  ):
    """Test for cartesian_product_with_keys function."""
    # TODO(ehlavnova)
    self.assertCountEqual(
        expected,
        utils.cartesian_product_with_keys(
            vectors,
            same_keys,
            same_keys_cfg,
            duplicate_rules_to_root,
            short_rule_to_rules,
        ),
    )

  @parameterized.named_parameters(
      ("5_to_11", [1, 2, 3, 4, 5], 11, [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1]),
      ("5_to_2", [1, 2, 3, 4, 5], 2, [1, 2]),
      ("5_to_0", [1, 2, 3, 4, 5], 0, []),
      ("empty", [], 1, []),
  )
  def test_repeat_to_expected_length(self, l, n, expected):
    """Test for repeat_to_expected_length function."""
    self.assertEqual(expected, utils.repeat_to_expected_length(l, n))

  @parameterized.named_parameters(
      (
          "basic",
          "Hello {first_name}, I am {ART} {job}",
          {"job", "first_name", "ART"},
      ),
      (
          "basic_double",
          "Hello {first_name1} and {first_name2}, I am {ART} {job}",
          {"job", "first_name1", "first_name2", "ART"},
      ),
      (
          "basic_duplicates",
          "Hello {first_name}, I am {first_name}",
          {"first_name"},
      ),
      ("empty", "", set({})),
  )
  def test_find_all_keys(self, s, expected):
    """Test for find_all_keys function."""
    self.assertEqual(
        set(k for k in expected), set(k for k in utils.find_all_keys(s))
    )

  @parameterized.named_parameters(
      ("basic", "A.B.C", ("A", "B", "C")),
      ("duplicate", "A.B.C.B", ("A", "B", "C")),
      ("empty", "", [""]),
  )
  def test_key_to_set(self, s, expected):
    """Test for key_to_set function."""
    self.assertEqual(frozenset(expected), utils.key_to_set(s))

  @parameterized.named_parameters(
      ("basic", [{"a.b": "1", "c.d": "3"}, {"b.d", "2"}], {"a", "b", "c", "d"}),
      (
          "duplicate",
          [{"a": "1", "c": "3"}, {"b": "2", "c": "4"}],
          {"a", "b", "c"},
      ),
      ("empty", [], set({})),
  )
  def test_unique_keys(self, l, expected):
    """Test for unique_keys function."""
    self.assertEqual(expected, utils.unique_keys(l))

  def test_add_cbs(self):
    """Test for add_cbs function."""
    s = "test"
    expected = "{test}"
    self.assertEqual(expected, utils.add_cbs(s))

  @parameterized.named_parameters(
      ("standard", "job.test", "job"),
      ("standard_brackets", "<job.test>", "job"),
      ("empty", "", ""),
  )
  def test_get_front_arg(self, s, expected):
    """Test for get_front_arg function."""
    self.assertEqual(expected, utils.get_front_arg(s))

  @parameterized.named_parameters(
      ("standard", "job.test", ["test"]),
      ("standard_brackets", "job.<test>", []),
      ("standard_long", "job.test.future", ["test", "future"]),
      (
          "standard_long_brackets",
          "name.<job.test.future>.number.<job>",
          ["number"],
      ),
      ("empty", "", []),
  )
  def test_get_last_args(self, s, expected):
    """Test for get_last_args function."""
    self.assertEqual(expected, utils.get_last_args(s))

  @parameterized.named_parameters(
      ("standard", [1, 2, 3], 3, [1, 1, 1, 2, 2, 2, 3, 3, 3]),
      ("standard_one", [1, 2, 3], 1, [1, 2, 3]),
      ("empty", [], 2, []),
      ("empty_count", [1, 2, 3], 0, []),
  )
  def test_repeat_n_times(self, l, n, expected):
    """Test for repeat_n_times function."""
    self.assertEqual(expected, utils.repeat_n_times(l, n))

  @parameterized.named_parameters(
      ("simple", "$f(a,b,c,d)", ("f", ["a", "b", "c", "d"])),
      ("long", "$call(def,abc)", ("call", ["def", "abc"])),
  )
  def test_parse_fn_str(self, s, expected):
    """Test for parse_fn_str function."""
    self.assertEqual(expected, utils.parse_fn_str(s))

  def test_topological_sort_rules(self):
    """Test for topological_sort_rules function."""
    self.assertEqual(
        ["job", "name.<job.NUMBER>", "obj.<name.GENDER>"],
        utils.topological_sort_rules(
            ["name.<job.NUMBER>", "job", "obj.<name.GENDER>"]
        ),
    )


if __name__ == "__main__":
  absltest.main()

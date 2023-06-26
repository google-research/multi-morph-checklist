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
"""Util functions for multi_morph_checklist.

Functions to operate on Frozen sets and strings.
"""

import collections
import copy
import functools
import itertools
import math
import re
import string
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

T = TypeVar("T")


def frozensets_contain(
    keys: FrozenSet[str], fsets: List[FrozenSet[str]]
) -> Optional[List[FrozenSet[str]]]:
  """Returns each frozen set in fsets that are parts of all keys."""
  out = []
  for fs in fsets:
    if all(k in fs for k in keys):
      out.append(fs)
  return out


def get_frozen_set_option(
    options: List[Dict[str, str]]
) -> List[Dict[str, FrozenSet[str]]]:
  """Get attribute values in a frozen set for each rule."""
  # Example: From [{"r1.a1": "v1", "r1.a2": "v2", "r2.a1": "v3", "r2.a2": "v4"}]
  # to [{"r1": frozenset{"v1", "v2"}, "r2": frozenset{"v3", "v4"}}]
  out = []
  for op in options:
    d = collections.defaultdict(list)
    for k, v in op.items():
      # get rule name (example: 'first_name1')
      k = k.split(".")[0]
      d[k].append(v)
    out.append({k: frozenset(v) for k, v in d.items()})
  return out


def find_duplicate_values(
    options: List[Dict[str, FrozenSet[str]]],
    subset_keys: Optional[List[str]] = None,
) -> List[Optional[List[str]]]:
  """Handle duplicate examples based on subset_keys."""
  out = []
  if subset_keys:
    subset_keys = set(subset_keys)
    # pylint: disable=g-complex-comprehension
    options = [
        {k: v for k, v in option.items() if k in subset_keys}
        for option in options
    ]

  for d in options:
    new = [[k for k in d if d[k] == n] for n in set(d.values())]
    new = [e for e in new if len(e) > 1]
    out.append(new)
  return out


def combine_dimensions(
    d: Dict[str, Any], duplicates: List[str]
) -> Dict[str, str]:
  res = {}
  for dup in duplicates:
    out = []
    for k, v in d.items():
      if get_front_arg(k) == dup:
        out.append(v)
    if out:
      res[".".join(out)] = ""
  return res


def get_dimensions_conditions(
    conditions_features: Dict[str, List[str]],
    duplicate_front_rules_to_root: Optional[Dict[str, str]] = None,
    same_keys_cfg: Optional[Dict[str, Dict[str, bool]]] = None,
    verbose: bool = False,
) -> List[Dict[str, str]]:
  """Finds combination of dimensions from feature conditions.

  Args:
    conditions_features: dict of placeholder.dimension to values
    duplicate_front_rules_to_root: mapping of duplicated placeholder to their
      root. ex: {"name1": "name"}
    same_keys_cfg: duplicate config for each placeholder
    verbose: Prints to debug

  Returns:
    List of each placeholder.dimension mapping to its value
    Ex: [{'name1.NUMBER': 'SG', 'name1.GENDER': 'FEM', 'name2.NUMBER': 'SG',
      'name2.GENDER': 'FEM'}, ..]
  """
  if duplicate_front_rules_to_root is None:
    duplicate_front_rules_to_root = {}
  if verbose:
    print(f"duplicate_front_rules_to_root:\n{duplicate_front_rules_to_root}")
  if same_keys_cfg is None:
    same_keys_cfg = {}
  vectors = copy.copy(conditions_features)
  # ex: {"name": {"name1": ["name1.NUMBER", "name1.GENDER"] }}
  same_keys_dimensions_nested = {}
  for k in conditions_features.keys():
    # number1, _
    front_rule, _ = k.split(".")
    if verbose:
      print(f"key {k}")
    if front_rule in duplicate_front_rules_to_root:
      if verbose:
        print(f"-> past if: key {k}")
      root = duplicate_front_rules_to_root[front_rule]
      if root not in same_keys_dimensions_nested:
        same_keys_dimensions_nested[root] = {}
      if front_rule not in same_keys_dimensions_nested[root]:
        same_keys_dimensions_nested[root][front_rule] = []
      same_keys_dimensions_nested[root][front_rule].append(k)

  if verbose:
    print(f"same_keys_dimensions_nested:\n{same_keys_dimensions_nested}")
  products = []
  for root, d in same_keys_dimensions_nested.items():
    # ex: root=fruit, d = {'fruit1':["fruit1.NUMBER", "fruit1.GENDER"]}
    pre_prod = []
    for front_rule, vals in d.items():
      # ex: front_rule = "fruit1", vals = ["fruit1.NUMBER", "fruit1.GENDER"]
      pre_prod.append(
          cartesian_product_with_keys({k: vectors.pop(k) for k in vals})
      )
    if verbose:
      print(f"root: {root}")
      print(f"pre_prod:\n{pre_prod}")
    products.append(
        nested_dict_product_order_repeat(
            pre_prod, True, same_keys_cfg[root]["order"]
        )
    )
  if vectors:
    # call cartesian product of rest (if not empty)
    products.append(cartesian_product_with_keys(vectors))

  if verbose:
    print(f"products:\n{products}")
  return combine_cartesian_products(products)


def flatten_nested_list_of_dict(
    l: List[List[Dict[Any, Any]]]
) -> List[Dict[Any, Any]]:
  """Merge list of list of dictionaries together into a list of dictionaries.

  Used for combining cartesian products by aggregating at the dictionary level.

  Args:
    l: list of list of dictionaries to unnest

  Returns:
    List of unested dictionaries.
  """
  out = []
  for nest in l:
    out.append({})
    for e in nest:
      out[-1] = {**out[-1], **e}
  return out


def flatten_unique_list_of_list(l: List[Union[Any, List[Any]]]) -> List[Any]:
  """Flatten list of list together into a list of unique elements."""
  out = []
  for nest in l:
    out.extend(nest)
  return list(set(out))


def combine_cartesian_products(
    products: List[List[Dict[Any, Any]]]
) -> List[Dict[Any, Any]]:
  """Combines multiple cartesian products into one by merging nested dicts."""
  return flatten_nested_list_of_dict(
      list(itertools.product(*[p for p in products if p]))
  )


def convert_options_to_values(d: Dict[str, List[str]]) -> List[Dict[str, str]]:
  """Convert options dict {rule->options} to values ([{rule->value}])."""
  out = []
  for key, vals in d.items():
    out.extend({} for _ in range(max(0, len(vals) - len(out))))
    for i, val in enumerate(vals):
      out[i][key] = val
  return out


def product_order_repeat(
    values: List[T], count: int, repeat: bool = False, order: bool = False
) -> List[Tuple[Union[List[T], T], ...]]:
  """Support for different order/repeat product of values."""
  if not order and repeat:
    it = itertools.product(*[values for _ in range(count)])
  elif not order and not repeat:
    it = itertools.permutations(values, count)
  elif order and not repeat:
    it = itertools.combinations(values, count)
  else:
    it = itertools.combinations_with_replacement(values, count)
  return list(it)


def nested_dict_product_order_repeat(
    vals: List[List[Dict[str, str]]], repeat: bool = False, order: bool = False
) -> List[Dict[str, str]]:
  """Product of list of nested dictionaries with order/repeat."""
  n, m = len(vals), max([len(e) for e in vals])
  indices = product_order_repeat(list(range(m)), n, repeat=repeat, order=order)
  combine_dict = lambda x, y: {**x, **y}
  # pylint: disable=g-complex-comprehension
  return [
      functools.reduce(
          combine_dict,
          [vals[j][k] for j, k in enumerate(idxs) if j < n],
      )
      for idxs in indices
      if all([idx < len(vals[i]) for i, idx in enumerate(idxs)])
  ]


def all_options_same_size(l: List[Dict[str, str]]) -> int:
  return len(l[0]) if min(map(len, l)) == max(map(len, l)) else -1


def compute_number_options(
    n: int, duplicates: int = 1, order: bool = True, repeat: bool = False
) -> int:
  """Support for different order/repeat options to compute combinations."""
  # order = True, repeat = False => comb
  if order and not repeat:
    out = math.comb(n, duplicates)
  # order = False, repeat = False => perm
  elif not order and not repeat:
    out = math.perm(n, duplicates)
  # order = True, repeat = True => comb with replacement
  elif order and repeat:
    out = math.comb(n + duplicates - 1, duplicates)
  # order = False, repeat = True => perm with replacement
  elif not order and repeat:
    out = math.pow(n, duplicates)
  return int(out)


def conditional_same_size(
    nb_options: int,
    dims: List[int],
    duplicates: int = 1,
    order: bool = True,
    repeat: bool = False,
) -> int:
  """Compute conditional combinations based on config, dims and duplicates."""
  nb_unique = 1
  for dim in dims:
    nb_unique *= dim

  base = compute_number_options(
      nb_options, duplicates, order=order, repeat=repeat
  )
  if order:
    return base * compute_number_options(
        nb_unique, duplicates, order=order, repeat=True
    )
  else:
    for dim in dims:
      base *= compute_number_options(dim, duplicates, order=order, repeat=True)
    return base


def cartesian_product_with_keys(
    vectors: Dict[str, List[str]],
    same_keys: Optional[List[Tuple[List[str], bool]]] = None,
    same_keys_cfg: Optional[Dict[str, Dict[str, bool]]] = None,
    duplicate_front_rules_to_root: Optional[Dict[str, str]] = None,
    short_rule_to_rules: Optional[Dict[str, List[str]]] = None,
    verbose: bool = False,
) -> List[Dict[str, Union[str, List[str]]]]:
  """Cartesian product with conditionality (keys).

  Args:
    vectors: Dict mapping "ruleX.attributeY" to all possible values
    same_keys: List of names of duplicate rule names (ex: ['r1', 'r2'])
    same_keys_cfg: Dict of configuration for order/repeat options
    duplicate_front_rules_to_root: Dict mapping from duplicate name to root name
      (ex: {'r1': 'r'})
    short_rule_to_rules: Dict mapping from short rule name to complete rule
      names (ex: {'job2': ['job2.<first_name2.NUMBER>'], 'exp_0':
      ['exp_0.<first_name2.NUMBER>']})
    verbose: bool for debugging

  Returns:
    List of cartesian product of vectors according to unique keys
    combination
    Example: cartesian_product_with_keys(
       {'first_name2.NUMBER': ['SG', 'PL'],
        'first_name1.NUMBER': ['SG', 'PL']}) = [
       {'first_name2.NUMBER': 'SG', 'first_name1.NUMBER': 'SG'},
       {'first_name2.NUMBER': 'SG', 'first_name1.NUMBER': 'PL'},
       {'first_name2.NUMBER': 'PL', 'first_name1.NUMBER': 'SG'},
       {'first_name2.NUMBER': 'PL', 'first_name1.NUMBER': 'PL'} ]
  """
  vectors = copy.deepcopy(vectors)
  if not same_keys:
    same_keys = []
  if not same_keys_cfg:
    same_keys_cfg = {}
  if not duplicate_front_rules_to_root:
    duplicate_front_rules_to_root = {}
  if not short_rule_to_rules:
    short_rule_to_rules = {}
  if same_keys:
    assert duplicate_front_rules_to_root is not None

  keys, vals = zip(*vectors.items())
  if same_keys:
    products = []
    # Loop over key fronts (key fronts with unicity param):
    # e.g. (["job1", "job2"], True) , (["first_name1", "first_name2"], False)
    for same_key, unique in same_keys:
      root_key = duplicate_front_rules_to_root.get(same_key[0], same_key[0])
      if verbose:
        print(f"root_key:\n{root_key}")
      # find right order/repeat params
      # pylint: disable=g-long-ternary
      if root_key in same_keys_cfg:
        repeat, order = (
            same_keys_cfg[root_key]["repeat"] if not unique else True,
            same_keys_cfg[root_key]["order"] if not unique else False,
        )
      else:
        repeat, order = (False, True)
      vals = []
      if verbose:
        print(f"same_key:\n{same_key}")
        print(f"unique:\n{unique}")
        print(f"vectors:\n{vectors}")
      for front_key in same_key:
        sub_options = {
            k: vectors.pop(k) for k in short_rule_to_rules[front_key]
        }
        vals.append(convert_options_to_values(sub_options))
      if verbose:
        print("vals: (before nested_dict_product_order_repeat)\n{vals}")
      products.append(nested_dict_product_order_repeat(vals, repeat, order))
      if verbose:
        print(f"products:\n{products}")
    if vectors:
      # call cartesian product of rest (if not empty)
      products.append(cartesian_product_with_keys(vectors))
      if verbose:
        print(f"products after rest is added:\n{products}")
    return combine_cartesian_products(products)
  else:
    vals = [[(key, v) for v in val] for key, val in zip(keys, vals)]
    res = list(itertools.product(*vals))
    return [{o[0]: o[1] for o in options} for options in res]


def repeat_to_expected_length(l: List[T], n: int) -> List[T]:
  if not l:
    return []
  mult = (n + len(l) - 1) // len(l)
  return (l * mult)[:n]


def find_all_keys(s: str) -> List[str]:
  """Finds all tag keys in string.

  Take string with keys like "Hello {first_name}, I am {ART} {job}" and extracts
  the keys in {}.

  Args:
    s: string with keys

  Returns:
    List of all keys (with options)
  """
  ret = []
  f = string.Formatter()
  for x in f.parse(s):
    r = x[1] if not x[2] else "%s:%s" % (x[1], x[2])
    ret.append(r)
  # filter duplicate while preserving the order
  seen = set()
  seen_add = seen.add
  # NOMUTANTS -- mutant alterate expected return instead of failing test
  return [x for x in ret if x and not (x in seen or seen_add(x))]


def key_to_set(s: str) -> FrozenSet[str]:
  """Converts string of features to a frozen set.

  Takes input like "ART.DEFINITE.FEM" and converts the list of features to
  a frozen set of string. Helps to 'hash' a list of features.

  Args:
    s: string representing a list of features separated by points, e.g. "A.B.C"

  Returns:
    A frozen set containing the extracted features from the string.
  """
  return frozenset(s.split("."))


def unique_keys(l: List[Dict[str, str]]) -> Optional[Set[str]]:
  s = set()
  for e in l:
    if isinstance(e, dict):
      for ks in e:
        s.update(ks.split("."))
  return s


def sort_keys(s: str) -> str:
  return ".".join(sorted(s.split(".")))


def add_cbs(s: str) -> str:
  """Add curly brackets before and after a string."""
  return "{" + s + "}"


def get_front_arg(s: str) -> str:
  return s.split(".")[0].strip("<>")


def get_last_args(s: str, non_dependence: bool = True) -> Optional[List[str]]:
  """Get last args except from conditional ones (with <>)."""
  if non_dependence:
    # removes any <...>
    s = re.sub("<[a-zA-Z0-9._|:,]*[^<]>", "", s)
    # replaces any multiple ... by one .
    s = re.sub(r"\.{2,}", ".", s).strip(".")
    args = s.split(".")
    return args[1:] if len(args) > 1 else []
  else:
    # replace any non <x.y.z>
    s = re.sub(".[a-zA-Z0-9_]+^[>.]", "", s)
    args = re.findall(".<([a-zA-Z0-9._]*)>", s)
    return args if args else []


def repeat_n_times(l: List[Any], n: int) -> List[Any]:
  """Repeats n times each element of list l."""
  return list(itertools.chain.from_iterable(itertools.repeat(x, n) for x in l))


def parse_fn_str(s: str) -> Tuple[str, List[str]]:
  """Parse fn string '$fn_name(arg1,...,argn)' to (fn_name, [arg1,...,argn])."""
  for e in "$()":
    assert e in s
  fn_name = s[1:].split("(")[0]
  args = s[:-1].split("(")[1].split(",")
  return fn_name, args


class GraphTopologicalSort:
  """Topological Sort of graph with node values."""

  def __init__(self, nodes: List[str]):
    self.graph = collections.defaultdict(list)
    self.nb_vertices = len(nodes)
    self.int_to_key = nodes
    self.key_to_int = {k: i for i, k in enumerate(self.int_to_key)}

  def add_edge(self, u: str, v: str) -> None:
    """Method to add an edge to graph."""
    self.graph[self.key_to_int[u]].append(self.key_to_int[v])

  def _topological_sort_util(
      self, v: int, visited: List[bool], results: List[int]
  ) -> None:
    """Recursive method for topological sort."""
    visited[v] = True
    for i in self.graph[v]:
      if not visited[i]:
        self._topological_sort_util(i, visited, results)
    results.insert(0, v)

  def topological_sort(self) -> List[str]:
    visited = [False for _ in range(self.nb_vertices)]
    results = []
    for i in range(self.nb_vertices):
      if not visited[i]:
        self._topological_sort_util(i, visited, results)
    return list(map(lambda x: self.int_to_key[x], results))


def topological_sort_rules(
    rules: List[str], verbose: bool = False
) -> List[Optional[str]]:
  """Sort according to Topological Sort a set of dependent rules."""
  # WRONG -> what if "first_name.SG" and "first_name.<job.NUMBER>"
  short_to_long = {get_front_arg(r): r for r in rules}
  nodes = short_to_long.keys()
  graph_ts = GraphTopologicalSort(list(nodes))
  if verbose:
    print("Performing Topological Sort...")
  for r in rules:
    front_rule, args = get_front_arg(r), get_last_args(r, False)
    for arg in args:
      front_arg_dep = get_front_arg(arg)
      if front_arg_dep in nodes:
        graph_ts.add_edge(front_arg_dep, front_rule)
        if verbose:
          print(f"Edge from '{front_arg_dep}' to '{front_rule}'.")

  return list(map(short_to_long.get, graph_ts.topological_sort()))

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
"""All core classes of library: Dimension(s), Rule, Template, Language.

Dimension: holds features associated with a Unimorph-style dimension
  ex: GENDER with features MASC, FEM, NEUT
Dimensions: collection of Dimensions
  ex: [GENDER, NUMBER]
Rule: holds hashmap to get conditional values
  ex: {MASC: [John, Paul], FEM: [Christine, Jennifer]}
Template: string with placeholders converted to function sampling strings
  ex: "{first_name} is {adj}"
Language: metadata for a language (i.e. Dimensions, Rule)
  ex: {"EN", [GENDER, NUMBER], Rule(article)}
"""

import collections
import copy
import itertools
import math
import os
import random
import re
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple, Union
from multi_morph_checklist import utils


# pylint: disable=g-multiple-import
# pylint: disable=g-import-not-at-top
try:
  from unimorph_inflect.src.inflection import inflect
# pylint: disable=bare-except
except ImportError:
  inflect = None


# DIMENSIONS
class Dimension:
  """Dimension (ex: Gender, Number) to hold features (ex: MASC, PL)."""

  def __init__(
      self,
      name: str,
      features: List[str],
      sampleable: bool = True,
      call: Optional[Callable[[str], str]] = None,
  ):
    self.name = name
    self.features = features
    self.call = None if call is None else (lambda s: None if not s else call(s))
    self.sampleable = sampleable

  def __call__(self, x: str) -> Optional[Union[str, NotImplementedError]]:
    return (
        self.call(x)
        if self.call is not None
        else NotImplementedError(f"No caller implemented in dimension {self}")
    )

  def __str__(self) -> str:
    return (
        f"'{self.name}': {str(self.features)} (sampleable: {self.sampleable})"
    )


class Dimensions:
  """List of Dimension object."""

  def __init__(self, dimensions: List[Dimension]) -> None:
    self.dimensions = dimensions
    self._name_to_dim = self._create_mapping_to_dimension()
    self.name_to_features = self._create_mapping_to_features()
    self.feature_to_name = self._reverse_dict()
    self.keys = list(self.name_to_features.keys())

  def _create_mapping_to_features(self) -> Dict[str, List[str]]:
    """Get mapping from name to features."""
    return {d.name: d.features for d in self.dimensions}

  def _create_mapping_to_dimension(self) -> Dict[str, Dimension]:
    """Get mapping from name to dimensions."""
    return {d.name: d for d in self.dimensions}

  def _reverse_dict(self):
    """Get mapping from unique feature to name."""
    # pylint: disable=g-complex-comprehension
    return {
        val: category
        for category, vals in self.name_to_features.items()
        for val in vals
    }

  def get_dimensions_from_features(
      self, s: Union[Iterable[str], Set[str]]
  ) -> List[Dimension]:
    """Get all dimensions from a sequence of feature names."""
    if not isinstance(s, set):
      s = set(s)
    set_dimensions = set()
    for f in s:
      set_dimensions.add(self.feature_to_name[f])
    return [self.__call__(dim) for dim in set_dimensions]  # pytype: disable=bad-return-type  # always-use-return-annotations

  def __contains__(self, key: str) -> bool:
    return key in self.keys

  def __call__(self, name: str) -> Union[Dimension, KeyError]:
    if name not in self._name_to_dim:
      raise KeyError(
          f"Cannot find dimension called '{name}' in Dimensions object:\n{self}"
      )
    return self._name_to_dim[name]

  def __str__(self) -> str:
    return "\n".join(map(str, self.dimensions)) + "\n"


class LRUCache:
  """Least Recently Used cache implementation."""

  def __init__(self, capacity: int):
    self.cache = collections.OrderedDict()
    self.capacity = capacity

  def __getitem__(self, key: Any) -> Any:
    if key not in self:
      return -1
    else:
      self.cache.move_to_end(key)
      return self.cache[key]

  def __setitem__(self, key: Any, value: Any) -> None:
    self.cache[key] = value
    self.cache.move_to_end(key)
    if len(self.cache) > self.capacity:
      self.cache.popitem(last=False)

  def __contains__(self, key: Any) -> bool:
    return key in self.cache


# RULE
class Rule:
  """Maps features to value(s) to perform (conditional) sampling.

  Example: english_article = Rule(
    "ART", Dimensions([binary_article, binary_starts_with, binary_number]), {
        "SG.INDEFINITE.VOW": "an",
        "SG.INDEFINITE.CONS": "a",
        "PL.INDEFINITE": "",
        "DEFINITE": "the"
    })
  """

  def __init__(
      self,
      name: str,
      dimensions: Dimensions,
      rules: Dict[str, str],
      uniques: Optional[Dict[str, bool]] = None,
      cache_size: int = 100,
  ) -> None:
    self.name = name
    self.code = name if "." not in name else name.split(".")[0]
    self.dimensions = dimensions
    self.rules = self._create_mapping(rules)
    self.uniques = copy.deepcopy(uniques) if uniques is not None else {}
    self.cache_size = cache_size
    self.call_cache = LRUCache(self.cache_size)

  def _to_set(self, s: Union[str, FrozenSet[str]]) -> FrozenSet[str]:
    return s if isinstance(s, frozenset) else utils.key_to_set(s)

  def keys(self) -> List[FrozenSet[str]]:
    return list(self.rules.keys())

  def unique_features(self) -> Set[str]:
    out = set()
    for ks in self.keys():
      for k in ks:
        out.add(k)
    return out

  def unique_features_per_dimension(self) -> Dict[str, Set[str]]:
    res = {}
    for f in self.unique_features():
      dim = self.dimensions.feature_to_name[f]
      if dim not in res:
        res[dim] = set()
      res[dim].add(f)
    return res

  def _create_mapping(self, rules: Dict[str, str]) -> Dict[FrozenSet[str], str]:
    """Convert input dict of rules to final frozen set hashmap for lookup."""
    new_rules = {}
    dimensions = find_all_sub_dimensions([rules], self.dimensions)
    # create initial mapping
    for rule, val in rules.items():
      # convert rule to set and add
      key = utils.key_to_set(rule)
      new_rules[key] = val
      # find missing categories
      missing = list(
          set(dimensions.keys) - set(dimensions.feature_to_name[k] for k in key)
      )
      # create all missing set combinations
      vals = [dimensions.name_to_features[k] for k in missing]
      init_keys = list(key)
      for to_add in itertools.product(*vals):
        new_rules[frozenset(init_keys + list(to_add))] = val
    self.dimensions = dimensions
    return new_rules

  def __str__(self) -> str:
    s = f"Rule '{self.name}':\n"
    for k, v in self.rules.items():
      s += "'" + ".".join(sorted(k)) + "': " + str(v) + "\n"
    if self.uniques:
      s += "Unique dimensions:\n"
      for k, v in self.uniques.items():
        s += f"'{k}': {v}\n"
    return s

  def __contains__(self, key: str) -> bool:
    return bool(utils.frozensets_contain(self._to_set(key), self.keys()))

  def _get_partial_matches(
      self, key: Union[str, FrozenSet[str]]
  ) -> Dict[FrozenSet[str], str]:
    # filter to the partial match frozen sets
    sub_keys = utils.frozensets_contain(self._to_set(key), self.keys())
    # extract the sub rules
    sub_rules = {keys: self.rules[keys] for keys in sub_keys}
    return sub_rules

  def generate_mapping_feature(
      self, keys: Union[str, FrozenSet[str]]
  ) -> Dict[str, str]:
    res = {}
    if isinstance(keys, str):
      keys = self._to_set(keys)
    for key in keys:
      dim = self.dimensions.feature_to_name[key]
      res[f"{self.code}.{dim}"] = key
    return res

  def __call__(
      self,
      inputs: Union[str, FrozenSet[str], List[str], List[FrozenSet[str]]],
      overwrite_rules: Optional[
          Union[Dict[str, List[str]], Dict[FrozenSet[str], str]]
      ] = None,
  ) -> Union[str, List[str], List[List[str]]]:
    rules = overwrite_rules or self.rules
    if isinstance(inputs, list) or inputs not in self.call_cache:
      if isinstance(inputs, str):
        inputs_set = utils.key_to_set(inputs)
        self.call_cache[inputs] = self(
            inputs_set, overwrite_rules=overwrite_rules
        )
      elif isinstance(inputs, frozenset):
        if inputs in rules:
          self.call_cache[inputs] = rules[inputs]
        else:
          # partial match of keys "ACC" for ["ACC.FEM", "ACC.SG", "ACC.PL"]
          sub_rules = self._get_partial_matches(inputs)
          sub_vals = [
              vs if isinstance(vs, list) else [vs] for vs in sub_rules.values()
          ]
          out = utils.flatten_unique_list_of_list(sub_vals) if sub_rules else []
          if isinstance(out, list) and len(out) == 1:
            out = out[0]
          self.call_cache[inputs] = out
      elif isinstance(inputs, list):
        out = []
        for e in inputs:
          out.append(self(e, overwrite_rules=overwrite_rules))
        return (
            out  # pytype: disable=bad-return-type  # always-use-return-annotations
        )
    return self.call_cache[inputs]


def find_unique_dimensions(
    dicts: List[Dict[str, str]], dimensions: Dimensions, verbose: bool = False
) -> Dict[str, bool]:
  """Given a list of attribute entities, find the dimensions that are unique.

  Conditions to be unique: be constant in same entity and be defined
  in all entities.

  Args:
    dicts: list of attribute entity with morphology key and value
    dimensions: Dimensions to be considered
    verbose: to print debugging info if True

  Returns:
    Dict of dimension names to true if differ, false otherwise.
    Examples:
      [
          {"SG.MASC": "x", "PL.MASC": "x"},
          {"SG.FEM": "x", "PL.FEM": "x"}
      ] => {"NUMBER": False, "GENDER": True}
  """
  all_dims = set()
  out = {}
  always_present = set(
      dimensions.feature_to_name[k] for k in list(dicts[0].keys())[0].split(".")
  )
  # loop over each entity
  i = -1
  for d in dicts:
    if verbose:
      i += 1
      print(" ==> ", i, d)
      print("always_present:\n", always_present)
    # loop over each of entity's keys
    vals_in_key = collections.defaultdict(set)
    for keys in d.keys():
      dims_in_key = set()
      # loop over each feature of key
      for k in keys.split("."):
        dim = dimensions.feature_to_name[k]
        all_dims.add(dim)
        vals_in_key[dim].add(k)
        dims_in_key.add(dim)
      if verbose:
        print("dims_in_key:\n", dims_in_key)
      for dim in list(always_present):
        if dim not in dims_in_key or len(vals_in_key[dim]) > 1:
          always_present.remove(dim)

  for k in all_dims:
    # dimension must be always referenced AND must be sampleable
    try:
      dim = dimensions(k)
      sampleable = dim.sampleable if not isinstance(dim, KeyError) else False
    except KeyError:
      sampleable = False
    out[k] = k in always_present and sampleable

  return out


def find_changing_dimensions(
    d: Dict[str, Union[str, List[str]]], dimensions: Dimensions
) -> Dict[str, bool]:
  """Given a single attribute entity, find the dimension that differs in key.

  Args:
    d: attribute entity with morphology key and value/list of values
    dimensions: Dimensions to be considered

  Returns:
    Dict of dimension names to true if differ, false otherwise.
    Examples:
      {"SG.FEM": "x", "SG.MASC": "y"} => {"NUMBER": False, "GENDER": True}
      {"SG": "x", "PL": "x"} => {"NUMBER": True}
  """
  vals = collections.defaultdict(set)
  for keys in d.keys():
    for k in keys.split("."):
      vals[dimensions.feature_to_name[k]].add(k)

  out = {}
  for k, v in vals.items():
    out[k] = len(v) > 1

  return out


def find_constraints_placeholder(s: str, dimensions: Dimensions) -> List[str]:
  """Finds the dimensions (if any) with a direct constraint on a placeholder.

  Args:
    s: placeholder string
    dimensions: Dimensions to be considered

  Returns:
    List of dimension names that have constraints on the placeholder.
    Examples:
      job.SG -> ["NUMBER"], name.<job.GENDER>.PL -> ["GENDER", "NUMBER"]
  """
  # extract the direct features
  features = utils.get_last_args(s, non_dependence=True)
  dims = set(dimensions.feature_to_name[f] for f in features)
  # extract the dependence with dimension names
  extra_dims = set(
      utils.flatten_unique_list_of_list(
          [
              utils.get_last_args(dep, non_dependence=True)
              for dep in utils.get_last_args(s, non_dependence=False)
          ]
      )
  )
  for dim in extra_dims:
    dims.add(dim)
  return list(dims)


def convert_list_to_vector_rule(
    l: List[Dict[str, str]], name: str, init_dimensions: Dimensions
) -> Rule:
  """Utility to list of dict (attributes) to a Rule."""
  dimensions = find_all_sub_dimensions(l, init_dimensions)
  rules = collections.defaultdict(list)
  uniques = find_unique_dimensions(l, dimensions)
  # process rules
  for d in l:
    for key, val in d.items():
      # find missing categories (if any)
      missing = list(
          set(dimensions.keys)
          - set(dimensions.feature_to_name[k] for k in utils.key_to_set(key))
      )
      if not missing:
        rules[utils.sort_keys(key)].append(val)
      else:
        # create all missing set combinations
        vals = [dimensions.name_to_features[k] for k in missing]
        for to_add in itertools.product(*vals):
          target_key = utils.sort_keys(".".join([key] + list(to_add)))
          rules[target_key].append(val)
  return Rule(name, dimensions, rules, uniques=uniques)


def find_all_sub_dimensions(
    l: List[Dict[str, str]], dimensions: Dimensions
) -> Dimensions:
  """From list of dimension mappings get all unique dimensions."""
  features = utils.unique_keys(l)
  return Dimensions(dimensions.get_dimensions_from_features(features))


def determine_same_keys(
    options: List[Dict[str, str]],
    duplicate_options: Dict[str, List[str]],
    dimensions: Dimensions,
    unicity: Optional[Dict[str, Dict[str, bool]]] = None,
    short_rule_to_rules: Optional[Dict[str, List[str]]] = None,
    duplicate_front_rules_to_root: Optional[Dict[str, str]] = None,
    non_allowed_rules: Optional[Set[str]] = None,
) -> List[Optional[List[Tuple[Union[List[str], str], bool]]]]:
  """Finds commom keys between all rules.attributes keys.

  Args:
    options: list of dict mapping each rule.attribute key to a feature value
    duplicate_options: dict of rule name to duplicate rule names (r1, r2)
    dimensions: dimensions available from the language
    unicity: dict of root rule to dict of dimension unicity (bool)
    short_rule_to_rules: dict of rule front name to list of rules
    duplicate_front_rules_to_root: dict of front name to root rule
    non_allowed_rules: rule names for which same keys are not allowed

  Returns:
    For each element, the keys that have the same features for all attributes.
    Example:
    determine_same_keys([
      {'first_name1.NUMBER': 'SG', 'first_name2.NUMBER': 'SG'},
      {'first_name1.NUMBER': 'SG', 'first_name2.NUMBER': 'PL'}
    ], duplicate_options = {
      'first_name': ['first_name2', 'first_name1']
    }) = [[['first_name2', 'first_name1']], []]
  """
  if duplicate_front_rules_to_root is None:
    duplicate_front_rules_to_root = {}
  if non_allowed_rules is None:
    non_allowed_rules = set()
  out = [[] for _ in range(len(options))]

  for i, v in enumerate(options):
    for k, dups in duplicate_options.items():
      if all(v not in non_allowed_rules for v in dups):
        # v={"p1.SG":"MASC", ...}, k="p", dups=[p1,p2]
        # ONLY WORKS for 2 dups, for more need to generalize
        # with each (pi,pj) instead of just (p1,p2)
        d = utils.combine_dimensions(v, dups)
        diffs = find_changing_dimensions(d, dimensions)
        unique = False
        # if any dim with diff is unique then comb is unique
        for dim, diff in diffs.items():
          if diff and unicity[k][dim]:
            unique = True
        out[i].append((dups, unique))

  # 2. Duplicate front rule (r.<a> and r.<b>) without duplicate keys
  if short_rule_to_rules:
    for k, v in short_rule_to_rules.items():
      # front rule has more than one placeholder
      # and is not a duplicate (e.g. r1, r2)
      print(v)
      if (
          not any([e in v for e in non_allowed_rules])
          and len(v) > 1
          and duplicate_front_rules_to_root.get(k, k) not in duplicate_options
      ):
        for i, _ in enumerate(out):
          out[i] += [([k], False)]
  return out


# TEMPLATE
ROOT_FOLDER = "/cns/il-d/home/ehlavnova/m2c_export/"


class Template:
  """Template structure holding templating processing and logic."""

  def __init__(
      self,
      template: str,
      dimensions: Dimensions,
      rules: Optional[List[Rule]] = None,
      max_count: Optional[int] = None,
      verbose: bool = False,
      same_keys_cfg: Optional[Dict[str, Dict[str, bool]]] = None,
      post_processing: Optional[Callable[[str], str]] = None,
      modifiers: Optional[Dict[str, Callable[[str], str]]] = None,
      export_file_path: Optional[str] = None,
      **kwargs,
  ):
    self.template = template
    self.dimensions = dimensions
    self.rules = rules
    self.verbose = verbose
    self.same_keys_cfg = same_keys_cfg if same_keys_cfg is not None else {}
    self.post_processing = post_processing
    self.sampled_rules, self.sampled_att_rules, self.non_sampled_rules = (
        [],
        [],
        [],
    )
    self.count_values_placeholders, self.same_size_values_placeholders = {}, {}
    self.functions = collections.defaultdict(dict)
    self.short_rule_to_rules = collections.defaultdict(list)
    self.duplicate_rules, self.duplicate_rules_to_front = (
        collections.defaultdict(list),
        collections.defaultdict(list),
    )
    self.duplicate_rules_to_root, self.duplicate_front_rules_to_root = {}, {}
    self.dimensions_to_compute = collections.defaultdict(list)
    self.mapping_to_rules = {r.name: r for r in self.rules}
    self.modifiers = modifiers if modifiers else {}
    self.rule_to_modifiers = collections.defaultdict(list)
    self.rule_no_modifiers_to_with_modifiers = collections.defaultdict(list)
    self.keys = utils.find_all_keys(self.template)
    self.fill_in_values = self.get_rules(max_count, **kwargs)
    (
        self.conditions_features,
        self.dimensions_options,
        self.values_options,
        self.no_options,
        self.roots,
        self.roots_unicity,
    ) = self.pre_populate_rules()
    self.export_file_path = export_file_path
    self.theoretical_nb_options = self._compute_number_options()

  def export(self, outputs: List[str], sanitize_output: bool = True):
    """Exports outputs to self.export_file_path as txt."""
    if self.export_file_path is None:
      raise ValueError("export_file_path is not set.")

    file_path = os.path.join(ROOT_FOLDER, self.export_file_path)
    if not os.path.isfile(file_path):
      os.mkdir(os.path.dirname(file_path))

    if sanitize_output:
      n = len(outputs)
      seen = set()
      seen_add = seen.add
      outputs = [x for x in outputs if not (x in seen or seen_add(x))]
      if len(outputs) != n:
        print(
            f"Warning: generated {n} items, but removed"
            f" {n-len(outputs)} duplicates."
        )

    with open(file_path, "wt") as f:
      for line in outputs:
        f.write(line + "\n")

  def get_rules(
      self, max_count: Optional[int] = None, **kwargs
  ) -> Dict[str, List[str]]:
    """Initial setup of rules for template."""
    compiles, rules, values = {}, {}, {}
    exp_id = 0
    for arg in kwargs:
      if re.search(r"\d+$", arg):
        raise (
            KeyError(
                "Error: input keys cannot end in integers, we use that to index"
                ' multiple copies of the same key (offending key: "%s")' % arg
            )
        )
    # preprocessing function modifiers like .TO_CAPS
    clean_template = self.template
    old_nb_keys = len(self.keys)
    if self.modifiers:
      patterns = "|".join(map(lambda x: "." + x, self.modifiers.keys()))
      modifiers_patterns = f"({patterns})"
      for k in self.keys:
        new_key = k
        mods = []
        for mod in re.findall(modifiers_patterns, k):
          new_key = new_key.replace(mod, "")
          clean_template = clean_template.replace(mod, "")
          mods.append(mod[1:])
        if mods:
          self.rule_to_modifiers[k] = mods
          self.rule_no_modifiers_to_with_modifiers[new_key].append(k)
    # re-update the keys
    self.keys = utils.find_all_keys(clean_template)
    removed_keys = old_nb_keys - len(self.keys)

    if self.verbose:
      print(f"Cleaned template: {clean_template}")

    for k in self.keys:
      if k.startswith("$"):
        # function form
        fn_name, fn_args = utils.parse_fn_str(k)
        assert fn_name in kwargs
        self.functions[k] = {"args": fn_args, "fn": kwargs[fn_name]}
        continue

      if all(e in k for e in ":|") and all(e not in k for e in "<>"):
        # expression to compile
        rule, new_key = self.compile_exp_to_rule(k, exp_id)
        exp_id += 1
        compiles[k] = new_key
        rules[new_key] = rule
        self.non_sampled_rules.append(new_key)
        continue

      # preprocessing on k to only extract front key (start of key)
      start_k = re.sub(r"\..*", "", k)
      start_k = re.sub(r"\[.*\]", "", start_k)
      start_k = re.sub(r".*?:", "", start_k)
      new_k = re.sub(r"\d+$", "", start_k)
      if re.compile(r"mask\d*").match(start_k):
        continue

      if new_k in kwargs:
        if new_k != start_k:
          # number case (e.g. first_name1, first_name2)
          # Variables values: k = name1.<x>, start_k = name1, new_k = name
          self.duplicate_rules[new_k].append(k)
          if start_k not in self.duplicate_rules_to_front[new_k]:
            self.duplicate_rules_to_front[new_k].append(start_k)
          self.duplicate_front_rules_to_root[start_k] = new_k
          self.duplicate_rules_to_root[k] = new_k

        if k == start_k and all(isinstance(e, str) for e in kwargs[new_k]):
          # key is passed as kwargs (standard checklist)
          values[start_k] = kwargs[new_k]
          self.sampled_rules.append(start_k)
          self.short_rule_to_rules[start_k].append(start_k)
        else:
          # rule based kwargs (conditional checklist)
          rules[k] = convert_list_to_vector_rule(
              kwargs[new_k], new_k, self.dimensions
          )
          self.sampled_att_rules.append(k)
        self.count_values_placeholders[new_k] = len(kwargs[new_k])
        self.same_size_values_placeholders[new_k] = utils.all_options_same_size(
            kwargs[new_k]
        )
      elif new_k in self.mapping_to_rules:
        # already existing rule (i.e. provided by the language)
        rules[k] = self.mapping_to_rules[new_k]
        self.non_sampled_rules.append(k)
      else:
        raise KeyError(f'Error: key "{new_k}" not in items or lexicons')

      if max_count:
        values[start_k] = values[start_k][:max_count]

    # re-compiled template
    self.compiled_template = self.template
    if compiles:
      # replace conditional rules after compilation (e.g. {v1:c1|...|vn:cn})
      for key, new_key in compiles.items():
        self.compiled_template = self.compiled_template.replace(
            utils.add_cbs(key), utils.add_cbs(new_key)
        )

    # go through arg of functions to sample if template is used only in fn
    # only working for rules without attributes for now
    all_sampled_front_rules = set(
        utils.get_front_arg(r) for r in self.sampled_rules
    )
    for fn, v in self.functions.items():
      for arg in v["args"]:
        if arg not in all_sampled_front_rules and arg in kwargs:
          if self.verbose:
            print(
                f"For fn '{fn}', arg {arg} not sampled independently. Adding to"
                " sampled rules."
            )
          values[arg] = kwargs[arg]
          self.sampled_rules.append(arg)
          self.short_rule_to_rules[arg].append(arg)
          self.count_values_placeholders[arg] = len(kwargs[arg])
          self.same_size_values_placeholders[arg] = utils.all_options_same_size(
              kwargs[arg]
          )

    self.nb_keys = (
        len(set(re.findall("{[a-zA-Z0-9_.<>:|,]*}", self.compiled_template)))
        + len(self.functions)
        - removed_keys
    )

    # save compiled rules and order them
    self.compiled_rules = rules
    for r in rules:
      self.short_rule_to_rules[utils.get_front_arg(r)].append(r)

    self.non_sampled_rules = utils.topological_sort_rules(
        self.non_sampled_rules, False
    )
    self.compiled_root_to_rule = {
        k.name: k for k in self.compiled_rules.values()
    }

    # find dimensions that require to be computed (like STARTSWITH)
    for rule_name in self.non_sampled_rules:
      for dep in re.findall("<[.a-zA-Z0-9._]+>", rule_name):  # NOTYPO
        target_name = utils.get_front_arg(dep)
        for arg in utils.get_last_args(dep.strip("<>")):
          if arg in self.dimensions:
            dim = self.dimensions(arg)
            if not isinstance(dim, KeyError) and dim.call is not None:
              self.dimensions_to_compute[target_name].append(dim)

    if self.verbose:
      print(" => RULES")
      print("self.sampled_rules:")
      print(self.sampled_rules)
      print("self.sampled_att_rules:")
      print(self.sampled_att_rules)
      print("self.non_sampled_rules:")
      print(self.non_sampled_rules)

      print("self.compiled_rules:")
      print(self.compiled_rules)
      for v in self.compiled_root_to_rule.values():
        print(v)

      print("self.duplicate_rules:")
      print(self.duplicate_rules)

      print("self.dimensions_to_compute:")
      print(self.dimensions_to_compute)

      print("self.short_rule_to_rules:")
      print(self.short_rule_to_rules)

      print("self.rule_to_modifiers:")
      print(self.rule_to_modifiers)

      print("self.rule_no_modifiers_to_with_modifiers:")
      print(self.rule_no_modifiers_to_with_modifiers)

      print("self.count_values_placeholders")
      print(self.count_values_placeholders)
      print("self.same_size_values_placeholders")
      print(self.same_size_values_placeholders)

      print("self.compiled_template:")
      print(self.compiled_template)

    return values

  def __call__(
      self,
      n_samples: int,
      randomized: bool = True,
      returns: bool = True,
      export: bool = True,
      checks: bool = True,
  ) -> Optional[List[str]]:
    """Main caller for generating n_samples samples from template."""
    if n_samples == -1:
      n_samples = math.inf
    vals = self.populate_rules(n_samples, randomized=randomized)
    if self.post_processing is not None:
      vals = [self.post_processing(v) for v in vals]

    if checks:
      unique = len(set(vals))
      out = len(vals)
      # unique always same out, if n_samples above theory, then out = theory,
      # otherwise n_samples = out
      ok = (unique == out) and (
          (
              n_samples >= self.theoretical_nb_options
              and self.theoretical_nb_options == out
          )
          or (n_samples < self.theoretical_nb_options and n_samples == out)
      )
      print(
          f"Check number of samples: {'SUCCESS' if ok else 'FAIL'}\nTheory:"
          f" {self.theoretical_nb_options} | Unique: {unique} | Output:"
          f" {out} {'<' if n_samples > out else ('>' if n_samples < out else '=')} {n_samples} requested"
          " samples"
      )

    vals = (
        sorted(set(vals), key=vals.index)
        if not isinstance(vals, ValueError)
        else []
    )
    if export:
      print(f"Exporting outputs to {self.export_file_path}...")
      self.export(vals)

    if not returns:
      for v in vals:
        print(v)
    else:
      return vals if not isinstance(vals, ValueError) else None

  def _compute_number_options(self) -> int:
    """Tentatively computes number of theoretical options from template."""
    rules, prod, duplicates = set(), 1, {}
    # find rules to consider and their duplicates
    for rule_name in self.sampled_rules + self.sampled_att_rules:
      name = utils.get_front_arg(rule_name)
      if bool(re.search(r"\d+$", name)):
        name = re.sub(r"\d+$", "", name)
        duplicates[name] = len(
            set(utils.get_front_arg(r) for r in self.duplicate_rules[name])
        )
      else:
        duplicates[name] = 1
      rules.add(name)

    for name in rules:
      # finds corresponding dup settings (if present)
      if name in self.same_keys_cfg:
        d = self.same_keys_cfg[name]
        order, repeat = d["order"], d["repeat"]
      else:
        order, repeat = True, False
      n, dup = self.count_values_placeholders[name], duplicates[name]
      if self.verbose:
        print(
            f"Rule {name}:\n"
            " self.same_size_values_placeholders[name]:"
            f" {self.same_size_values_placeholders[name]}"
        )
      if (
          dup == 1
          or name not in self.roots
          or self.same_size_values_placeholders[name] == 1
      ):
        mul = utils.compute_number_options(n, dup, order, repeat)
      else:
        # is a root with strictly more than 1 dup
        raw_placeholders = self.duplicate_rules[name]
        constraints = {
            placeholder: find_constraints_placeholder(
                placeholder, self.dimensions
            )
            for placeholder in raw_placeholders
        }
        if len(set(tuple(l) for l in constraints.values())) != 1:
          print(
              "Warning: the constraints of these duplicate placeholders are"
              " different, the theoretical size computation is not supported."
          )
          print("Constraints:")
          for p, cs in constraints.items():
            print(f"Placeholder {p}: {cs}")
          return -1
        constraint = list(constraints.values())[0]

        # for each placeholder get unique features
        # ex: {'GENDER': {'FEM', 'MASC'}, 'NUMBER': {'PL', 'SG'}}
        unique_features = self.compiled_root_to_rule[
            name
        ].unique_features_per_dimension()
        # get unique dimensions with format from {root}.{dimension} to dimension
        # ex: ["obj.GENDER", "job.NUMBER"] -> ["GENDER"]
        roots_unicity = [
            e.split(".")[1]
            for e in self.roots_unicity
            if e.split(".")[0] == name
        ]
        # check all options of the rule have the same size
        if (
            name not in self.same_size_values_placeholders
            or self.same_size_values_placeholders[name] == -1
        ):
          print(
              "Warning: a root placeholder with dependence has options with"
              " different sizes. The estimation of theoretical number of values"
              " is not yet supported."
          )
          return -1
        dims_to_consider = set(unique_features.keys()) - set(constraint)
        dims_shape = [
            len(unique_features[d])
            for d in dims_to_consider
            if d not in roots_unicity
        ]
        mul = utils.conditional_same_size(n, dims_shape, dup, order, repeat)

      # if mul >= 1:
      if self.verbose:
        print(f"For rule {name}, multiplier is {mul} (n={n}, dup={dup})")
      prod *= mul
      # else:
      #   print(f"Warning: For rule {name}, multiplier is {mul}, which is < 1.")
    return prod

  def compile_exp_to_rule(self, exp: str, idx: int) -> Tuple[Rule, str]:
    """Convert conditional expression to Rule."""
    options = exp.split("|")
    rules, name = {}, None
    features = set()
    for opt in options:
      val, key = opt.split(":")
      first_key = key.split(".")[0]
      # validation on name (first key in each option condition)
      if name is None:
        name = first_key
      elif name != first_key:
        raise KeyError(
            "Got conflicting keys during compilations of expression"
            f" '{exp}'\nObtained first key {name} but then a different one:"
            f" {first_key}"
        )
      # get all keys except first
      clean_key = ".".join(key.split(".")[1:])
      # get unique dimensions
      for k in clean_key.split("."):
        features.add(k)
      # add keys in mapping
      rules[clean_key] = val
    # get all dimensions
    dimensions = Dimensions(
        self.dimensions.get_dimensions_from_features(features)
    )

    # format rule's name to be unique
    name_rule = f"exp_{idx}"

    # compute compiled expression
    attribute = ".".join([name] + list(dimensions.keys))
    compiled_exp = name_rule + f".<{attribute}>"
    return Rule(name_rule, dimensions, rules), compiled_exp

  def pre_populate_rules(self):
    """Static part of populating rules only ran at initialization."""
    # 1.a. find conditions between rules
    conditions = []

    ### condition from dependence with other rules
    for rule_name in self.compiled_rules:
      to_compile = re.findall("<[.a-zA-Z0-9._]+>", rule_name)  # NOTYPO
      # ex: pattern = <first_name.GENDER.NUMBER>
      for pattern in to_compile:
        name, args = utils.get_front_arg(pattern), utils.get_last_args(
            pattern.strip("<>")
        )
        for arg in args:
          # add if not already there and if can be added
          # (only sampled rule or non_sampled rule with sampleable dim)
          try:
            dim = self.dimensions(arg)
            sampleable = (
                dim.sampleable if not isinstance(dim, KeyError) else False
            )
          except KeyError:
            sampleable = False
          if (name, arg) not in conditions and (
              rule_name in self.sampled_att_rules or sampleable
          ):
            conditions.append((name, arg))
    if self.verbose:
      print(
          "Before second wave of conditions without dependence...\nconditions:"
      )
      print(conditions)
    ### conditions from one rule without dependence in other rules
    rules_fetched = set(c[0] for c in conditions)
    if self.verbose:
      print("Second pass on conditions:")
    for rule_name in self.compiled_rules:
      front_name = utils.get_front_arg(rule_name)
      # if name not in rules_fetched:
      dims = set(self.get_dimensions_rule(rule_name))
      if self.verbose:
        print(f"\nIn rule '{rule_name}':")
        print(f"Dims: {dims}")
      if front_name in rules_fetched:
        for name, arg in [
            (c, arg) for c, arg in conditions if c == name and arg in dims
        ]:
          dims.remove(arg)
      # find dimensions not to consider from dependence
      # ex: first_name.<job.GENDER>
      patterns = re.findall("<[.a-zA-Z0-9.:|,_]+>", rule_name)
      for pattern in patterns:
        # for normal <x.f> patterns
        if ":" not in pattern and "|" not in pattern:
          args = utils.get_last_args(pattern.strip("<>"))
          for arg in args:
            if arg in dims:
              dims.remove(arg)
        else:
          # for <v0:c0|...> patterns
          vals = pattern.strip("<>").split("|")
          # v0:c01,c02
          for val in vals:
            split = val.split(":")
            if len(split) == 2:
              # since could be default value
              value = split[0]
              dim = self.dimensions.feature_to_name[value]
              if dim in dims:
                dims.remove(dim)
      # find dimensions not to consider because already determined
      # ex: first_name.PL
      for feature in utils.get_last_args(rule_name):
        dim = self.dimensions.feature_to_name[feature]
        if dim in dims:
          dims.remove(dim)
      # add remaining conds
      for dim in dims:
        # if not sampleable AND not in sampled_att_rules, then do not add
        try:
          dim_sampleable = self.dimensions(dim)
          sampleable = (
              dim_sampleable.sampleable
              if not isinstance(dim_sampleable, KeyError)
              else False
          )
        except KeyError:
          sampleable = False
        if rule_name in self.sampled_att_rules or sampleable:
          if self.verbose:
            print(f"Adding in rule '{rule_name}': {front_name}, {dim}")
          conditions.append((front_name, dim))

    if self.verbose:
      print(
          "After second wave of conditions without dependence...\nconditions:"
      )
      print(conditions)
    # get dict mapping conditions to options
    conditions_features = {}
    for name, arg in conditions:
      if name in self.short_rule_to_rules:
        for r in self.short_rule_to_rules[name]:
          if r in self.compiled_rules:
            rule = self.compiled_rules[r]
            valid_features = rule.unique_features()
            if arg in rule.dimensions.name_to_features:
              features = [
                  e
                  for e in rule.dimensions.name_to_features[arg]
                  if e in valid_features
              ]
              if self.verbose:
                print(
                    f"Adding to conditions_features '{name}.{arg}': {features}"
                )
                print(f"Added from rule '{r}'")
              conditions_features[f"{name}.{arg}"] = features

    ### self.sampled_att_rules not empty and conditions_features empty
    if not conditions_features:
      for r in self.sampled_att_rules:
        front_rule, args = utils.get_front_arg(r), utils.get_last_args(r)
        for arg in args:
          key = f"{front_rule}.{self.dimensions.feature_to_name[arg]}"
          conditions_features[key] = [arg]

    if self.verbose:
      print("\n => 1.a.")
      print("Dimensions to sample from (conditions_features):")
      print(conditions_features)

    if conditions_features:
      # 1.b. get cartesian product of dimensions

      # compute all combinations of dimensions
      # make sure if order=False for duplicated placeholders, cross product
      # are removed to avoid duplicates values due to e.g. [SG,PL] and [PL,SG]

      front_rules = set(
          utils.get_front_arg(k) for k in conditions_features.keys()
      )
      if self.verbose:
        print(f"front_rules: {front_rules}")
      roots = set(
          self.duplicate_front_rules_to_root[k]
          for k in front_rules
          if k in self.duplicate_front_rules_to_root
      )
      if self.verbose:
        print(f"roots: {roots}")
      roots_unicity = utils.flatten_unique_list_of_list(
          [
              [f"{k}.{u}" for u, b in rule.uniques.items() if b]
              for k, rule in self.compiled_root_to_rule.items()
              if k in self.duplicate_rules and k in roots
          ]
      )
      dimensions_options = utils.get_dimensions_conditions(
          conditions_features,
          {
              k: v
              for k, v in self.duplicate_front_rules_to_root.items()
              if k in front_rules
          },
          {
              k: {"repeat": True, "order": v["order"]}
              for k, v in self.same_keys_cfg.items()
              if k in roots
          },
          verbose=self.verbose,
      )

      if self.verbose:
        print("\n => 1.b.")
        print("Perform cartesian product over dimensions vectors of attributes")
        print("Unique attribute combinations and length (dimensions_options):")
        print(dimensions_options)
        print(len(dimensions_options))
        print("\n => 1.c.")

      # 1.c. get all values for each rule for each unique combination
      # of attribute dimension
      ### get values for attributes rules (same length as dimensions_options)
      # ex: {'obj.GENDER': "FEM", 'subj.NUMBER': "SG"}
      values_options, no_options = None, set()
      for rule_name in self.sampled_att_rules:
        values, to_add_attributes = self.get_values_rule(
            rule_name, dimensions_options
        )
        for k, v in to_add_attributes.items():
          for i, e in enumerate(v):
            dimensions_options[i][k] = e

        if self.verbose:
          print(f"Values of rule_name '{rule_name}':")
          print(values)
          print("")

        if values_options is None:
          values_options = [{rule_name: v} if v else {} for v in values]
          for i, val in enumerate(values):
            if not val:
              no_options.add(i)
        else:
          for i, val in enumerate(values):
            if i not in no_options:
              if not val:
                no_options.add(i)
              else:
                values_options[i][rule_name] = val

      if self.verbose:
        print(
            "Combined values for each unique combination of attribute"
            " (values_options):"
        )
        print(values_options)
        print("dimensions_options:")
        print(dimensions_options)
        print("no_options:")
        print(no_options)
        print("List attributes without options:")
        for i in no_options:
          print(f"[{i}] {dimensions_options[i]}")
    else:
      dimensions_options = []
      values_options = []
      no_options = []
      roots = []
      roots_unicity = []

    return (
        conditions_features,
        dimensions_options,
        values_options,
        no_options,
        roots,
        roots_unicity,
    )

  def populate_rules(
      self, n_samples: int, randomized: bool
  ) -> Union[List[str], ValueError]:
    """Fills in template with values for n_samples samples."""
    samples_per_option = n_samples

    non_attribute_rules_values = {
        k: self.fill_in_values[k] for k in self.sampled_rules
    }
    if non_attribute_rules_values:
      same_keys, keys_found = [], set()
      for k in non_attribute_rules_values:
        if k in self.duplicate_rules_to_root:
          root_key = self.duplicate_rules_to_root[k]
          if root_key not in keys_found:
            # unique
            same_keys.append((self.duplicate_rules_to_front[root_key], False))
            keys_found.add(root_key)
      # cartesian product on non-attribute rules (supports numbered placeholder)
      non_attribute_cartesian_product = utils.cartesian_product_with_keys(
          non_attribute_rules_values,
          same_keys,
          self.same_keys_cfg,
          self.duplicate_front_rules_to_root,
          self.short_rule_to_rules,
      )
      if self.verbose:
        print(
            "Products for non-attribute rules:"
            " (non_attribute_cartesian_product)"
        )
        print(non_attribute_cartesian_product)
        print("non_attribute_rules_values:")
        print(non_attribute_rules_values)

    if self.conditions_features:
      # 1.d. make cartesian product of values
      # (from dimensions and non-attribute rules)
      if self.verbose:
        print("\n => 1.d.")
        print("Perform cartesian product over values (unconditional)")

      sampled_formatting, dimensions_final_options = [], []
      if self.verbose:
        print("To determine_same_keys:")
        print(f"duplicate_rules_to_front:\n{self.duplicate_rules_to_front}")
        print(f"short_rule_to_rules:\n{self.short_rule_to_rules}")
        print(
            f"duplicate_front_rules_to_root:\n{self.duplicate_front_rules_to_root}"
        )
      unicity = {
          k: copy.deepcopy(v.uniques)
          for k, v in self.compiled_root_to_rule.items()
          if k in self.duplicate_rules
      }
      same_keys = determine_same_keys(
          self.dimensions_options,
          self.duplicate_rules_to_front,
          self.dimensions,
          unicity,
          self.short_rule_to_rules,
          self.duplicate_front_rules_to_root,
          set(self.sampled_rules),
      )
      if self.verbose:
        print("same_keys:")
        print(same_keys)
        print()

      for i, (val, same_key) in enumerate(zip(self.values_options, same_keys)):
        if i not in self.no_options:
          values = utils.cartesian_product_with_keys(
              val,
              same_key,
              self.same_keys_cfg,
              self.duplicate_front_rules_to_root,
              self.short_rule_to_rules,
          )
          if non_attribute_rules_values:
            values = utils.combine_cartesian_products(
                [values, non_attribute_cartesian_product]
            )

          length_chunk = min(samples_per_option, len(values))
          if randomized:
            random.shuffle(values)
          sampled_formatting.extend(values[:length_chunk])
          dimensions_final_options.extend(
              copy.deepcopy(self.dimensions_options[i])
              for _ in range(length_chunk)
          )
    elif non_attribute_rules_values:
      length_chunk = min(n_samples, len(non_attribute_cartesian_product))
      sampled_formatting = non_attribute_cartesian_product[:length_chunk]
      dimensions_final_options = [{} for _ in range(length_chunk)]
    else:
      raise ValueError

    ### data validation of dimensions_final_options and sampled_formatting
    ##### each element of sampled_formatting should have all rules in
    ##### sampled_rules and sampled_att_rules
    if self.verbose:
      print(
          "Before data validation:"
          f" len(sampled_formatting)={len(sampled_formatting)}"
      )
    rules_to_check = self.sampled_att_rules + self.sampled_rules
    to_remove = []
    for i, option in enumerate(sampled_formatting):
      if not all(rule_name in option for rule_name in rules_to_check):
        to_remove.append(i)

    if to_remove and self.verbose:
      print(f"Removing {len(to_remove)} options...")
    sampled_formatting = [
        e for i, e in enumerate(sampled_formatting) if i not in to_remove
    ]
    dimensions_final_options = [
        e for i, e in enumerate(dimensions_final_options) if i not in to_remove
    ]

    if self.verbose:
      print(
          "After data validation:"
          f" len(sampled_formatting)={len(sampled_formatting)}"
      )

    ### truncate in case too many values
    length_samples = min(n_samples, len(sampled_formatting))
    if randomized:
      indices = list(range(len(sampled_formatting)))
      random.shuffle(indices)
      sampled_formatting = [sampled_formatting[i] for i in indices]
      dimensions_final_options = [dimensions_final_options[i] for i in indices]

    sampled_formatting = sampled_formatting[:length_samples]
    dimensions_final_options = dimensions_final_options[:length_samples]

    # 2. run non-sampled rules (100% deterministic)
    ### get some potential additional data if some dimensions are functions
    computed_front_rules = set()
    for rule_front in self.dimensions_to_compute.keys():
      for rule_name in self.short_rule_to_rules.get(rule_front, [rule_front]):
        for dim in self.dimensions_to_compute[rule_front]:
          for i, option in enumerate(dimensions_final_options):
            # update dimension values
            if rule_name in sampled_formatting[i]:
              option[f"{rule_front}.{dim.name}"] = dim(
                  sampled_formatting[i][rule_name]
              )
              computed_front_rules.add(rule_name)

    if self.verbose:
      print("Final attribute for each sample (dimensions_final_options):")
      print(dimensions_final_options)
      print("Rules computed:")
      print(computed_front_rules)

      print("\n => 2.")
      print("Run non-sampled rules")

    for rule_name in self.non_sampled_rules:
      values, to_add_attributes = self.get_values_rule(
          rule_name, dimensions_final_options
      )
      for k, v in to_add_attributes.items():
        for i, e in enumerate(v):
          dimensions_final_options[i][k] = e
      if self.verbose:
        print(f"Non-sample rule '{rule_name}' values:")
        print(values)
      for i, v in enumerate(values):
        sampled_formatting[i][rule_name] = (
            v if not isinstance(v, list) else (v[0] if v else [])
        )

      # compute additional dimension in case not already computed before
      for rule_front in [
          e
          for e in self.dimensions_to_compute.keys()
          if rule_name
          and rule_name.startswith(e)
          and e not in computed_front_rules
      ]:
        for dim in self.dimensions_to_compute[rule_front]:
          for i, option in enumerate(dimensions_final_options):
            # update dimension values
            if rule_name in sampled_formatting[i]:
              option[f"{rule_front}.{dim.name}"] = dim(
                  sampled_formatting[i][rule_name]
              )
              computed_front_rules.add(rule_name)

    # add function values
    for fn_placeholder, fn_dict in self.functions.items():
      arg_names = fn_dict["args"]
      fn = fn_dict["fn"]
      for i, _ in enumerate(sampled_formatting):
        # sampled_formatting[i] needs to contain all placeholders values
        # except from the functional ones, since they should be added here
        if len(sampled_formatting[i]) >= self.nb_keys - len(self.functions):
          # if arg in sampled_formatting use value, otherwise look for
          # value of dimension in dimensions_final_options
          sampled_formatting[i][fn_placeholder] = fn(
              *[
                  sampled_formatting[i][arg_name]
                  if arg_name in sampled_formatting[i]
                  else dimensions_final_options[i][arg_name]
                  for arg_name in arg_names
              ]
          )

    if self.verbose:
      print("Final values for each sample with all rules (sampled_formatting):")
      print(sampled_formatting)

    if self.verbose:
      print("\n" + 45 * "_" + "\n")

    # 3. apply formatting to get strings
    out = []
    for d in sampled_formatting:
      if len(d) >= self.nb_keys:
        out.append(self.format_sample(d))
      else:
        print(
            f"Error: more keys ({self.nb_keys}) than"
            f" sampled_formatting ({len(d)})."
        )

    return out

  def get_dimensions_rule(self, rule_name: str) -> List[str]:
    rule = self.compiled_rules[rule_name]
    return [d.name for d in rule.dimensions.dimensions]

  def format_sample(self, mapping: Dict[str, str]) -> str:
    """Format compiled template from mapping of rule to value."""
    out = self.compiled_template
    for key, val in mapping.items():
      out = out.replace(utils.add_cbs(key), val)
      for real_key in self.rule_no_modifiers_to_with_modifiers[key]:
        new_val = val
        for mod in self.rule_to_modifiers[real_key]:
          new_val = self.modifiers[mod](new_val)
        out = out.replace(utils.add_cbs(real_key), new_val)

    return out

  def get_values_rule(
      self,
      rule_name: str,
      dimensions_options: List[Dict[str, Union[str, List[str]]]],
  ) -> Tuple[List[List[str]], collections.defaultdict[str, Any]]:
    """Get all option values for a given rule."""
    rule = self.compiled_rules[rule_name]
    rule_front = utils.get_front_arg(rule_name)
    to_compile = re.findall("<[.a-zA-Z0-9._]+>", rule_name)  # NOTYPO
    to_format = {}
    to_add_attributes = collections.defaultdict(list)
    # ex: pattern = <first_name.GENDER.NUMBER>
    for pattern in to_compile:
      name, args = utils.get_front_arg(pattern), utils.get_last_args(
          pattern.strip("<>")
      )
      to_format[pattern] = []
      for op in dimensions_options:
        to_format[pattern].append("")
        for arg in args:
          to_format[pattern][-1] += op[f"{name}.{arg}"] + "."
          to_add_attributes[f"{rule_front}.{arg}"].append(op[f"{name}.{arg}"])
        # remove extra point at end
        to_format[pattern][-1] = to_format[pattern][-1][:-1]
    if self.verbose:
      print(f"to_format ({rule_name}):\n{to_format}")

    # ex: 'first_name' given {'first_name.NUMBER': 'SG'}
    # only add features if their placeholder matches the rule (ex: first_name)
    # AND if that dimension is not already defined in the rule_name
    # AND no attributes in rule_name are from same dim as atts in previous atts
    implied_atts = []
    for options in dimensions_options:
      implied_atts.append([])
      for k, v in options.items():
        if k.split(".")[0] == rule_front and k.split(".")[1] not in rule_name:
          # dimension infered by feature v is not represented in rule_name
          if self.dimensions.feature_to_name[v] not in [
              self.dimensions.feature_to_name[e]
              for e in utils.get_last_args(rule_name)
              if e in self.dimensions.feature_to_name
          ] + [
              self.dimensions.feature_to_name[e.split(":")[0]]
              for e in re.findall("<([_a-zA-Z0-9.:|,_]+)>", rule_name)
              if all(c in e for c in ":|")
              and e.split(":")[0] in self.dimensions.feature_to_name
          ]:
            implied_atts[-1].append(v)

    if self.verbose:
      print(f"implied_atts ({rule_name}):\n{implied_atts}")

    # construct inputs
    back_rule_name = ".".join(rule_name.split(".")[1:])
    conds_args = [
        e
        for e in re.findall("<[_a-zA-Z0-9.:|,_]+>", back_rule_name)
        if all(c in e for c in ":|")
    ]

    if (implied_atts and implied_atts[0]) or conds_args:
      # if <{v1:c1|...|vn:cn}>, implied_atts values
      # should replace the conditional
      if self.verbose:
        print(f"back_rule_name: {back_rule_name}")
      if conds_args:
        # v0:c01,c02
        for cond_args in conds_args:
          conditions = []
          for conds in cond_args.strip("<>").split("|"):
            # find all conditions for this value:cond pair
            split = conds.split(":")
            if len(split) == 2:
              val, cs = split
              conditions.append(({}, val))
              for cond in cs.split(","):
                first_key = cond.split(".")[0]
                args = utils.get_last_args(cond)
                for arg in args:
                  conditions[-1][0][
                      f"{first_key}.{self.dimensions.feature_to_name[arg]}"
                  ] = arg
            else:
              # for default value, must be the last provided
              # e.g. v2 here: <v0:c0|v1:c1|v2>
              val = conds
              conditions.append(({}, conds))
          if self.verbose:
            print(f"Found conditions for cond_arg '{cond_args}':\n{conditions}")
          # use conditions to find appropriate values in implied_atts
          for i, _ in enumerate(dimensions_options):
            for cond, val in conditions:
              if all(
                  dimensions_options[i][k] == v
                  if k in dimensions_options[i]
                  else False
                  for k, v in cond.items()
              ):
                arg = self.dimensions.feature_to_name[val]
                if i == len(to_add_attributes[f"{rule_front}.{arg}"]):
                  to_add_attributes[f"{rule_front}.{arg}"].append(val)
                implied_atts[i].append(val)
                break
        # remove cond_args from the template rule
        for cond in conds_args:
          back_rule_name = back_rule_name.replace(cond, "")
        back_rule_name = re.sub(r"\.{2,}", ".", back_rule_name).strip(".")
      if self.verbose:
        print(f"back_rule_name: {back_rule_name}")
      inputs = [
          ".".join(atts + [back_rule_name]).strip(".") for atts in implied_atts
      ]
    else:
      base_input = ".".join(rule_name.split(".")[1:])
      inputs = [base_input for _ in range(len(dimensions_options))]

    for k, values in to_format.items():
      for i, val in enumerate(values):
        inputs[i] = re.sub(k, val, inputs[i])

    if self.verbose:
      print("Inputs to rule:")
      print(inputs)

    out = rule(inputs)
    out = [e if isinstance(e, list) else [e] for e in out]

    if self.verbose:
      print(f"to_add_attributes:\n{to_add_attributes}")
    return out, to_add_attributes


# LANGUAGE
TO_CAPITALIZE = lambda x: x.capitalize()


class Language:
  """Language structure to hold language specific metadata."""

  def __init__(
      self,
      name: str,
      code: str,
      dimensions: Dimensions,
      rules: List[Rule],
      modifiers: Optional[Dict[str, Callable[[str], str]]] = None,
  ):
    self.name = name
    self.code = code
    self.dimensions = dimensions
    self.rules = rules
    self.modifiers = (
        modifiers if modifiers else {"TO_CAPITALIZE": TO_CAPITALIZE}
    )

  def post_processing(self, s: str) -> str:
    """Standard processing function to prettify string output from template."""
    # remove trailing spaces
    s = s.strip(" ")
    # remove multiple spaces
    s = re.sub(r"\s+", " ", s)
    # remove space after "'"
    s = re.sub(r"'\s", "'", s)
    # starts with capital letter
    if len(s) > 1:
      s = s[0].upper() + s[1:]
    # force capital letter after %

    # next word after a "." with a capital letter
    # pylint: disable=anomalous-backslash-in-string
    s = re.sub("(^|[.?!])\s*([^\W\d_])", lambda p: p.group(0).upper(), s)
    return s

  def template(
      self, s: str, template_name: Optional[str] = None, **kwargs
  ) -> Template:
    return Template(
        s,
        dimensions=self.dimensions,
        rules=self.rules,
        post_processing=self.post_processing,
        modifiers=self.modifiers,
        export_file_path=os.path.join(template_name, self.code.lower() + ".txt")
        if template_name
        else None,
        **kwargs,
    )

  def generate_with_unimorph_inflect(
      self,
      features: List[List[str]],
      roots: Union[List[str], List[Dict[str, str]]],
      unimorph_format: str,
  ) -> List[Dict[str, str]]:
    """Use unimorph_inflect to inflect a list of roots according to features."""
    # TODO(ehlavnova): add support for direct dimension naming
    # i.e. '{DIMENSION}' instead of only '{}'
    assert(inflect is not None)
    if isinstance(roots[0], dict):
      assert all(len(d) == 1 for d in roots)
      keys = [list(r)[0] for r in roots]
      roots = [list(r.values())[0] for r in roots]
    else:
      keys = None
    all_features = list(itertools.product(*features))
    size = len(all_features)
    # generate the inflections in one pass
    inflected = inflect(
        utils.repeat_n_times(roots, size),
        [unimorph_format.format(*fts) for fts in all_features] * len(roots),
        language=self.code,
    )
    # format the output as [{"K1":"V1",..,"Kn":"Vn"}]
    out = []
    for i in range(len(roots)):
      out.append({})
      for val, fts in zip(inflected[i * size : (i + 1) * size], all_features):
        out[-1][".".join([keys[i]] + list(fts) if keys else fts)] = val
    return out

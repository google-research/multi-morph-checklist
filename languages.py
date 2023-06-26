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
"""Pre-defined languages.

A set of 12 languages with rudimentary dimensions and rules:
- English EN (en)
- French FR (fr)
- German DE (de)
- Italian IT (it)
- Slovak SK (sk)
- Mandarin ZH (zh)
- Swedish SV (sv)
- Swahili SW (sw)
- Russian RU (ru)
- Arabic AR (ar)
- Spanish ES (es)
- Finnish FI (fi)
"""

from multi_morph_checklist.core import Dimension
from multi_morph_checklist.core import Dimensions
from multi_morph_checklist.core import Language
from multi_morph_checklist.core import Rule

# ENGLISH AND FRENCH
binary_article = Dimension("DEFINITENESS", ["DEF", "INDF"])
binary_gender = Dimension("GENDER", ["FEM", "MASC"])
binary_number = Dimension("NUMBER", ["SG", "PL"])
binary_directive = Dimension(
    "DIRECTIVE", ["DIRECT", "INDIRECT"], sampleable=False
)
binary_article_de_prep = Dimension("ARTICLEDE", ["PASDE", "DE"])
binary_starts_with = Dimension(
    "STARTSWITH",
    ["VOW", "CONS"],
    call=lambda s: "VOW" if s[0].lower() in "aeiouyéè" else "CONS",
    sampleable=True,
)
binary_order = Dimension("ORDER", ["GT", "LT"], sampleable=False)
french_tenses = Dimension("TENSE", ["CONDL", "INF"])

all_fr_dimensions = Dimensions([
    binary_article,
    binary_gender,
    binary_starts_with,
    binary_number,
    binary_directive,
    binary_order,
    binary_article_de_prep,
    french_tenses,
])

all_en_dimensions = Dimensions([
    binary_article,
    binary_gender,
    binary_starts_with,
    binary_number,
    binary_order,
])

french_article = Rule(
    "ART",
    Dimensions(
        [binary_article, binary_gender, binary_starts_with, binary_number]
    ),
    {
        "SG.INDF.FEM": "une",
        "SG.INDF.MASC": "un",
        "SG.DEF.VOW": "l'",
        "SG.DEF.CONS.FEM": "la",
        "SG.DEF.CONS.MASC": "le",
        "PL.INDF": "des",
        "PL.DEF": "les",
    },
)

french_prep_article = Rule(
    "PREP_ART",
    Dimensions(
        [binary_article, binary_gender, binary_starts_with, binary_number]
    ),
    {
        "SG.DEF.VOW": "de l'",
        "SG.DEF.CONS.FEM": "de la",
        "SG.DEF.CONS.MASC": "du",
    },
)

french_prep_opt_article = Rule(
    "PREP_OPT_ART",
    Dimensions([
        binary_article,
        binary_gender,
        binary_starts_with,
        binary_article_de_prep,
    ]),
    {
        "DEF.PASDE.VOW": "l'",
        "DEF.PASDE.CONS.FEM": "la",
        "DEF.PASDE.CONS.MASC": "le",
        "DEF.DE.VOW": "de l'",
        "DEF.DE.CONS.FEM": "de la",
        "DEF.DE.CONS.MASC": "du",
    },
)

french_complement_pronouns = Rule(
    "PRO_COMPL",
    all_fr_dimensions,
    {
        "DIRECT.VOW": "l'",
        "DIRECT.CONS": "le",
    },
)

french_negation = Rule(
    "NEGATION",
    Dimensions([binary_starts_with]),
    {
        "VOW": "n'",
        "CONS": "ne",
    },
)

english_article = Rule(
    "ART",
    Dimensions([binary_article, binary_starts_with, binary_number]),
    {"SG.INDF.VOW": "an", "SG.INDF.CONS": "a", "PL.INDF": "", "DEF": "the"},
)

english_pronouns = Rule(
    "PRO",
    Dimensions([binary_gender, binary_starts_with, binary_number]),
    {
        "SG.MASC": "he",
        "SG.FEM": "she",
        "PL": "they",
    },
)

french_pronouns = Rule(
    "PRO",
    Dimensions([binary_gender, binary_starts_with, binary_number]),
    {
        "SG.MASC": "il",
        "SG.FEM": "elle",
        "PL.MASC": "ils",
        "PL.FEM": "elles",
    },
)

EN_rules = [english_article, english_pronouns]
EN = Language("ENGLISH", "en", all_en_dimensions, EN_rules)

FR_rules = [
    french_article,
    french_prep_article,
    french_negation,
    french_complement_pronouns,
    french_prep_opt_article,
    french_pronouns,
]
FR = Language("FRENCH", "fr", all_fr_dimensions, FR_rules)

# IT


def starts_with_fn_it(s: str) -> str:
  """Italian logic for VOW, CONS, CONS determination."""
  if s[0].lower() in "aeiouy":
    return "VOW"
  elif (
      s.startswith("gn")
      or s.startswith("ps")
      or s[0] == "z"
      or (len(s) > 1 and s[0] == "s" and s[1].lower() not in "aeiouy")
  ):
    return "CONS2"
  else:
    return "CONS"


ternary_starts_with = Dimension(
    "STARTSWITH",
    ["VOW", "CONS", "CONS2"],
    call=starts_with_fn_it,
    sampleable=False,
)

all_it_dimensions = Dimensions([
    binary_article,
    binary_gender,
    ternary_starts_with,
    binary_number,
    binary_order,
])

italian_article = Rule(
    "ART",
    all_it_dimensions,
    {
        # DEF MASC
        "SG.DEF.MASC.CONS": "il",
        "SG.DEF.MASC.CONS2": "lo",
        "SG.DEF.MASC.VOW": "l'",
        "PL.DEF.MASC.CONS": "i",
        "PL.DEF.MASC.CONS2": "gli",
        "PL.DEF.MASC.VOW": "gli",
        # DEF FEM
        "SG.DEF.FEM.CONS": "la",
        "SG.DEF.FEM.CONS2": "la",
        "SG.DEF.FEM.VOW": "l'",
        "PL.DEF.FEM": "le",
        # INDF MASC
        "SG.INDF.MASC.CONS": "un",
        "SG.INDF.MASC.CONS2": "uno",
        "SG.INDF.MASC.VOW": "un",
        # INDF FEM
        "SG.INDF.FEM.CONS": "una",
        "SG.INDF.FEM.CONS2": "una",
        "SG.INDF.FEM.VOW": "un'",
        # INDF PL
        "PL.INDF": "",
    },
)

italian_article_with_in = Rule(
    "IN_ART",
    all_it_dimensions,
    {
        # DEF MASC
        "SG.DEF.MASC.CONS": "nel",
        "SG.DEF.MASC.CONS2": "nello",
        "SG.DEF.MASC.VOW": "nell'",
        "PL.DEF.MASC.CONS": "nei",
        "PL.DEF.MASC.CONS2": "negli",
        "PL.DEF.MASC.VOW": "negli",
        # DEF FEM
        "SG.DEF.FEM.CONS": "nella",
        "SG.DEF.FEM.CONS2": "nella",
        "SG.DEF.FEM.VOW": "nell'",
        "PL.DEF.FEM": "nelle",
        # INDF MASC
        "SG.INDF.MASC.CONS": "in un",
        "SG.INDF.MASC.CONS2": "in uno",
        "SG.INDF.MASC.VOW": "in un",
        # INDF FEM
        "SG.INDF.FEM.CONS": "in una",
        "SG.INDF.FEM.CONS2": "in una",
        "SG.INDF.FEM.VOW": "in un'",
        # INDF PL
        "PL.INDF": "in",
    },
)

IT_rules = [italian_article, italian_article_with_in]
IT = Language("ITALIAN", "it", all_it_dimensions, IT_rules)

# SV

sv_gender_pronouns = Dimension("GENDERPRON", ["MASC", "FEM"])
sv_gender = Dimension("GENDER", ["COMMON", "NEUT"])
sv_with_article = Dimension(
    "WITHARTICLE", ["WITH_ART", "NO_ART"], sampleable=False
)
binary_ends_with = Dimension(
    "ENDSWITH",
    ["VOW", "CONS"],
    call=lambda s: "VOW" if s[-1].lower() in "aeiouy" else "CONS",
    sampleable=False,
)

SV_dimensions = Dimensions([
    binary_number,
    sv_gender,
    sv_gender_pronouns,
    binary_ends_with,
    sv_with_article,
    binary_order,
])

swedish_article_def = Rule(
    "ART_DEF",
    SV_dimensions,
    {
        "SG.COMMON.CONS": "en",
        "SG.COMMON.VOW": "n",
        "SG.NEUT": "et",
        "PL.COMMON": "na",
        "PL.NEUT": "na",
    },
)

swedish_article_indf = Rule(
    "ART_INDF",
    SV_dimensions,
    {"SG.COMMON": "en", "SG.NEUT": "ett", "PL.COMMON": "", "PL.NEUT": ""},
)

SV_rules = [swedish_article_def, swedish_article_indf]

SV = Language("SWEDISH", "sv", SV_dimensions, SV_rules)

# SK

gender = Dimension("GENDER", ["FEM", "MASC", "NEUT"])
sk_animacy = Dimension("ANIMACY", ["ANIM", "INAN"])
sk_cases = Dimension(
    "CASE", ["NOM", "GEN", "DAT", "ACC", "LOK", "INS"], sampleable=False
)
# Declension of numerals in Slovak:
# https://outils.apprenti-polyglotte.net/sk/explication_numeral.php?lang=en
# GTPL, for numbers higher or equal to 5
ternary_number = Dimension("NUMBER", ["SG", "PL", "GTPL"])

SK_dimensions = Dimensions(
    [ternary_number, gender, sk_cases, sk_animacy, binary_order]
)
SK_rules = []

SK = Language("SLOVAK", "sk", SK_dimensions, SK_rules)

# DE

gender = Dimension("GENDER", ["FEM", "MASC", "NEUT"])
binary_number = Dimension("NUMBER", ["SG", "PL"])
de_cases = Dimension("CASE", ["NOM", "GEN", "DAT", "ACC"])
DE_dimensions = Dimensions(
    [binary_number, gender, de_cases, binary_article, binary_order]
)

german_article = Rule(
    "ART",
    DE_dimensions,
    {
        # ACC
        "SG.INDF.FEM.ACC": "eine",
        "SG.INDF.MASC.ACC": "einen",
        "SG.INDF.NEUT.ACC": "ein",
        "PL.INDF.ACC": "",
        "SG.DEF.FEM.ACC": "die",
        "SG.DEF.MASC.ACC": "den",
        "SG.DEF.NEUT.ACC": "das",
        "PL.DEF.ACC": "die",
        # DAT
        "SG.DEF.FEM.DAT": "der",
        "SG.DEF.MASC.DAT": "dem",
        "SG.DEF.NEUT.DAT": "dem",
        "PL.DEF.DAT": "den",
        # NOM
        "SG.DEF.FEM.NOM": "die",
        "SG.DEF.MASC.NOM": "der",
        "SG.DEF.NEUT.NOM": "das",
        "PL.DEF.NOM": "die",
    },
)

DE_rules = [german_article]

DE = Language("GERMAN", "de", DE_dimensions, DE_rules)

# ZH

classifier_types = Dimension(
    "CLASSIFIER_TYPE",
    ["GENERIC", "PAIR", "ANIMAL", "LONG_NARROW", "FLAT", "SHEET", "CLOTHING"],
    sampleable=False,
)

ZH_dimensions = Dimensions(
    [classifier_types, binary_order, binary_gender, binary_number]
)

chinese_article = Rule(
    "CLASSIFIER",
    ZH_dimensions,
    {
        "GENERIC": "个",
        "PAIR": "双",
        "ANIMAL": "只",
        "LONG_NARROW": "条",
        "FLAT": "张",
        "SHEET": "本",
        "CLOTHING": "件",
    },
)

ZH_rules = [chinese_article]

ZH = Language("CHINESE", "zh", ZH_dimensions, ZH_rules)

# SW
person = Dimension("PERSON", ["1", "2", "3"])
tense = Dimension("TENSE", ["PRS", "PST", "FUT"])
aspect = Dimension(
    "ASPECT", ["PRF", "PRV", "IPFV"]
)  # PERFECT, PERFECTIVE, IMPERFECTIVE

binary_number = Dimension("NUMBER", ["SG", "PL"])
swahili_noun_class = Dimension("NOUNSW", ["MWA", "NN"])
sw_noun_class = Dimension("NOUNCLASS", ["KIVI", "IZI", "IYA", "LIYA"])
sw_desc_class = Dimension("DESCCLASS", ["NI", "ANA"])
SW_dimensions = Dimensions([
    binary_number,
    person,
    tense,
    aspect,
    swahili_noun_class,
    binary_order,
    sw_noun_class,
    sw_desc_class,
])

sw_person_prefix_rule = Rule(
    "PERSON_PREFIX",
    SW_dimensions,
    {
        "1.SG": "ni",
        "2.SG": "u",
        "3.SG": "a",
        "1.PL": "tu",
        "2.PL": "m",
        "3.PL": "wa",
    },
)

sw_tense_prefix_rule = Rule(
    "TENSE_PREFIX",
    SW_dimensions,
    {"PST.PRV": "li", "PRS.PRV": "na", "FUT.PRV": "ta", "PRS.PRF": "ne"},
)

sw_where = Rule(
    "SWHERE",
    Dimensions([sw_noun_class, binary_number]),
    {
        "KIVI.SG": "iko",
        "KIVI.PL": "viko",
        "IZI.SG": "iko",
        "IZI.PL": "ziko",
        "IYA.SG": "iko",
        "IYA.PL": "yako",
    },
)

SW_rules = [sw_person_prefix_rule, sw_tense_prefix_rule, sw_where]

SW = Language("SWAHILI", "sw", SW_dimensions, SW_rules)

# RU

gender = Dimension("GENDER", ["FEM", "MASC", "NEUT"])
ternary_number = Dimension("NUMBER", ["SG", "PL", "GTPL"])
ru_animacy = Dimension("ANIMACY", ["ANIM", "INAN"])
ru_direction = Dimension("DIRECTION", ["D", "MD"])
ru_cases = Dimension("CASE", ["NOM", "GEN", "DAT", "ACC", "INS", "PREP"])


def russian_cons_fn(s):
  if (
      len(s) > 1
      and s[0].lower() not in "ауоыиэяюёе"
      and s[1].lower() not in "ауоыиэяюёе"
  ):
    return "CONS2"
  else:
    return "NO_CONS2"


binary_starts_with_c = Dimension(
    "STARTSWITH2C",
    ["NO_CONS2", "CONS2"],
    call=russian_cons_fn,
    sampleable=False,
)
RU_dimensions = Dimensions([
    gender,
    ternary_number,
    ru_cases,
    ru_direction,
    ru_animacy,
    binary_starts_with_c,
    binary_order,
])
RU_rules = []

RU = Language("RUSSIAN", "ru", RU_dimensions, RU_rules)

# AR

gender = Dimension("GENDER", ["FEM", "MASC"])
derived_gender = Dimension("DERIVED_GENDER", ["DFEM", "DMASC"])
# Number in Arabic: https://www.fluentarabic.net/numbers-in-arabic/
quaternary_number = Dimension("NUMBER", ["SG", "DU", "PL", "PAUC"])
arabic_cases = Dimension("CASE", ["NOM", "ACC"])
number_placement = Dimension("NBPLACE", ["BEF", "AFT"], sampleable=False)

AR_dimensions = Dimensions([
    gender,
    derived_gender,
    quaternary_number,
    arabic_cases,
    number_placement,
    binary_order,
])
AR_rules = []

AR = Language("ARABIC", "ar", AR_dimensions, AR_rules)

# ES

all_es_dimensions = Dimensions(
    [binary_article, binary_gender, binary_number, binary_order]
)

spanish_article = Rule(
    "ART",
    all_es_dimensions,
    {
        "SG.INDF.FEM": "una",
        "SG.INDF.MASC": "un",
        "PL.INDF": "",
        "SG.DEF.MASC": "el",
        "SG.DEF.FEM": "la",
        "PL.DEF.MASC": "los",
        "PL.DEF.FEM": "las",
    },
)

ES_rules = [spanish_article]
ES = Language("SPANISH", "es", all_es_dimensions, ES_rules)

# FI
fi_cases = Dimension("CASE", ["NOM", "GEN", "PRT", "ALL", "POS"])

all_fi_dimensions = Dimensions([binary_number, fi_cases, binary_order])

FI_rules = []
FI = Language("FINNISH", "fi", all_fi_dimensions, FI_rules)

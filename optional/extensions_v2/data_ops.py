"""Deterministic numeric MathQA adaptation and lexical contamination checks."""
import ast
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import html
import json
import math
from pathlib import Path
import random
import re
import unicodedata
import zipfile

SOURCE_URL = "https://math-qa.github.io/math-QA/data/MathQA.zip"
SOURCE_SHA256 = "7344f30456a7aef3176d4866cc953b35b41bec44eda6b00cdbcfde2876b2f07a"
EXTRA_COUNT = 2000
DATA_SEED = 20260927
NUMERIC = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/[+-]?\d+)?\Z")
NUMBERS = re.compile(r"(?<!\w)[+-]?\d+(?:\.\d+)?")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value_text(value):
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def numeric_option(row):
    text = row["options"]
    marks = list(re.finditer(r"(?:^|[,\s])([a-e])\s*\)", text, re.I))
    if [m.group(1).lower() for m in marks] != list("abcde"):
        raise ValueError("ambiguous_options")
    label = row["correct"].strip().lower()
    if label not in "abcde" or len(label) != 1:
        raise ValueError("invalid_label")
    i = ord(label) - ord("a")
    end = marks[i + 1].start() if i < 4 else len(text)
    raw = text[marks[i].end():end].strip().strip(",").strip()
    compact = re.sub(r"\s+", "", raw)
    if not NUMERIC.fullmatch(compact):
        raise ValueError("non_numeric_option")
    return Fraction(compact.replace(",", "")), raw


def formula_trace(text):
    """Interpret a small arithmetic whitelist; never eval dataset code."""
    tree = ast.parse(text, mode="eval")
    steps, literals = [], []
    operations = {"add": (lambda a, b: a + b, "+"), "subtract": (lambda a, b: a - b, "-"),
                  "multiply": (lambda a, b: a * b, "*"), "divide": (lambda a, b: a / b, "/")}

    def visit(node, depth=0):
        if depth > 16 or len(steps) > 12:
            raise ValueError("formula_too_long")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            number = Fraction(ast.get_source_segment(text, node))
            literals.append(number)
            return number
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            operand = visit(node.operand, depth + 1)
            return -operand if isinstance(node.op, ast.USub) else operand
        if isinstance(node, ast.Name) and re.fullmatch(r"const_\d+(?:_\d+)?", node.id):
            return Fraction(node.id[6:].replace("_", "."))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            name = node.func.id
            if name not in operations or len(node.args) != 2:
                raise ValueError("unsupported_operation")
            left, right = (visit(arg, depth + 1) for arg in node.args)
            fn, symbol = operations[name]
            result = fn(left, right)
            if abs(result) > 10**12 or result.denominator > 10**9:
                raise ValueError("extreme_result")
            steps.append(f"Step {len(steps) + 1}: ({value_text(left)}) {symbol} ({value_text(right)}) = {value_text(result)}.")
            return result
        raise ValueError("unsupported_formula")

    result = visit(tree.body)
    if not 1 <= len(steps) <= 12:
        raise ValueError("formula_too_long")
    return result, steps, literals


def adapt_row(row, index):
    question = html.unescape(row["Problem"]).strip()
    if not 5 <= len(question.split()) <= 160 or "####" in question:
        raise ValueError("invalid_question")
    if re.search(r"which.*following|which.*(?:true|false)|(?:above|below) statements|answer choices|option [a-e]", question, re.I):
        raise ValueError("choice_dependent_question")
    expected, option = numeric_option(row)
    result, steps, literals = formula_trace(row["annotated_formula"])
    if expected != result:
        raise ValueError("formula_answer_mismatch")
    question_numbers = {Fraction(v) for v in NUMBERS.findall(question)}
    if not set(literals) & question_numbers:
        raise ValueError("no_question_number_in_formula")
    return {"question": question, "answer": "\n".join(steps + [f"#### {value_text(expected)}"]),
            "external_id": f"mathqa-train-{index:05d}", "original_train_index": index,
            "correct_option": row["correct"], "numeric_option_text": option,
            "annotated_formula": row["annotated_formula"], "category": row["category"]}


def tokens(text):
    text = unicodedata.normalize("NFKC", html.unescape(text)).casefold()
    return re.findall(r"[a-z]+|\d+(?:\.\d+)?", text)


def masked_key(words):
    return tuple("NUMBER" if word[0].isdigit() else word for word in words)


class OverlapIndex:
    """Exact, numeral-masked, token-set and trigram Jaccard checks."""
    def __init__(self):
        self.records = []
        self.exact, self.masked = {}, {}
        self.word_index, self.gram_index = defaultdict(set), defaultdict(set)

    def add(self, question, identifier):
        words = tokens(question)
        wordset, grams = set(words), set(zip(words, words[1:], words[2:]))
        i = len(self.records)
        self.records.append((identifier, wordset, grams))
        self.exact.setdefault(tuple(words), i)
        self.masked.setdefault(masked_key(words), i)
        for word in wordset:
            self.word_index[word].add(i)
        for gram in grams:
            self.gram_index[gram].add(i)

    def match(self, question):
        words = tokens(question)
        key, mask = tuple(words), masked_key(words)
        for kind, table, query in (("normalized_exact", self.exact, key), ("number_masked_exact", self.masked, mask)):
            if query in table:
                return {"method": kind, "matched_id": self.records[table[query]][0], "similarity": 1.0}
        wordset = set(words)
        grams = set(zip(words, words[1:], words[2:]))
        for kind, items, lookup, threshold, field in (
                ("token_jaccard", wordset, self.word_index, .90, 1),
                ("trigram_jaccard", grams, self.gram_index, .80, 2)):
            if not items:
                continue
            counts = Counter()
            for item in items:
                counts.update(lookup.get(item, ()))
            for i, shared in sorted(counts.items()):
                if shared < math.ceil(threshold * len(items)):
                    continue
                identifier, *unused = self.records[i]
                other = self.records[i][field]
                score = shared / len(items | other)
                if score >= threshold:
                    return {"method": kind, "matched_id": identifier, "similarity": score}
        return None


def filter_external(archive, protected_questions, count=EXTRA_COUNT):
    if sha(archive) != SOURCE_SHA256:
        raise ValueError("MathQA source checksum differs.")
    with zipfile.ZipFile(archive) as zipped:
        # Only the official training member is read. No MathQA dev/test data.
        raw = json.loads(zipped.read("train.json"))
        training_sha = hashlib.sha256(zipped.read("train.json")).hexdigest()
    if len(raw) != 29837:
        raise ValueError("MathQA training size differs.")
    protected, unique = OverlapIndex(), OverlapIndex()
    for identifier, question in protected_questions:
        protected.add(question, identifier)
    rejected = Counter()
    overlaps, eligible = [], []
    order = list(range(len(raw)))
    random.Random(DATA_SEED).shuffle(order)
    for index in order:
        try:
            item = adapt_row(raw[index], index)
        except (ValueError, SyntaxError, ZeroDivisionError, KeyError, TypeError, OverflowError) as exc:
            rejected[str(exc) if isinstance(exc, ValueError) else type(exc).__name__] += 1
            continue
        match = protected.match(item["question"])
        if match:
            rejected["protected_question_overlap"] += 1
            overlaps.append({"external_id": item["external_id"], **match})
            continue
        match = unique.match(item["question"])
        if match:
            rejected["internal_duplicate_or_near_duplicate"] += 1
            continue
        unique.add(item["question"], item["external_id"])
        eligible.append(item)
    if len(eligible) < count:
        raise ValueError(f"Only {len(eligible)} eligible MathQA training records; need the fixed {count}.")
    selected = eligible[:count]
    for item in selected:
        if protected.match(item["question"]) is not None:
            raise ValueError("Selected external data overlaps the protected question set.")
    audit = {"dataset": "MathQA", "source_url": SOURCE_URL, "source_zip_sha256": SOURCE_SHA256,
             "source_train_member_sha256": training_sha, "source_split": "train", "source_count": len(raw),
             "sampling_seed": DATA_SEED, "eligible_unique_count": len(eligible), "selected_count": count,
             "rejected_counts": dict(sorted(rejected.items())), "overlap_rejections": overlaps,
             "protected_question_count": len(protected_questions),
             "numeric_policy": "Correct choice must be a single unit-free number/fraction and exactly match a whitelist arithmetic program.",
             "target_policy": "Question without options; worked arithmetic trace derived from annotated_formula, then #### numeric answer. Original noisy rationale is excluded.",
             "overlap_policy": "Normalized exact; number-masked exact; token-set Jaccard >= 0.90; word-trigram Jaccard >= 0.80. Also applied within the external pool.",
             "scope": "Lexical checks cover the supplied GSM8K/SVAMP questions. Paraphrased overlap and logically wrong but numerically matching source programs can remain.",
             "category_counts": dict(Counter(item["category"] for item in selected))}
    return selected, audit


def clean_gsm(training, extract):
    cleaned, changed, removed = [], 0, 0
    for row in training:
        answer = row["answer"]
        count = len(re.findall(r"<<[^<>\n]*>>", answer))
        new = re.sub(r"<<[^<>\n]*>>", "", answer)
        if extract(answer)[1] is None or extract(new)[1] != extract(answer)[1]:
            raise ValueError("Cleaning changed or invalidated a gold final answer.")
        cleaned.append({"question": row["question"], "answer": new})
        changed += new != answer
        removed += count
    return cleaned, {"n_examples": len(training), "changed_targets": changed,
                     "removed_calculator_annotations": removed, "gold_answers_preserved": True,
                     "policy": "Remove only <<calculator annotations>> from GSM8K training targets; retain prose, arithmetic and final answers. Validation/test stay unchanged."}

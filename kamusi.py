import code
import json
import pdb
import random
import re
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from functools import cache, partial
from itertools import dropwhile, takewhile
from operator import sub
from textwrap import dedent
from typing import Any, TypedDict, Optional, TypeAlias

import pdfquery
from icecream import ic
from loguru import logger
from pypdf import PdfReader
from rich.pretty import pprint
from statemachine import State, StateMachine
from tabulate import tabulate
from toolz import curry, flip, keyfilter, pipe
from tqdm import tqdm


class TextPiece(TypedDict):
    font: str
    text: str


Stream: TypeAlias = list[TextPiece]


@dataclass
class WordEntry:
    pos: str = None
    english: str = None
    swahili: str = None


@dataclass
class DictEntry:
    swahili: str = ""  # swahili word
    swahili_alt: Optional[str] = (
        None  # alternate form of the swahili word with same meaning
    )
    part_of_speech: str = ""  # part of speech tag
    plural: Optional[str] = None  # plural form of swahili noun word
    ngeli: Optional[str] = None  # ngeli of noun word if one exists
    english: list[str] = field(
        default_factory=list
    )  # english meaning(s) of swahili word
    examples: Optional[list[tuple[str, str]]] = (
        None  # example usages in form (swahili text, english text)
    )
    alternates: Optional[list[tuple[str, str]]] = (
        None  # alternate forms of swahili verb word
    )


def save_dictionary(words: [WordEntry], dict_file="data/dict.jsonl") -> None:
    with open(dict_file, "w") as fp:
        for word in words:
            json.dump(asdict(word), fp)
            fp.write("\n")
    return


def show_entry(v: pdfquery.pdfquery.LayoutElement) -> None:
    tbl = [["text", v.text]]
    tbl += [[k, v] for k, v in v.attrib.items()]
    print(tabulate(tbl, tablefmt="simple"))


def nearest_lookup(
    elems: [pdfquery.pdfquery.LayoutElement],
    v: pdfquery.pdfquery.LayoutElement,
    dim="y0",
):
    min_val = elems[0]
    for elem in elems[1:]:
        ya = float(elem.attrib["y0"])
        yb = float(min_val.attrib["y0"])
        if abs(float(v.attrib[dim]) - ya) < abs(float(v.attrib[dim]) - yb):
            min_val = elem
    return min_val


# process 'Kiingereza_Swahili - Kamusi ya kiswahili na kiingereza kwa .pdf'
def process_ki_swa(pdf_obj: pdfquery.PDFQuery) -> [WordEntry]:
    page_nums = pdf_obj.tree.xpath("//*/LTPage/@pageid")
    page_elements = [
        pdf_obj.tree.xpath(
            f"//*/LTPage[@pageid='{page}']//LTTextLineHorizontal"
        )
        for page in page_nums
    ]
    words = []
    for k, elements in enumerate(page_elements):
        swa = [i for i in elements if 295 < float(i.attrib["x0"]) < 315]
        eng = [i for i in elements if 130 < float(i.attrib["x0"]) < 250]
        pos = [i for i in elements if 70 < float(i.attrib["x0"]) < 90]
        for word in swa:
            entry = None
            swahili_text = word.text.strip()
            english_word = nearest_lookup(eng, word)
            english_word_text = english_word.text.strip()
            pos_tag = nearest_lookup(pos, word)
            pos_tag_text = pos_tag.text.strip()
            if not re.match("^\d\.", english_word_text):
                entry = WordEntry(pos_tag_text, english_word_text, swahili_text)
            else:
                idx = eng.index(english_word)
                # ic(word.text)
                candidates = filter(
                    lambda x: not re.match(r"^\d\.", x.text),
                    reversed(eng[:idx]),
                )
                try:
                    actual_word = next(candidates)
                except StopIteration:
                    eng = [
                        i
                        for i in page_elements[k - 1]
                        if 130 < float(i.attrib["x0"]) < 250
                    ]
                    candidates = filter(
                        lambda x: not re.match(r"^\d\.", x.text), reversed(eng)
                    )
                    actual_word = next(candidates)
                english_word_text = (
                    f"{actual_word.text.strip()} ({english_word_text})"
                )
                entry = WordEntry(pos_tag_text, english_word_text, swahili_text)
            words.append(entry)
    return words


def visitor_body(
    text: str,
    cm: [float],
    tm: [float],
    font_dict: dict[str, Any],
    font_size: float,
    page_no: int,
    parts: [Any],
):
    if cleaned_text := str.strip(text):
        parts.append(
            {
                "text": cleaned_text,
                "font": font_dict["/BaseFont"],
                "dimensions": tm,
                "page": page_no,
            }
        )
    return


def process_pdf():
    reader = PdfReader(
        "kamusi/Mulokozi M.M. - Tuki kamusi ya kiswahili-kiingereza. Swahili-English dictionary.pdf"
    )
    parts = []
    for i, page in enumerate(reader.pages):
        page.extract_text(
            visitor_text=partial(visitor_body, page_no=i, parts=parts)
        )
    with open("data/pdf_parts.jsonl", "w") as fp:
        for part in parts:
            json.dump(part, fp)
            fp.write("\n")
        logger.info("wrote parts to data/pdf_parts.jsonl")
    return parts


def is_italic(piece: TextPiece) -> bool:
    return piece["font"].endswith("Italic")


def is_bold(piece: TextPiece) -> bool:
    return piece["font"].endswith("Bold")


def is_normal(piece: TextPiece) -> bool:
    return not (is_italic(piece) or is_bold(piece))


def as_text(stream: Stream) -> str:
    return " ".join(texts(stream))


@cache
def read_parts_as_entries() -> Stream:
    with open("data/word_indices.txt", "r") as fp:
        indices = fp.readlines()
        indices = pipe(
            indices,
            curry(map, str.strip),
            curry(map, int),
            curry(map, flip(sub, 1)),
            list,
        )
    with open("data/pdf_parts.jsonl", "r") as fp:
        lines = [json.loads(line) for line in fp.readlines()]

    entries = []
    for i, k in enumerate(indices):
        if i == len(indices) - 1:
            entry = lines[k:]
        else:
            entry = lines[k : indices[i + 1]]
        entries.append(
            pipe(
                entry,
                curry(map, curry(keyfilter, lambda x: x in ["text", "font"])),
                list,
            )
        )
    return entries


def texts(stream: Stream) -> list[str]:
    return [x["text"] for x in stream]


class DictEntryModel:
    def __init__(self, stream: Stream):
        self.stream = stream
        self.text = as_text(stream)
        self.entry: DictEntry = DictEntry()
        self.parse_ok = True
        self.state = None

    def __repr__(self):
        return (
            self.text
            + "\n\n"
            + tabulate(asdict(self.entry).items(), ["key", "value"], "simple")
        )

    def head_bold(self):
        return is_bold(self.stream[0])

    def extract_swahili(self):
        stream = deepcopy(self.stream)
        word = stream[0]
        # read word if broken up into multiple text blocks
        rest = list(
            takewhile(lambda x: is_bold(x) or x["text"] == ".", stream[1:])
        )
        swa_word = pipe(
            word["text"] + "".join([i["text"] for i in rest]),
            str.strip,
            str.lower,
        )
        begin = 1 + len(rest)
        remaining = []
        for i, v in enumerate(stream[begin:]):
            if not is_normal(v):
                # already finished swahili word section so stop processing and return remaining stream
                remaining = remaining + stream[i + begin :]
                break
            t = pipe(v["text"], str.strip)
            # remove number indicating multiple meanings or '*'
            if re.match("\*|\d|\*\d", t):
                continue
            else:
                remaining.append(v)
        # already found swahili word, add alternate meaning instead
        if self.entry.swahili:
            self.entry.swahili_alt = swa_word
        else:
            self.entry.swahili = swa_word
        self.stream = remaining

    def has_extra(self):
        w1 = self.stream[0]
        return is_normal(w1) and re.match(r"\s*pia\s*", w1["text"])

    def extract_extra(self):
        if self.head_bold():
            self.extract_swahili()
        else:
            # logger.error(
            #     "expected pia form got {} on {}", self.stream, self.entry
            # )
            self.parse_ok = False

    def extract_pos(self):
        # FIXME: handle cases where no POS see test_cases
        w1 = self.stream[0]
        pos = re.split(r"\s+", w1["text"])
        if not re.match(r"\s*[a-z]{2}\s*", pos[0]):
            self.parse_ok = False
            # logger.error("Unable to find part of speech for {}", self.text)
            return
        if len(pos) == 1:
            self.entry.part_of_speech = pos[0]
            self.stream = self.stream[1:]
        else:
            self.entry.part_of_speech = pos[0]
            self.stream = [{"font": w1["font"], "text": pos[1]}] + self.stream[
                1:
            ]

    def has_plural(self):
        w1 = self.stream[0]
        w2 = self.stream[1]
        # plural must only occur when ngeli is present rest is used to check for ngeli
        rest = list(
            dropwhile(
                lambda x: re.match(r"\s*[-)\[]+\s*", x["text"]), self.stream[2:]
            )
        )
        rest_ = list(
            takewhile(lambda x: "[" in x["text"] or is_italic(x), rest)
        )
        if "(" in w1["text"] and is_italic(w2):
            return True
        if is_italic(w1) and len(rest_) > 2:
            return True
        return False

    def extract_plural(self):
        stream = deepcopy(self.stream)
        # plural
        end = 0
        for i, v in enumerate(stream):
            end = i
            if "(" in v["text"]:
                continue
            if is_italic(v):
                if i < len(stream) and is_italic(stream[i + 1]):
                    plural = [v, stream[i + 1]]
                    end = i + 1
                else:
                    plural = [v]
                break
        plural_txt = pipe(
            "".join(texts(plural)), curry(re.sub, r"[-(]", ""), str.strip
        )
        stream = list(dropwhile(lambda x: "-" in x["text"], stream[end + 1 :]))
        self.stream = stream
        self.entry.plural = plural_txt

    def is_noun(self):
        return self.entry.part_of_speech == "nm"

    def extract_ngeli(self):
        stream = deepcopy(self.stream)
        stream = list(
            dropwhile(lambda x: re.match(r"\s*-\*s", x["text"]), stream)
        )
        w1 = stream[0]
        if "[" in w1["text"]:
            ngeli_toks = list(takewhile(lambda x: "]" not in x, texts(stream)))
        elif is_italic(w1):
            ngeli_toks = texts(list(takewhile(lambda x: is_italic(x), stream)))
        else:
            self.parse_ok = False
            return
        ngeli = pipe(
            "".join(ngeli_toks),
            curry(re.sub, r"\s+", ""),
            curry(re.sub, r"^[)\]]", ""),
        )
        patt = re.compile(r"\[?[a-z]{1,2}-(?:/[a-z]{2}-)?")
        if m := re.match(patt, ngeli):
            self.entry.ngeli = re.sub(r"\[|/", "", m.string)
            stream = stream[len(ngeli_toks) :]
        self.stream = stream

    def has_multiple_defs(self):
        return any(
            [
                re.match(r"\.?\s*(?:[1-9]|\([1-9]\))\s*[a-zA-Z]+", i)
                for i in texts(self.stream)
            ]
        )

    def extract_english(self):
        # FIXME: handle cases where multiple definitions have different pos_tag. see examples
        stream = deepcopy(self.stream)
        stream = list(
            dropwhile(lambda x: "[" in x["text"] or is_italic(x), stream)
        )
        defs = list(takewhile(lambda x: is_normal(x), stream))
        txt = pipe(
            "".join(texts(defs)), curry(re.sub, r"^\s*]\s*", ""), str.strip
        )
        mult = list(filter(str.strip, re.split(r"\d\s*", txt)))
        # check if we've already found another definition previously
        if self.entry.english:
            self.entry.english.append(mult[0])
        else:
            self.entry.english = [mult[0]]
        # handle other definitions by placing them back on the stream for processing
        if len(mult) > 1:
            extra = [
                {"text": f"{i + 2} {x}", "font": "/Times New Roman"}
                for i, x in enumerate(mult[1:])
            ]
            self.stream = extra + stream[len(defs) :]
            # ic(self.stream)
        else:
            self.stream = stream[len(defs) :]
        self.stream = list(
            dropwhile(
                lambda x: (is_italic(x) and re.match(r"\(?ms\)?", x["text"]))
                or re.match(r"\s*\(\s*|\s*\)\s*", x["text"]),
                self.stream,
            )
        )

    def has_example(self):
        head = list(takewhile(is_italic, self.stream))
        txt = "".join(texts(head))
        swa = self.entry.swahili.lower()
        plural = self.entry.plural
        if head:
            if (
                "~" in txt
                or swa in txt.lower()
                or (plural and plural.lower() in txt.lower())
            ):
                return True
        return False

    def extract_example(self):
        stream = deepcopy(self.stream)
        examples = []
        while True:
            # find the swahili example and it's english equivalent
            swa_ex = list(takewhile(is_italic, stream))
            eng_ex = list(takewhile(is_normal, stream[len(swa_ex) :]))
            swa_ex_txt = "".join(texts(swa_ex))
            eng_ex_txt = "".join(texts(eng_ex))
            joined_txt = list(filter(str.strip, re.split(r"\.\s*", eng_ex_txt)))
            if len(joined_txt) > 1:
                # if example is combined with something else, place that back on the stream
                eng_ex_txt = joined_txt[0]
                stream = [
                    {
                        "text": "".join(joined_txt[1:]),
                        "font": "/Times New Roman",
                    }
                ] + stream[len(eng_ex) + len(swa_ex) :]
            else:
                stream = stream[len(eng_ex) + len(swa_ex) :]
            examples.append((swa_ex_txt, eng_ex_txt))
            if ";" in eng_ex_txt:
                continue
            else:
                break
        if self.entry.examples:
            self.entry.examples.extend(examples)
        else:
            self.entry.examples = examples
        self.stream = stream

    def has_alternates(self):
        return any(
            [
                is_italic(v) and re.match(r"\(?td[a-z]{1,2}\)?", v["text"])
                for v in self.stream
            ]
        )

    def extract_alternates(self):
        stream = deepcopy(self.stream)
        indices = []
        altkeys = []
        altvals = []
        for i, v in enumerate(stream):
            if is_italic(v) and re.match(r"\(?td[a-z]{1,2}\)?", v["text"]):
                indices.append(i)
                altkeys.append(v["text"].strip("()"))
        end = 0
        for i in indices:
            key = next((i for i in stream[i:] if is_bold(i)), None)
            if key is None:
                continue
            else:
                end = stream.index(key)
                altvals.append(key["text"])
        self.entry.alternates = list(zip(altkeys, altvals))
        self.stream = stream[end + 1 :] if end < len(stream) else []


class DictEntryMachine(StateMachine):
    start = State(initial=True)
    swahili = State(enter="extract_swahili")
    extra = State(enter="extract_extra")
    pos = State(enter="extract_pos")
    ngeli = State(enter="extract_ngeli")
    plural = State(enter="extract_plural")
    english = State(enter="extract_english")
    examples = State(enter="extract_example")
    alternates = State(enter="extract_alternates")
    stop = State(final=True)

    parse = (
        start.to(swahili, cond="head_bold")
        | swahili.to(extra, cond=["has_extra"])
        | swahili.to(pos)
        | extra.to(pos)
        | pos.to(plural, cond=["is_noun", "has_plural"])
        | pos.to(ngeli, cond="is_noun")
        | plural.to(ngeli)
        | pos.to(english)
        | ngeli.to(english)
        | english.to(examples, cond="has_example")
        | english.to(alternates, cond="has_alternates")
        | english.to(english, cond="has_multiple_defs")
        | english.to(stop)
        | examples.to(english, cond="has_multiple_defs")
        | examples.to(alternates, cond="has_alternates")
        | examples.to(stop)
        | alternates.to(stop)
    )


def process(entry):
    model = DictEntryModel(entry)
    # dc = DictEntryControl(model, listeners=[LogListener("debug:")])
    machine = DictEntryMachine(model)
    while (
        machine.current_state not in machine.final_states
        and machine.model.parse_ok
    ):
        machine.parse()
    return model


def main():
    print("extracting entries")
    entries = read_parts_as_entries()
    stats = {"succeeded": 0, "failed": 0}
    fp = open("data/swahili-english-dict.jsonl", "w")
    fp2 = open("data/failed_stream.jsonl", "w")
    id_ = 1
    for entry in tqdm(entries):
        try:
            de = process(entry)
            if de.parse_ok and len(de.stream) < 3:
                d = asdict(de.entry)
                d["id"] = id_
                s = json.dumps(d) + "\n"
                fp.write(s)
                stats["succeeded"] += 1
                id_ += 1
            else:
                s = json.dumps(entry) + "\n"
                fp2.write(s)
                stats["failed"] += 1
        except Exception as e:
            s = json.dumps(entry) + "\n"
            fp2.write(s)
            stats["failed"] += 1
    fp.close()
    fp2.close()
    pprint(stats)


class LogListener(object):
    def __init__(self, name):
        self.name = name

    def after_transition(self, event, source, target):
        print(f"{self.name} after: {source.id}--({event})-->{target.id}")

    def on_enter_state(self, target, event):
        print(f"{self.name} enter: {target.id} from {event}")


def process_debug(entry):
    model = DictEntryModel(entry)
    machine = DictEntryMachine(model, listeners=[LogListener("debug:")])
    while (
        machine.current_state not in machine.final_states
        and machine.model.parse_ok
    ):
        machine.parse()
    return model, machine


def run(sample_size=10):
    print(
        dedent("""
    r - regenerate sample
    n - process next entry
    i - inspect current entry
    w - write current entry to data/test_cases.jsonl
    """)
    )
    entries = read_parts_as_entries()
    sample = random.sample(entries, sample_size)
    next_ = None
    model = None
    sm = None
    while command := input("> "):
        print(f"{len(sample)} of {sample_size} left\n")
        match command[0].lower():
            case "r":
                sample = random.sample(entries, sample_size)
                continue
            case "n":
                if not sample:
                    print("sample done. Use [r] to generate new sample")
                    continue
                next_ = sample.pop()
                print(f"processing:\n\t{as_text(next_)}")
                model, sm = process_debug(next_)
            case "i":
                code.interact(
                    banner="process inspect REPL.\n try:\n\twat / model\n\twat / sm",
                    local=locals() | globals(),
                )
            case "w":
                print("writing example to test_cases\n")
                with open("data/test_cases.jsonl", "a") as fp:
                    s = json.dumps(next_) + "\n"
                    fp.write(s)
    return


pos_map = {
    "nm": "nomino (noun)",
    "kt": "kitenzi (verb)",
    "kv": "kivumishi (adjective)",
    "kl": "kielezi (adverb)",
    "kw": "kiwakilishi (pronoun)",
    "ki": "kiingizi (interjection)",
    "ku": "kiunganishi (conjunction)",
}


def well_formed(entry):
    result = True
    result = (
        result
        and all([entry[i] for i in ["part_of_speech", "swahili", "english"]])
        and all(entry["english"]),
    )
    if entry["ngeli"]:
        result = result and bool(
            re.match(r"[a-z]{1,2}-(?:[a-z]{1,2}-)?", entry["ngeli"])
        )
    result = result and entry["part_of_speech"] in (
        set(pos_map.keys()) | set(pos_map.values())
    )
    return result


def transform_swahili(s):
    return pipe(
        s,
        lambda x: x.replace(".", ""),
        str.strip,
    )


def transform_swahili_alt(s):
    return transform_swahili(s)


def transform_part_of_speech(s):
    return pos_map[s]


def transform_english(xs):
    rs = list(
        map(
            lambda s: pipe(
                s,
                curry(re.sub, r"\s*[-():;]\s*$", ""),
                curry(re.sub, r"^\s*[-():;]\s*", ""),
                curry(re.sub, r"^\s*[-()]", ""),
                curry(
                    re.sub, r"\s*\(?(?:Kar|Kre|Khi|Kya|Kaj|Kng)\)?\s*\.?$", ""
                ),
                str.strip,
                lambda x: x + "." if not re.match(r"\.\s*$", x) else x,
                curry(re.sub, r"\.{2,}", "."),
            ),
            xs,
        )
    )
    return rs


def transform_examples(xs, word):
    rs = list(
        map(
            lambda s: pipe(
                s[0],
                lambda x: x.replace("~", word),
                curry(re.sub, r"\s*[-():;]\s*$", ""),
                curry(re.sub, r"^\s*[-():;]\s*", ""),
                curry(re.sub, r"\s*\([0-9]+\)\s*$", ""),
                str.strip,
                lambda x: [x, s[1]],
            ),
            xs,
        )
    )
    return rs


def transform(entry):
    res = deepcopy(entry)
    for k, v in res.items():
        if v:
            if f := globals().get(f"transform_{k}", False):
                if k == "examples":
                    res[k] = f(v, res["swahili"])
                else:
                    res[k] = f(v)
                continue
    return res


def run_transformations():
    entries = read_entries()
    id_ = 1
    print("running transformations")
    with open("data/swahili-english-dict.jsonl", "w") as fp:
        for entry in tqdm(entries):
            if well_formed(entry):
                entry["id"] = id_
                id_ += 1
                s = json.dumps(transform(entry))
                fp.write(s)
                fp.write("\n")
    return


def read_entries(file="data/swahili-english-dict.jsonl"):
    results = []
    with open(file, "r") as fp:
        while line := fp.readline():
            r = json.loads(line)
            results.append(r)
    return results

if __name__ == '__main__':
    main()
    run_transformations()

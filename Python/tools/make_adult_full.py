"""Turns the two UCI Adult files (adult.data and adult.test) into the
comma separated format of ../../adult.txt, so SubDisc can load them.

    python tools/make_adult_full.py <folder with adult.data and adult.test> [output folder]

What changes on the way:
- the header line of adult.txt is added
- the space after every comma goes
- '?' becomes an empty field, the SubDisc loader reads that as missing
  (it would keep '?' as an ordinary value otherwise)
- <=50K / >50K become leq50K / gr50K like in adult.txt, the trailing
  '.' of the test file goes too
- adult.test starts with a '|1x3 Cross validator' line, it is skipped
"""

import os
import sys

HEADER = ("age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,"
          "race,sex,capital-gain,capital-loss,hours-per-week,native-country,target")
NR_FIELDS = 15


def convert_line(line):
    fields = [f.strip() for f in line.split(",")]
    if len(fields) != NR_FIELDS:
        return None
    label = fields[-1].rstrip(".")
    if label == "<=50K":
        fields[-1] = "leq50K"
    elif label == ">50K":
        fields[-1] = "gr50K"
    else:
        return None
    fields = ["" if f == "?" else f for f in fields]
    return ",".join(fields)


def convert(src, dst):
    kept = skipped = 0
    with open(src, encoding="utf-8") as r, open(dst, "w", encoding="utf-8", newline="") as w:
        w.write(HEADER + "\n")
        for line in r:
            line = line.strip()
            if not line or line.startswith("|"):
                continue
            out = convert_line(line)
            if out is None:
                skipped += 1
                continue
            w.write(out + "\n")
            kept += 1
    return kept, skipped


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    raw = argv[1]
    out = argv[2] if len(argv) > 2 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(out, exist_ok=True)
    for name, target in (("adult.data", "adult_full_train.txt"), ("adult.test", "adult_full_test.txt")):
        kept, skipped = convert(os.path.join(raw, name), os.path.join(out, target))
        print("%s -> %s: %d rows, %d lines skipped" % (name, target, kept, skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

"""Writes data/survival_demo.txt, a made up clinical data set for the
survival target type. Nothing in it is real.

    python tools/make_survival_demo.py [rows] [seed]

Hidden in it on purpose (so one can check that the search finds it):
patients on treatment B with biomarker >= 115 have a 3.5 times higher
hazard. Age, stage and smoking also matter a bit, region is pure noise.

Times are in months. Follow up stops at 60 months, and on top of that some
patients drop out early; those rows are censored (event = 0).
"""

import math
import os
import random
import sys


def make(nr_rows=1500, seed=7):
    rnd = random.Random(seed)
    stages = (("I", 0.30, 0.0), ("II", 0.30, 0.25), ("III", 0.25, 0.5), ("IV", 0.15, 0.8))
    rows = []
    for _ in range(nr_rows):
        age = rnd.randint(30, 85)
        sex = "Female" if rnd.random() < 0.5 else "Male"
        u = rnd.random()
        acc = 0.0
        stage, stage_effect = stages[-1][0], stages[-1][2]
        for name, p, eff in stages:
            acc += p
            if u < acc:
                stage, stage_effect = name, eff
                break
        treatment = "A" if rnd.random() < 0.5 else "B"
        biomarker = round(rnd.gauss(100.0, 15.0), 1)
        smoker = 1 if rnd.random() < 0.3 else 0
        region = rnd.choice(("North", "South", "East", "West"))

        lp = 0.02 * (age - 60) + stage_effect + 0.3 * smoker
        if treatment == "B" and biomarker >= 115.0:
            lp += math.log(3.5)
        rate = 0.015 * math.exp(lp)
        t_event = rnd.expovariate(rate)
        t_drop = rnd.expovariate(0.008)
        t_censor = min(60.0, t_drop)
        if t_event <= t_censor:
            time, event = t_event, 1
        else:
            time, event = t_censor, 0
        time = max(0.1, round(time, 1))
        rows.append((age, sex, stage, treatment, biomarker, smoker, region, time, event))
    return rows


def main(argv):
    n = int(argv[1]) if len(argv) > 1 else 1500
    seed = int(argv[2]) if len(argv) > 2 else 7
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "survival_demo.txt")
    rows = make(n, seed)
    with open(path, "w", encoding="utf-8", newline="") as w:
        w.write("age,sex,stage,treatment,biomarker,smoker,region,time,event\n")
        for r in rows:
            w.write(",".join(str(v) for v in r) + "\n")
    events = sum(r[-1] for r in rows)
    print("%s: %d rows, %d events, %d censored" % (path, len(rows), events, len(rows) - events))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

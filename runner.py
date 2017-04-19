import datetime
import random
from typing import List


def writelines(lines: List[str], fname: str):
    lines = [l + "\n" for l in lines]
    with open(fname, "w") as f:
        f.writelines(lines)


def create_data(prefix: str, N: int, create_vocab: bool):
    startdate = datetime.date(1900, 1, 1)
    target = []
    source = []
    for i in range(N):
        date = startdate + datetime.timedelta(random.randint(0, 200 * 365))
        target.append(date.isoformat())
        source.append(date.strftime("%d %b %Y"))

    writelines(source, "%s-source.txt" % prefix)
    writelines(target, "%s-target.txt" % prefix)
    if create_vocab:
        vocab = list(set("".join(target)).union(set("".join(source))))
        writelines(vocab, "vocab.txt")


create_data("train", 10000, True)
create_data("dev", 1000, False)

import datetime
import random
from typing import List

import babel
from babel.dates import format_date


def writelines(lines: List[str], fname: str):
    lines = [l + "\n" for l in lines]
    with open(fname, "w") as f:
        f.writelines(lines)


formats = ["short", "medium", "long"]
locales = babel.localedata.locale_identifiers()


def create(prefix: str, N: int, create_vocab: bool):
    startdate = datetime.date(1900, 1, 1)
    target = []
    source = []
    for i in range(N):
        date = startdate + datetime.timedelta(random.randint(0, 200 * 365))
        target.append(date.isoformat())

        datestr = format_date(date, format=random.choice(formats), locale=random.choice(locales))
        source.append(datestr)

    writelines(source, "/tmp/%s-source.txt" % prefix)
    writelines(target, "/tmp/%s-target.txt" % prefix)
    if create_vocab:
        writelines(list(set("".join(source))), "/tmp/vocab-source.txt")
        writelines(list(set("".join(target))), "/tmp/vocab-target.txt")
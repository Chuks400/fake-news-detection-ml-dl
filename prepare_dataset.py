import csv
import re
from pathlib import Path

def clean(value):
    # Preserve a one-line, safely delimited format compatible with the original parsers.
    return re.sub(r"\s+", " ", (value or "").replace("'", " ")).strip()

def main():
    count = 0
    with Path("fake_news.arff").open("w", encoding="utf-8", newline="\n") as out:
        out.write("@RELATION fake_news\n@ATTRIBUTE news_text STRING\n@ATTRIBUTE publish_date STRING\n@ATTRIBUTE class_label {fake,real}\n@DATA\n")
        for filename, label in (("Fake.csv", "fake"), ("True.csv", "real")):
            with Path(filename).open(encoding="utf-8-sig", newline="") as source:
                rows = csv.DictReader(source)
                if not rows.fieldnames or "text" not in rows.fieldnames:
                    raise ValueError(f"{filename} must contain a text column")
                for row in rows:
                    text = clean(row.get("text"))
                    date = clean(row.get("date"))
                    if text:
                        out.write(f"'{text}','{date}',{label}\n")
                        count += 1
    print(f"Wrote {count} examples to fake_news.arff")

if __name__ == "__main__":
    main()

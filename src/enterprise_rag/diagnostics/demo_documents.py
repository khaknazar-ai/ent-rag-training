"""Generate authored synthetic demo inputs; never download or overwrite data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def make_spanning_pdf(*, open_sides=True, bottom_rule=True, rotated=False):
    from io import BytesIO

    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = writer._add_object(
        DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )
    commands = []
    for y in [650, 590, 560] + ([530] if bottom_rule else []):
        commands.append(f"50 {y} m 550 {y} l S")
    commands.append("170 620 m 430 620 l S")
    for x in [170, 430] + ([] if open_sides else [50, 550]):
        commands.append(f"{x} 530 m {x} 650 l S")
    commands.append("300 530 m 300 620 l S")
    labels = [
        (65, 630, "Key"),
        (205, 633, "Metrics"),
        (455, 630, "Outcome"),
        (190, 601, "Rate"),
        (320, 601, "Score"),
        (65, 574, "Alpha"),
        (190, 574, "11"),
        (320, 574, "22"),
        (455, 574, "Ready"),
        (65, 544, "Beta"),
        (190, 544, "33"),
        (320, 544, "44"),
        (455, 544, "Hold"),
    ]
    for x, y, text in labels:
        commands.append(f"BT /F1 10 Tf {x} {y} Td ({text}) Tj ET")
    stream = DecodedStreamObject()
    stream.set_data(" ".join(commands).encode())
    page[NameObject("/Contents")] = writer._add_object(stream)
    if rotated:
        page.rotate(90)
    out = BytesIO()
    writer.write(out)
    return out.getvalue()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.exit(2, "Choose a new output directory; existing files are preserved\n")
    try:
        content = make_spanning_pdf()
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "matrix.pdf").write_bytes(content)
        (args.output / "welcome.txt").write_text(
            "Demo handbook\nThe safecanoe export procedure requires signed approval.\nThis is synthetic demonstration content.\n",
            encoding="utf-8",
        )
        (args.output / "questions.json").write_text(
            json.dumps(
                {
                    "synthetic": True,
                    "mode": "Page answer · local",
                    "questions": [
                        {"question": "Which Rate is listed for Alpha?", "answer": "11"},
                        {
                            "question": "Which Outcome is listed for Beta?",
                            "answer": "Hold",
                        },
                        {
                            "question": "Which Metrics is listed for Alpha?",
                            "abstain": True,
                        },
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except (OSError, ImportError) as error:
        parser.exit(2, str(error) + "\n")
    print(
        json.dumps(
            {
                "status": "CREATED_SYNTHETIC_INPUTS",
                "output": str(args.output.resolve()),
                "model_calls": 0,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

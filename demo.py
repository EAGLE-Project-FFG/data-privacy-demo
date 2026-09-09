"""Anonymization and structure-preserving synthetic generation demo.

Run `uv run python demo.py` and open output/report.html. Needs OPENAI_API_KEY in
Anonymization/.env for the two LLM stages.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

from llm import infill, judge
from pipeline import (
    TEXT,
    infer_schema,
    key_candidates,
    plurel_schema,
    read_tables,
    sql_schema,
    sqlite_export,
    validate,
    write_json,
    write_tables,
)
from privacy import anonymize
from report import write_report
from synthesis import describe, fit, synthesize

HERE = Path(__file__).resolve().parent
# Files and directories this demo writes.
ARTIFACTS = {
    "report.html",
    "schema.sql",
    "plurel_schema.sql",
    "summary.json",
    "schema_decision.json",
    "pseudonym_map.json",
    "plurel.log",
    "source",
    "anonymized",
    "stringless",
    "synthetic",
}


def prepare(output):
    """Reuse the same output directory every run, without deleting other files."""
    if output.exists():
        unexpected = sorted(p.name for p in output.iterdir() if p.name not in ARTIFACTS)
        if unexpected:
            raise ValueError(f"{output} holds files this demo did not write: {unexpected[0]}")
        for name in ARTIFACTS:
            stale = output / name
            if stale.is_dir():
                shutil.rmtree(stale)
            elif stale.exists():
                stale.unlink()
    output.mkdir(parents=True, exist_ok=True)


def run(args):
    data = args.data.resolve()
    output = args.out.resolve()
    started = time.monotonic()

    source = read_tables(data)
    validate(source)
    sizes = ", ".join(f"{len(rows)} {table}" for table, rows in source.items())
    print(
        "EAGLE synthetic-data demo\n"
        f"  Code:   {HERE}\n"
        f"  Input:  {data} ({sizes})\n"
        f"  Output: {output}\n"
    )

    # Do not clear a previous run until the new input has been read and checked.
    prepare(output)
    write_tables(output / "source", source)

    print("1/3  Protect attribute values")
    anonymized, aliases = anonymize(source)
    validate(anonymized)
    write_tables(output / "anonymized", anonymized)
    write_json(output / "pseudonym_map.json", aliases.mapping)
    print(
        "        "
        + ", ".join(f"{len(v)} {k} pseudonyms" for k, v in aliases.mapping.items())
        + "; costs distorted, dates shifted, addresses masked, "
        "descriptions deleted, cost_center removed"
    )

    print("2/3  Generate new relational structure")
    print("     Induce the schema: heuristics propose candidates, the LLM judges")
    candidates = key_candidates(anonymized)
    decision = judge(anonymized, candidates)
    schema = infer_schema(anonymized, decision)
    (output / "schema.sql").write_text(sql_schema(schema))
    (output / "plurel_schema.sql").write_text(plurel_schema(schema))
    induction = {"candidates": candidates, "chosen": decision}
    write_json(output / "schema_decision.json", induction)
    print(
        "        primary keys "
        + ", ".join(f"{t}.{c}" for t, c in decision["primary_keys"].items())
        + f"; {len(decision['foreign_keys'])} of {len(candidates[1])} foreign key candidates kept,"
        " all confirmed against the data"
    )

    print("     Generate the stringless landscape with PluRel")
    fitted = fit(anonymized)
    if min(fitted["sizes"].values()) < 4:
        raise ValueError("PluRel's clustered sampler needs at least 4 rows in every table")
    stringless = synthesize(
        schema, output / "plurel_schema.sql", args.seed, fitted, output / "plurel.log"
    )
    validate(stringless)
    write_tables(output / "stringless", stringless)
    print(
        "        "
        + ", ".join(f"{k} {len(v)}" for k, v in stringless.items())
        + "; keys, categories, numbers and dates generated, text columns empty"
    )

    print("3/3  Fill the descriptive text with the LLM")
    synthetic = infill(stringless)
    validate(synthetic)
    write_tables(output / "synthetic", synthetic)
    sqlite_export(output / "synthetic" / "landscape.sqlite", synthetic, schema)
    print(
        "        "
        + ", ".join(f"{len(TEXT[k]) * len(v)} texts in {k}" for k, v in synthetic.items())
        + "; foreign keys checked in CSV and SQLite"
    )

    stages = {
        "source": source,
        "anonymized": anonymized,
        "stringless": stringless,
        "synthetic": synthetic,
    }
    summary = write_report(output, stages, args.seed, induction, fitted)
    write_json(output / "summary.json", {**describe(fitted), **summary})
    print(
        f"\nAnonymization preserved every edge: "
        f"{summary['anonymization_preserves_every_edge']}\n"
        f"Synthetic tables match the source row counts: "
        f"{summary['synthetic_matches_source_row_counts']}\n"
        f"Synthetic graph is not the source graph relabelled: "
        f"{summary['synthetic_graph_differs_from_source']}\n"
        f"Synthetic tables reproduce every category share: "
        f"{summary['synthetic_reproduces_every_category_share']}\n"
        f"Done in {time.monotonic() - started:.1f}s. Open {output / 'report.html'}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=HERE / "data")
    parser.add_argument("--out", type=Path, default=HERE / "output")
    parser.add_argument("--seed", type=int, default=42)
    try:
        run(parser.parse_args())
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

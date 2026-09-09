"""Reading the input tables, key discovery, schema construction and validation."""

import csv
import json
import math
import re
import sqlite3
from pathlib import Path

KEYS = {
    "applications": "application_id",
    "components": "component_id",
    "interfaces": "interface_id",
}
RELATIONS = [
    ("components", "application_id", "applications"),
    ("interfaces", "source_component_id", "components"),
    ("interfaces", "target_component_id", "components"),
]
# The columns step 3 asks PluRel to generate a feature for, and what that
# feature is turned back into. A category is drawn from the values observed in
# the anonymized table; a number and a date are placed in the observed range;
# `owners` reproduces how many owners there are and how large each portfolio is.
GENERATED = {
    ("applications", "owner"): "owners",
    ("applications", "criticality"): "category",
    ("applications", "annual_cost_eur"): "number",
    ("applications", "go_live_date"): "date",
    ("components", "component_type"): "category",
    ("components", "network_zone"): "category",
    ("interfaces", "protocol"): "category",
}
# Columns that follow another column in the anonymized data rather than being
# generated on their own: a protocol is served on its registered port, and the
# masked address says which network a zone uses. Both regularities survive
# step 1, so step 3 reproduces them and the synthetic rows stay consistent.
DERIVED = {
    ("interfaces", "port"): "protocol",
    ("components", "ip_address"): "network_zone",
}
FEATURES = {table: [column for (owner, column) in GENERATED if owner == table] for table in KEYS}
# The free-text columns. PluRel generates no text, so these leave step 3 empty
# and the LLM stage in `llm.py` fills them.
TEXT = {
    "applications": ["name", "description"],
    "components": ["name", "description"],
    "interfaces": ["description"],
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_tables(directory, tables):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        if not rows:
            raise ValueError(f"Cannot export empty table {name}")
        with (directory / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return [
            {key: value.strip() if value else None for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def number(value):
    """A CSV holds text, so read an integer-looking value back as an integer.

    Identifiers, ports and costs are numbers in the induced schema; a date such
    as `2019-03-14` is not an integer and stays text.
    """
    return int(value) if value is not None and re.fullmatch(r"-?\d+", value) else value


def read_tables(directory):
    """The input the demo starts from: one CSV per table."""
    directory = Path(directory)
    return {
        table: [
            {column: number(value) for column, value in row.items()}
            for row in read_csv(directory / f"{table}.csv")
        ]
        for table in KEYS
    }


def unique_key(rows, column):
    values = [row[column] for row in rows]
    return None not in values and len(set(values)) == len(values)


def key_candidates(tables):
    """The heuristics tool: every column that could be a key, by uniqueness alone.

    A primary key candidate is present and unique in every row. A foreign key
    candidate is a column whose values all appear in another table's primary key
    candidate. Both tests are syntactic, so they cannot tell an identifier from a
    column that happens to be unique; that is what the LLM judge decides.
    """
    primary = {
        table: [column for column in rows[0] if unique_key(rows, column)]
        for table, rows in tables.items()
    }
    foreign = []
    for table, rows in tables.items():
        for column in rows[0]:
            values = {row[column] for row in rows if row[column] is not None}
            for parent, parent_rows in tables.items():
                if parent == table or not values:
                    continue
                if any(values <= {row[key] for row in parent_rows} for key in primary[parent]):
                    foreign.append((table, column, parent))
    return primary, foreign


def infer_schema(tables, decision):
    """Build the relational schema the judge chose, after checking it against the data.

    Nothing downstream sees an unverified judgement: a chosen primary key must be
    present and unique, and a chosen foreign key's values must all appear in the
    parent's primary key. A decision that does not hold stops the run.
    """
    schema = {}
    for table, rows in tables.items():
        primary = decision["primary_keys"].get(table)
        if primary not in rows[0]:
            raise ValueError(f"{table}: {primary!r} is not a column of this table")
        if not unique_key(rows, primary):
            raise ValueError(f"{table}.{primary} is not a primary key")
        columns = {}
        for column in rows[0]:
            present = [row[column] for row in rows if row[column] not in (None, "")]
            columns[column] = (
                "REAL" if present and all(isinstance(v, (int, float)) for v in present) else "TEXT"
            )
        schema[table] = {"primary_key": primary, "columns": columns, "foreign_keys": {}}
    for table, column, parent in decision["foreign_keys"]:
        if table not in schema or parent not in schema or column not in tables[table][0]:
            raise ValueError(f"Cannot confirm {table}.{column} -> {parent}")
        children = {row[column] for row in tables[table] if row[column] is not None}
        parents = {row[schema[parent]["primary_key"]] for row in tables[parent]}
        if not children or not children <= parents:
            raise ValueError(f"Cannot confirm {table}.{column} -> {parent}")
        schema[table]["foreign_keys"][column] = parent
    return schema


def sql_schema(schema):
    """The anonymized relational schema, as inferred."""
    statements = []
    for table, info in schema.items():
        definitions = []
        for column, kind in info["columns"].items():
            primary = " PRIMARY KEY" if column == info["primary_key"] else ""
            definitions.append(f'  "{column}" {kind}{primary}')
        for column, parent in info["foreign_keys"].items():
            definitions.append(
                f'  FOREIGN KEY ("{column}") REFERENCES "{parent}" ("{schema[parent]["primary_key"]}")'
            )
        statements.append(f'CREATE TABLE "{table}" (\n' + ",\n".join(definitions) + "\n);")
    return "\n\n".join(statements) + "\n"


def plurel_schema(schema):
    """Keys, foreign keys and one numeric slot per generated feature.

    PluRel's SQL reader takes numeric columns, so the descriptive text columns
    are simply left out rather than projected onto meaningless numbers. Nothing
    here carries a value from the data: it is the shape of the schema only.
    """
    statements = []
    for table, info in schema.items():
        definitions = [f'  "{info["primary_key"]}" INTEGER PRIMARY KEY']
        definitions += [f'  "{column}" INTEGER' for column in info["foreign_keys"]]
        definitions += [f'  "{column}" REAL' for column in FEATURES[table]]
        definitions += [
            f'  FOREIGN KEY ("{column}") REFERENCES "{parent}" ("{schema[parent]["primary_key"]}")'
            for column, parent in info["foreign_keys"].items()
        ]
        statements.append(f'CREATE TABLE "{table}" (\n' + ",\n".join(definitions) + "\n);")
    return "\n\n".join(statements) + "\n"


def validate(tables):
    """Fail closed on broken identifiers, foreign keys or non-finite numbers."""
    for table, rows in tables.items():
        ids = [r[KEYS[table]] for r in rows]
        if not ids or None in ids or len(set(ids)) != len(ids):
            raise ValueError(f"Invalid primary keys in {table}")
        if any(isinstance(v, float) and not math.isfinite(v) for r in rows for v in r.values()):
            raise ValueError(f"Non-finite values in {table}")
    for table, column, parent in RELATIONS:
        parents = {r[KEYS[parent]] for r in tables[parent]}
        if any(r[column] is not None and r[column] not in parents for r in tables[table]):
            raise ValueError(f"Dangling foreign key: {table}.{column}")


def sqlite_export(path, tables, schema):
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(sql_schema(schema))
        for table, rows in tables.items():
            columns = list(schema[table]["columns"])
            names = ", ".join(f'"{c}"' for c in columns)
            placeholders = ", ".join("?" for _ in columns)
            connection.executemany(
                f'INSERT INTO "{table}" ({names}) VALUES ({placeholders})',
                [[row[c] for c in columns] for row in rows],
            )
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("SQLite foreign key validation failed")

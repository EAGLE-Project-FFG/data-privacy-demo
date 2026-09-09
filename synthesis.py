"""Generate a fresh landscape with PluRel, sized and weighted like the anonymized one.

Step 3 produces a landscape that is structurally like the source without
reproducing it. What crosses from step 1 into this module is a deliberate, small
summary, measured by `fit` on the *anonymized* tables only: the number of rows
per table, the share of each category, the range of each number and date, and
the shape of the owner portfolios. No row, no key, no alias and no edge crosses
over, and the graph is not measured at all.

PluRel builds the graph and one numeric feature per column from that summary and
its own priors. Its numbers mean nothing on their own, so none of them reaches
the output: each is mapped onto the fitted summary here, and the descriptive
text is written afterwards by `llm.infill`.
"""

import contextlib
import datetime
import json
from collections import Counter

from pipeline import DERIVED, GENERATED, KEYS, TEXT

# Public ID namespace for the generated rows. There is no mapping back to a
# source row, and none to the identifiers of the anonymized tables.
DOMAINS = {
    "application_id": "syn-app",
    "component_id": "syn-cmp",
    "source_component_id": "syn-cmp",
    "target_component_id": "syn-cmp",
    "interface_id": "syn-ifc",
}


def shares(rows, column):
    """Category shares of one anonymized column, most common first.

    Missing values are a category of their own, so the synthetic table
    reproduces how often the column is empty instead of filling it in. Ordering
    by count and then by label keeps a run reproducible.
    """
    counts = Counter(row[column] if row[column] not in ("", None) else None for row in rows)
    return sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))


def teams(rows, column="owner"):
    """The owner shape: how many owners there are, and how many rows each holds.

    The aliases themselves do not cross over — a synthetic `Team 01` says
    nothing about any pseudonym, and the two tables have no owner in common.
    Only the number of distinct owners and the size of each one's portfolio is
    reproduced, because that an owner holds several applications is structure.
    """
    counts = Counter(row[column] for row in rows)
    portfolios = sorted((count for owner, count in counts.items() if owner), reverse=True)
    unowned = [(None, counts[None])] if counts.get(None) else []
    return [(f"Team {n + 1:02d}", count) for n, count in enumerate(portfolios)] + unowned


def span(rows, column, kind):
    """The lowest and highest value of a numeric or date column, and nothing else."""
    values = [row[column] for row in rows if row[column] not in (None, "")]
    if kind == "date":
        values = [datetime.date.fromisoformat(str(v)).toordinal() for v in values]
    return [min(values), max(values)]


def follows(rows, column, leader):
    """Which value of `column` goes with each value of `leader`, most common first.

    A leading value observed without a follower keeps an empty follower, and
    ties break on the follower itself so a run is reproducible.
    """
    observed = {}
    for row in rows:
        if row[leader] in (None, ""):
            continue
        counts = observed.setdefault(row[leader], Counter())
        if row[column] not in (None, ""):
            counts[row[column]] += 1
    return {
        value: min(counts.items(), key=lambda item: (-item[1], str(item[0])))[0] if counts else None
        for value, counts in observed.items()
    }


def fit(anonymized):
    """Everything step 3 is allowed to know about step 1, and nothing else."""
    summary = {}
    for (table, column), kind in GENERATED.items():
        rows = anonymized[table]
        if kind == "owners":
            summary[(table, column)] = ("owners", teams(rows, column))
        elif kind == "category":
            summary[(table, column)] = ("category", shares(rows, column))
        else:
            summary[(table, column)] = (kind, span(rows, column, kind))
    return {
        "sizes": {table: len(rows) for table, rows in anonymized.items()},
        "columns": summary,
        "derived": {
            (table, column): follows(anonymized[table], column, leader)
            for (table, column), leader in DERIVED.items()
        },
    }


def describe(fitted):
    """The fitted summary as plain JSON, so a run records what crossed over."""
    described = {}
    for (table, column), (kind, value) in fitted["columns"].items():
        if kind in ("category", "owners"):
            described[f"{table}.{column}"] = {
                (label or "(empty)"): weight for label, weight in value
            }
        elif kind == "date":
            described[f"{table}.{column}"] = [
                datetime.date.fromordinal(v).isoformat() for v in value
            ]
        else:
            described[f"{table}.{column}"] = value
    return {
        "sizes": fitted["sizes"],
        "fitted": described,
        "derived": {
            f"{table}.{column} follows {DERIVED[(table, column)]}": value
            for (table, column), value in fitted["derived"].items()
        },
    }


def rank_map(rows, column, vocabulary, key):
    """Order rows by a generated feature, then cut the order into fitted shares.

    Using the rank rather than the value keeps whatever correlation PluRel put
    between this feature and the rest of the row, while the labels and their
    shares come from `vocabulary`. Ties break on the primary key, so a run is
    reproducible for a fixed seed.
    """
    total = sum(weight for _, weight in vocabulary)
    labels, running = [], 0
    for label, weight in vocabulary:
        running += weight
        labels += [label] * (round(len(rows) * running / total) - len(labels))
    for row, label in zip(sorted(rows, key=lambda r: (r[column], r[key])), labels):
        row[column] = label


def scale_map(rows, column, low, high, kind):
    """Place a generated feature in the fitted range, keeping its own shape.

    The feature is rescaled rather than ranked, so how the generated values
    bunch up inside the range is PluRel's and only the two endpoints come from
    the measurement.
    """
    values = [row[column] for row in rows]
    least, most = min(values), max(values)
    for row in rows:
        position = (row[column] - least) / (most - least) if most > least else 0.5
        value = round(low + position * (high - low))
        row[column] = datetime.date.fromordinal(value).isoformat() if kind == "date" else value


def generate(schema_path, seed, sizes, log_path):
    """Call PluRel at the fitted table sizes, then repair what 1.1.0 gets wrong."""
    import numpy as np
    import torch
    from plurel import Choices, Config, DatabaseParams, SCMParams, SyntheticDataset
    from plurel.bipartite import sample_bipartite_assignments

    class Sized(SyntheticDataset):
        """One row count per table.

        PluRel samples a single count for every entity table and a single count
        for every activity table, so `applications` and `components` could not
        otherwise differ in size. The schema graph it builds carries the count
        per table, so setting it there is enough, and the sampling above still
        runs and consumes the same random numbers as before.
        """

        def configure_table_relationships(self, **kwargs):
            relationships = super().configure_table_relationships(**kwargs)
            for node in relationships.nodes:
                relationships.nodes[node]["num_rows"] = sizes[relationships.nodes[node]["name"]]
            return relationships

    torch.set_num_threads(1)
    config = Config(
        schema_file=str(schema_path),
        database_params=DatabaseParams(
            num_rows_entity_table_choices=Choices("set", [sizes["components"]]),
            num_rows_activity_table_choices=Choices("set", [sizes["interfaces"]]),
            column_nan_perc_choices=Choices("set", [0.0]),
            col_transform_choices=Choices("set", ["identity"]),
        ),
        scm_params=SCMParams(
            scm_layout_choices=Choices("set", ["ErdosRenyi"]),
            activation_choices=Choices("set", [torch.tanh]),
            bi_hsbm_levels_choices=Choices("set", [2]),
            bi_hsbm_clusters_per_level_choices=Choices("set", [2]),
        ),
    )
    with open(log_path, "w") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            database = Sized(seed=seed, config=config).make_db()
    frames = {name: table.df.copy() for name, table in database.table_dict.items()}

    # PluRel 1.1.0 keys foreign key assignments by parent table, not by column,
    # so both interface endpoints come back identical. Resample the target role
    # with PluRel's own bipartite sampler; this is the one adapter in the demo.
    np.random.seed(seed + 1)
    frames["interfaces"]["target_component_id"] = sample_bipartite_assignments(
        size_a=len(frames["components"]),
        size_b=len(frames["interfaces"]),
        hierarchy_a=[2, 2],
        hierarchy_b=[2, 2],
    )
    # An interface connects two distinct components in this demo's domain.
    links = frames["interfaces"]
    loops = links["source_component_id"] == links["target_component_id"]
    sources = links.loc[loops, "source_component_id"].to_numpy()
    others = np.random.randint(0, len(frames["components"]) - 1, size=len(sources))
    links.loc[loops, "target_component_id"] = others + (others >= sources)

    tables = {}
    for name, frame in frames.items():
        # PluRel adds a timestamp to activity tables; it is not in this schema.
        frame = frame.drop(columns=["date"], errors="ignore")
        rows = json.loads(frame.to_json(orient="records"))
        for row in rows:
            for column, prefix in DOMAINS.items():
                if column in row:
                    row[column] = f"{prefix}-{int(row[column]) + 1:03d}"
        tables[name] = rows
    return tables


def realize(tables, fitted):
    """Turn the generated features into values, and leave the text columns empty."""
    result = {table: [dict(row) for row in rows] for table, rows in tables.items()}
    for (table, column), (kind, value) in fitted["columns"].items():
        if any(column not in row for row in result[table]):
            raise ValueError(f"PluRel did not generate {table}.{column}")
        if kind in ("category", "owners"):
            rank_map(result[table], column, value, KEYS[table])
        else:
            scale_map(result[table], column, value[0], value[1], kind)

    for (table, column), lookup in fitted["derived"].items():
        for row in result[table]:
            row[column] = lookup.get(row[DERIVED[(table, column)]])
    # The text columns stay empty here: this is the stringless data of Figure 1,
    # and `llm.infill` is what fills them.
    for table, columns in TEXT.items():
        for row in result[table]:
            for column in columns:
                row[column] = None
    return result


def order_columns(tables, schema):
    """Match the anonymized column order so the report compares stages in place."""
    return {
        table: [{column: row[column] for column in schema[table]["columns"]} for row in rows]
        for table, rows in tables.items()
    }


def synthesize(schema, schema_path, seed, fitted, log_path):
    """The stringless landscape: structure at the fitted sizes, values, no text."""
    generated = generate(schema_path, seed, fitted["sizes"], log_path)
    if set(generated) != set(KEYS):
        raise ValueError(f"PluRel returned unexpected tables: {sorted(generated)}")
    return order_columns(realize(generated, fitted), schema)

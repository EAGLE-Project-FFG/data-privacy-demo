import csv
import datetime
import json
import os
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from llm import infill, judge
from pipeline import (
    DERIVED,
    GENERATED,
    KEYS,
    RELATIONS,
    TEXT,
    infer_schema,
    key_candidates,
    plurel_schema,
    read_tables,
    validate,
)
from privacy import (
    OPERATIONS,
    PLAN,
    SPREAD,
    WORD_LISTS,
    Pseudonyms,
    anonymize,
    distort,
    examples,
    mask,
    shift_date,
)
from report import graph_svg, metrics, same_shares, topology
from synthesis import describe, fit, generate, realize

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"
SIZES = {"applications": 8, "components": 16, "interfaces": 18}
# The judgement the LLM is expected to reach, used wherever a test needs a
# schema without calling the model. `test_the_judge_chooses_the_identifiers`
# checks the model against it.
DECISION = {"primary_keys": KEYS, "foreign_keys": [list(item) for item in RELATIONS]}
LIVE = os.environ.get("RUN_LIVE_TESTS") == "1"
HAS_KEY = (HERE / ".env").exists() or "OPENAI_API_KEY" in os.environ
live = pytest.mark.skipif(
    not (LIVE and HAS_KEY),
    reason="set RUN_LIVE_TESTS=1 and provide OPENAI_API_KEY",
)


@pytest.fixture
def source():
    return read_tables(DATA)


@pytest.fixture
def anonymized(source):
    return anonymize(source)[0]


def test_the_input_tables_are_complete_and_consistent(source):
    validate(source)
    assert {table: len(rows) for table, rows in source.items()} == SIZES
    # Every column of every table is either generated, free text, a key, or
    # follows another column. Nothing in the schema is unaccounted for.
    accounted = set(KEYS.items()) | set(GENERATED) | set(DERIVED)
    accounted |= {(table, column) for table, columns in TEXT.items() for column in columns}
    accounted |= {(table, column) for table, column, _ in RELATIONS}
    for table, rows in source.items():
        unexplained = {(table, c) for c in rows[0]} - accounted - {(table, "cost_center")}
        assert not unexplained


def test_every_operation_in_the_paper_has_a_column_to_act_on():
    """The demo is meant to show all seven operations, not a subset of them."""
    used = {operation for columns in PLAN.values() for operation in columns.values()}
    assert used == set(OPERATIONS) - {"retain"}
    assert len(used) == 7


def test_pseudonyms_come_from_the_word_lists_and_are_reversible():
    aliases = Pseudonyms()
    people = [f"Person {n}" for n in range(20)]
    replaced = [aliases.replace(person, "people") for person in people]
    # A hash picks the entry and probing keeps the mapping injective, which is
    # what makes it reversible. nothing is invented outside the word list.
    assert len(set(replaced)) == len(people)
    assert set(replaced) <= set(WORD_LISTS["people"])
    assert all(aliases.restore(alias, "people") == p for p, alias in zip(people, replaced))
    # Stable within a run, and independent per domain.
    assert aliases.replace(people[0], "people") == replaced[0]
    assert aliases.replace("Anything", "components") in WORD_LISTS["components"]
    assert aliases.replace(None, "people") is None
    with pytest.raises(ValueError, match="fewer entries"):
        for n in range(len(WORD_LISTS["people"]) + 1):
            aliases.replace(f"Extra {n}", "people")


def test_masking_preserves_length_and_keeps_only_the_leading_octet():
    assert mask("172.16.4.11") == "172.**.*.**"
    assert len(mask("192.168.16.31")) == len("192.168.16.31")
    assert all(not character.isdigit() for character in mask("10.20.8.21")[3:])


def test_distortion_stays_inside_the_declared_range_and_is_deterministic():
    for value in (95000, 130000, 480000, 520000):
        assert distort(value) == distort(value)
        assert abs(distort(value) - value) <= SPREAD * value
    # Different values move by different amounts, so the column is not shifted
    # by one constant that a single known pair would reveal.
    assert len({distort(v) - v for v in (95000, 130000, 480000, 520000)}) == 4


def test_shift_date_yields_a_valid_date_that_keeps_its_year():
    for value in ("2019-03-14", "2016-05-30", "2015-08-11", "2020-02-29"):
        shifted = datetime.date.fromisoformat(shift_date(value))
        assert shifted.year == datetime.date.fromisoformat(value).year
        assert shift_date(value) == shift_date(value)
    # A day that would not exist in the month it lands in is folded into it.
    assert datetime.date.fromisoformat(shift_date("2021-01-31"))


def test_anonymization_replaces_the_values_and_leaves_the_graph_untouched(source, anonymized):
    validate(anonymized)
    # The removed column is gone everywhere; every other column survives.
    assert "cost_center" not in anonymized["applications"][0]
    assert set(anonymized["components"][0]) == set(source["components"][0])
    # No name, owner, cost, date, address or description passes through.
    for table, columns in PLAN.items():
        for column, operation in columns.items():
            if operation == "remove_column":
                continue
            before = {r[column] for r in source[table] if r[column] not in (None, "")}
            after = {r[column] for r in anonymized[table] if r[column] not in (None, "")}
            assert not before & after, (table, column)
    # Equality is preserved, which is what pseudonymization is for: the three
    # applications one person owns still share one owner.
    owners = Counter(r["owner"] for r in anonymized["applications"])
    assert sorted(owners.values(), reverse=True) == [3, 2, 2, 1]
    # And this is the point of the demo: the dependency graph is untouched.
    assert topology(source) == topology(anonymized)
    assert metrics(source) == metrics(anonymized)


def test_the_operations_table_shows_every_operation_acting_on_real_input(source, anonymized):
    rows = examples(source, anonymized)
    assert [row["Operation"] for row in rows] == [name for name, _, _ in OPERATIONS.values()]
    for row in rows:
        assert row["Columns"] and "→" in row["Example"]
    removal = next(row for row in rows if row["Operation"] == "Remove Column")
    assert removal["Example"].endswith("(column removed)")


def test_graph_shows_applications_components_and_interfaces(source):
    svg = graph_svg(source, "test")
    assert svg.count('class="application-node graph-detail"') == SIZES["applications"]
    assert svg.count('class="component-node graph-detail"') == SIZES["components"]
    assert svg.count('class="ownership-edge"') == SIZES["components"]
    assert svg.count('class="interface-hit graph-detail"') == SIZES["interfaces"]
    assert svg.count('marker-end="url(#arrow-test)"') == SIZES["interfaces"]
    assert "Payments Platform" in svg and "Protocol: HTTPS" in svg


def test_heuristics_leave_a_real_choice_to_the_judge(anonymized):
    primary, foreign = key_candidates(anonymized)
    # Uniqueness cannot separate the identifier from a column that happens to be
    # unique, and applications has three of those. That ambiguity is the judge's
    # job, and so is rejecting the references that only hold by coincidence.
    assert primary == {
        "applications": ["application_id", "name", "annual_cost_eur", "go_live_date"],
        "components": ["component_id", "name"],
        "interfaces": ["interface_id"],
    }
    assert set(RELATIONS) < set(foreign) and len(foreign) > len(RELATIONS)


def test_judge_is_asked_about_the_candidates_and_its_answer_is_verified(source, anonymized):
    candidates = key_candidates(anonymized)
    asked = {}

    def fake_ask(prompt):
        asked["prompt"] = prompt
        return DECISION

    assert judge(anonymized, candidates, ask=fake_ask) == DECISION
    assert "component_id" in asked["prompt"] and "name" in asked["prompt"]
    # The prompt carries anonymized values only.
    assert not any(row["name"] in asked["prompt"] for row in source["components"])
    assert "CC-4471" not in asked["prompt"]

    # A judgement that does not hold against the data stops the run, so a wrong
    # answer cannot reach PluRel: neither a column that is not unique, ...
    with pytest.raises(ValueError, match="not a primary key"):
        infer_schema(
            anonymized, {**DECISION, "primary_keys": {**KEYS, "components": "network_zone"}}
        )
    # ... nor a reference whose values are not in the parent key.
    with pytest.raises(ValueError, match="Cannot confirm"):
        infer_schema(
            anonymized, {**DECISION, "foreign_keys": [["interfaces", "protocol", "components"]]}
        )


@live
def test_the_judge_chooses_the_identifiers(anonymized):
    decision = judge(anonymized, key_candidates(anonymized))
    assert decision["primary_keys"] == KEYS
    assert sorted(decision["foreign_keys"]) == sorted(DECISION["foreign_keys"])


def test_fit_measures_sizes_shares_and_ranges_and_nothing_else(anonymized):
    fitted = fit(anonymized)
    assert fitted["sizes"] == SIZES
    assert dict(fitted["columns"][("components", "component_type")][1]) == {
        "Application": 7,
        "Database": 5,
        "Middleware": 3,
        "Fileshare": 1,
    }
    # A number and a date cross over as two endpoints, never as values.
    kind, (low, high) = fitted["columns"][("applications", "annual_cost_eur")]
    costs = [row["annual_cost_eur"] for row in anonymized["applications"]]
    assert (kind, low, high) == ("number", min(costs), max(costs))
    assert fitted["columns"][("applications", "go_live_date")][0] == "date"
    # The owner labels are invented here: only the number of owners and the size
    # of each portfolio crosses over, never a pseudonym.
    portfolios = dict(fitted["columns"][("applications", "owner")][1])
    assert portfolios == {"Team 01": 3, "Team 02": 2, "Team 03": 2, "Team 04": 1}
    assert not set(portfolios) & {r["owner"] for r in anonymized["applications"]}
    # Port follows protocol and the masked network follows the zone.
    assert fitted["derived"][("interfaces", "port")] == {
        "HTTPS": 443,
        "SQL": 1433,
        "SFTP": 22,
        "SMB": 445,
    }
    assert fitted["derived"][("components", "ip_address")] == {
        "DMZ": "172.**.*.**",
        "Internal": "10.**.*.**",
        "Restricted": "192.***.**.**",
    }
    assert set(describe(fitted)) == {"sizes", "fitted", "derived"}


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    output = tmp_path_factory.mktemp("plurel")
    anonymized = anonymize(read_tables(DATA))[0]
    schema = infer_schema(anonymized, DECISION)
    path = output / "plurel_schema.sql"
    path.write_text(plurel_schema(schema))
    fitted = fit(anonymized)
    return generate(path, 42, fitted["sizes"], output / "plurel.log"), path, output, fitted


def test_plurel_is_reproducible_and_has_no_self_links(generated):
    first, path, output, fitted = generated
    assert first == generate(path, 42, fitted["sizes"], output / "log2")
    validate(first)
    assert all(r["source_component_id"] != r["target_component_id"] for r in first["interfaces"])
    other = generate(path, 7, fitted["sizes"], output / "log3")
    validate(other)
    assert topology(first) != topology(other)


def test_plurel_generates_one_row_count_per_table(generated):
    first, _, _, fitted = generated
    # PluRel samples a single row count for all entity tables, so without the
    # per-table override applications and components could not differ in size.
    assert {table: len(rows) for table, rows in first.items()} == fitted["sizes"] == SIZES


def test_realize_reproduces_every_fitted_share_and_leaves_the_text_empty(generated):
    first, _, _, fitted = generated
    anonymized = anonymize(read_tables(DATA))[0]
    tables = realize(first, fitted)
    assert topology(tables) == topology(first)
    for (table, column), (kind, value) in fitted["columns"].items():
        counts = Counter(row[column] for row in tables[table])
        if kind in ("category", "owners"):
            assert sorted(counts.values(), reverse=True) == sorted(
                (weight for _, weight in value), reverse=True
            )
            assert set(counts) == {label for label, _ in value}
            if kind == "category":  # No label was invented; owners are.
                assert set(counts) <= {row[column] for row in anonymized[table]}
        else:  # A number and a date land inside the fitted range.
            low, high = value
            observed = [row[column] for row in tables[table]]
            if kind == "date":
                observed = [datetime.date.fromisoformat(v).toordinal() for v in observed]
            assert low <= min(observed) and max(observed) <= high
    assert same_shares(
        anonymized, tables, [t for t, (k, _) in fitted["columns"].items() if k == "category"]
    )
    # No generated feature reaches the output as a bare number, the derived
    # columns agree with the columns they follow, and the text is still empty.
    for table, rows in tables.items():
        for row in rows:
            assert all(not isinstance(v, float) for v in row.values()), row
            assert all(row[column] is None for column in TEXT[table])
    for (table, column), leader in DERIVED.items():
        lookup = fitted["derived"][(table, column)]
        assert all(row[column] == lookup.get(row[leader]) for row in tables[table])


def test_infill_fills_every_text_column_from_the_generated_row(generated):
    first, _, _, fitted = generated
    stringless = realize(first, fitted)
    prompts = []

    def fake_ask(prompt):
        prompts.append(prompt)
        rows = json.loads(prompt.split("Rows:\n", 1)[1].rsplit("\n\nReply", 1)[0])
        key = next(column for column in KEYS.values() if column in rows[0])
        return {row[key]: {"name": "A name", "description": "A description."} for row in rows}

    tables = infill(stringless, ask=fake_ask)
    for table, columns in TEXT.items():
        for row in tables[table]:
            assert all(row[column] for column in columns)
    # A component prompt carries the name its application was just given, which
    # is what lets a description name the service it belongs to.
    assert "A name" in prompts[1]

    # An empty or missing text stops the run rather than shipping a blank cell.
    with pytest.raises(ValueError, match="No name generated"):
        infill(stringless, ask=lambda prompt: {})


def run_cli(output, *extra):
    command = [sys.executable, str(HERE / "demo.py"), "--out", str(output), *extra]
    return subprocess.run(command, capture_output=True, text=True, timeout=600, cwd=HERE)


@pytest.fixture(scope="module")
def cli_output(tmp_path_factory):
    """One full run of the demo, shared by the tests below. It calls the model."""
    output = tmp_path_factory.mktemp("run") / "output"
    result = run_cli(output)
    assert result.returncode == 0, result.stdout + result.stderr
    return output


def read_stage(output, stage):
    return {name: list(csv.DictReader((output / stage / f"{name}.csv").open())) for name in KEYS}


@live
def test_cli_writes_every_stage_and_the_report(cli_output):
    for stage in ("source", "anonymized", "stringless", "synthetic"):
        assert {t: len(rows) for t, rows in read_stage(cli_output, stage).items()} == SIZES
    assert (cli_output / "report.html").exists()
    with sqlite3.connect(cli_output / "synthetic" / "landscape.sqlite") as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("SELECT count(*) FROM interfaces").fetchone()[0] == 18
    summary = json.loads((cli_output / "summary.json").read_text())
    assert summary["anonymization_preserves_every_edge"]
    assert summary["synthetic_matches_source_row_counts"]
    assert summary["synthetic_graph_differs_from_source"]
    assert summary["synthetic_reproduces_every_category_share"]


@live
def test_no_source_value_reaches_the_synthetic_tables(cli_output):
    source = read_stage(cli_output, "source")
    synthetic = read_stage(cli_output, "synthetic")
    page = (cli_output / "report.html").read_text()
    for table, rows in source.items():
        for column in ("name", "description"):
            if column not in rows[0]:
                continue
            values = {r[column] for r in rows if r[column]}
            assert not values & {r[column] for r in synthetic[table] if r[column]}
    # The report shows the source tab, so it does carry the input values; the
    # synthetic CSVs must not, and the identifiers live in their own namespace.
    assert "Payments Platform" in page
    assert all(r["application_id"].startswith("syn-app-") for r in synthetic["applications"])


@live
def test_the_llm_stage_is_what_fills_the_text_columns(cli_output):
    stringless = read_stage(cli_output, "stringless")
    synthetic = read_stage(cli_output, "synthetic")
    for table, columns in TEXT.items():
        for before, after in zip(stringless[table], synthetic[table]):
            assert all(not before[column] for column in columns)
            assert all(after[column].strip() for column in columns)
        # Everything the model was not asked to write is unchanged.
        for before, after in zip(stringless[table], synthetic[table]):
            assert {k: v for k, v in before.items() if k not in columns} == {
                k: v for k, v in after.items() if k not in columns
            }


def test_cli_refuses_a_directory_it_does_not_own(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "notes.txt").write_text("someone else's file")
    result = run_cli(output)
    assert result.returncode == 1 and "did not write" in result.stderr
    assert f"Code:   {HERE}" in result.stdout
    assert "8 applications, 16 components, 18 interfaces" in result.stdout
    assert (output / "notes.txt").exists()


def test_prepare_reuses_its_own_output_directory(tmp_path):
    import demo

    output = tmp_path / "output"
    output.mkdir()
    (output / "report.html").write_text("stale")
    (output / "synthetic").mkdir()
    demo.prepare(output)
    assert output.exists() and not list(output.iterdir())

"""One standalone HTML page: the four stages, the three graphs, the field operations."""

from collections import Counter
from html import escape

from llm import MODEL
from privacy import examples
from synthesis import shares

STAGES = [
    ("source", "1. Source", "The input tables, before anonymization."),
    (
        "anonymized",
        "2. Anonymized",
        "The same rows in the same order, after the field operations below. Switching between "
        "this tab and the previous one compares the tables row by row.",
    ),
    (
        "stringless",
        "3. Stringless",
        "What PluRel returns: keys, foreign keys, categories, numbers and dates, at the sizes "
        "measured on the anonymized tables. It generates no text.",
    ),
    (
        "synthetic",
        "4. Synthetic",
        f"The same rows after {MODEL} wrote the names and descriptions. No row here corresponds "
        "to a row of the source.",
    ),
]
# The graph is the same in the last two stages, so it is drawn once.
GRAPHS = [("source", "Source"), ("anonymized", "Anonymized"), ("synthetic", "Synthetic")]
MEASURES = [
    ("applications", "Applications"),
    ("components", "Components"),
    ("interfaces", "Interfaces"),
    ("edges", "Distinct directed edges"),
    ("mean_degree", "Mean component degree"),
    ("max_degree", "Highest component degree"),
    ("islands", "Disconnected groups"),
]


def graph(tables):
    import networkx as nx

    result = nx.DiGraph()
    result.add_nodes_from(row["component_id"] for row in tables["components"])
    result.add_edges_from(
        (r["source_component_id"], r["target_component_id"]) for r in tables["interfaces"]
    )
    return result


def topology(tables):
    """Edges by row position, which compares source and anonymized rows directly."""
    positions = {r["component_id"]: i for i, r in enumerate(tables["components"])}
    return Counter(
        (positions[r["source_component_id"]], positions[r["target_component_id"]])
        for r in tables["interfaces"]
    )


def metrics(tables):
    import networkx as nx

    g = graph(tables)
    degrees = dict(g.degree())
    return {
        "applications": len(tables["applications"]),
        "components": len(g),
        "interfaces": len(tables["interfaces"]),
        "edges": len(g.edges),
        "mean_degree": round(sum(degrees.values()) / len(degrees), 2) if degrees else 0,
        "max_degree": max(degrees.values(), default=0),
        "islands": nx.number_weakly_connected_components(g),
    }


def graph_svg(tables):
    import networkx as nx

    palette = [
        "#247ba0",
        "#70c1b3",
        "#f3a712",
        "#d95d39",
        "#8f6bb3",
        "#5f8f3d",
        "#d76a9a",
        "#68737d",
    ]
    application_index = {
        row["application_id"]: index for index, row in enumerate(tables["applications"])
    }
    component_application = {
        row["component_id"]: application_index[row["application_id"]]
        for row in tables["components"]
    }
    # Nodes are relabelled by position, so the source and anonymized graphs get
    # the same layout and any difference on the page is a real difference.
    original = graph(tables)
    g = nx.relabel_nodes(original, {node: i for i, node in enumerate(original)})
    # Kamada-Kawai, not spring: on a graph this small it separates the hubs and
    # leaves few edges crossing, so the shape of the landscape is readable. It
    # takes no seed and is deterministic, and the source and anonymized graphs
    # are relabelled identically, so those two always come out congruent.
    undirected = g.to_undirected()
    positions = nx.kamada_kawai_layout(undirected)
    degrees = dict(undirected.degree())
    xs = [float(p[0]) for p in positions.values()]
    ys = [float(p[1]) for p in positions.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-9)
    scale = 240 / span
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    coords = {
        n: (150 + (float(p[0]) - cx) * scale, 140 + (float(p[1]) - cy) * scale)
        for n, p in positions.items()
    }
    marks = [
        f'<path d="M{coords[a][0]:.1f},{coords[a][1]:.1f} L{coords[b][0]:.1f},{coords[b][1]:.1f}" '
        'stroke="#8d9aa8" fill="none" opacity=".55"/>'
        for a, b in g.edges
        if a != b
    ]
    # The radius follows the degree, so a hub is visible as a hub.
    nodes = list(original)
    marks += [
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{4 + degrees[n] * 0.7:.1f}" '
        f'fill="{palette[component_application[nodes[n]] % len(palette)]}" stroke="#fff" '
        f'stroke-width="1.5"><title>component {nodes[n]}; degree {degrees[n]}</title></circle>'
        for n, (x, y) in coords.items()
    ]
    return (
        '<svg viewBox="0 0 300 280" role="img" aria-label="Component graph">'
        + "".join(marks)
        + "</svg>"
    )


def html_table(rows, height=None):
    if not rows:
        return "<p>No rows</p>"
    columns = list(rows[0])
    header = "".join(f"<th>{escape(c)}</th>" for c in columns)
    body = "".join(
        "<tr>"
        + "".join(
            '<td class="empty">empty</td>'
            if r.get(c) in (None, "")
            else f"<td>{escape(str(r[c]))}</td>"
            for c in columns
        )
        + "</tr>"
        for r in rows
    )
    style = f' style="height:{height}"' if height else ""
    return (
        f'<div class="scroll"{style}><table><thead><tr>{header}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def stage_panel(key, stages):
    """All three tables for one stage, each in a box of fixed height.

    Fixed heights mean the tables sit at the same place in every tab, so
    switching stages compares a table against itself without scrolling.
    """
    blocks = "".join(
        f'<h3>{escape(name)} <span class="count">{len(rows)} rows</span></h3>'
        + html_table(rows, height="20rem")
        for name, rows in stages[key].items()
    )
    return f'<div class="panel" data-stage="{key}">{blocks}</div>'


def same_shares(anonymized, synthetic, columns):
    """Whether every fitted category share came back unchanged in the synthetic tables."""
    return all(
        dict(shares(anonymized[table], column)) == dict(shares(synthetic[table], column))
        for table, column in columns
    )


def write_report(output, stages, seed, induction, fitted):
    import networkx as nx

    source, anonymized, synthetic = (stages[key] for key in ("source", "anonymized", "synthetic"))
    counts = {key: metrics(stages[key]) for key, _ in GRAPHS}
    categories = [
        (table, column)
        for (table, column), (kind, _) in fitted["columns"].items()
        if kind == "category"
    ]
    summary = {
        "seed": seed,
        "counts": counts,
        "anonymization_preserves_every_edge": topology(source) == topology(anonymized),
        "synthetic_matches_source_row_counts": all(
            counts["synthetic"][t] == counts["source"][t]
            for t in ("applications", "components", "interfaces")
        ),
        # A synthetic graph isomorphic to the source would be the source
        # structure relabelled, which is the one outcome step 3 must not produce.
        "synthetic_graph_differs_from_source": not nx.is_isomorphic(
            graph(source), graph(synthetic)
        ),
        "synthetic_reproduces_every_category_share": same_shares(anonymized, synthetic, categories),
        "foreign_keys": "validated",
    }
    unchanged = summary["anonymization_preserves_every_edge"]
    chosen = induction["chosen"]
    rejected = len(induction["candidates"][1]) - len(chosen["foreign_keys"])

    graphs = "".join(
        f"<figure>{graph_svg(stages[key])}<figcaption>{escape(label)}</figcaption></figure>"
        for key, label in GRAPHS
    )
    switch = "".join(
        f'<input type="radio" name="stage" id="tab-{key}"{" checked" if i == 0 else ""}>'
        for i, (key, _, _) in enumerate(STAGES)
    )
    labels = "".join(f'<label for="tab-{key}">{escape(name)}</label>' for key, name, _ in STAGES)
    notes = "".join(
        f'<p class="stage-note" data-stage="{key}">{escape(note)}</p>' for key, _, note in STAGES
    )
    panels = "".join(stage_panel(key, stages) for key, _, _ in STAGES)
    measures = [
        {
            "Measure": label,
            "Source": counts["source"][key],
            "Anonymized": counts["anonymized"][key],
            "Synthetic": counts["synthetic"][key],
        }
        for key, label in MEASURES
    ]
    selected = "".join(
        f'#tab-{key}:checked~.tabs label[for="tab-{key}"]{{background:#fff;border-color:#9aa9b8;'
        f"font-weight:600}}#tab-{key}:checked~.panel[data-stage={key}],"
        f'#tab-{key}:checked~.stage-note[data-stage="{key}"]{{display:block}}'
        for key, _, _ in STAGES
    )
    page = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Anonymization and synthetic generation report</title><link rel="icon" href="data:,">
<style>
:root{{font-family:system-ui,sans-serif;color:#1b2b3a;background:#f4f6f8;line-height:1.55}}
body{{max-width:1080px;margin:32px auto;padding:0 22px}}
h1{{font-size:1.35rem;margin:0 0 10px;font-weight:600}}
h2{{font-size:1.05rem;margin:34px 0 8px;font-weight:600}}
h3{{font-size:.9rem;margin:16px 0 6px;font-weight:600}}
p{{max-width:80ch;margin:8px 0}}.count{{color:#6b7b8c;font-weight:400}}
.story{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:18px 0 26px}}
.story div{{background:#fff;border:1px solid #dde3e9;border-top:4px solid #356f6b;border-radius:5px;padding:12px 14px}}
.story b{{display:block;margin-bottom:3px}}
.graphs{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:12px 0}}
figure{{margin:0;background:#fff;border:1px solid #dde3e9;border-radius:5px;padding:10px}}
figure svg{{width:100%;display:block}}figcaption{{font-size:.85rem;text-align:center}}
.tabs{{display:flex;gap:6px;margin:14px 0 4px;flex-wrap:wrap}}
.tabs label{{padding:6px 14px;background:#e8edf1;border:1px solid #dde3e9;border-radius:4px;cursor:pointer;font-size:.9rem}}
input[name=stage]{{position:absolute;opacity:0;pointer-events:none}}
.panel,.stage-note{{display:none}}{selected}
.scroll{{overflow:auto;background:#fff;border:1px solid #dde3e9;border-radius:5px}}
table{{border-collapse:collapse;width:100%;font-size:.85rem}}
th,td{{text-align:left;border-bottom:1px solid #e6ebef;padding:6px 10px;vertical-align:top;
white-space:pre-wrap;overflow-wrap:anywhere;min-width:7rem}}
th{{background:#eef2f5;position:sticky;top:0;font-weight:600}}
.empty{{color:#93a1b0;font-style:italic}}code{{background:#e8edf1;padding:1px 4px}}
@media(max-width:800px){{.story,.graphs{{grid-template-columns:1fr}}}}
</style>
<h1>Anonymization and synthetic generation of an architecture landscape</h1>
<div class="story">
<div><b>1. Input CSV data</b>8 applications, 16 components, and 18 interfaces in three connected
toy tables.</div>
<div><b>2. Protect attributes</b>Seven anonymization and pseudonymization transformations hide
sensitive values—but leave every relationship intact.</div>
<div><b>3. Replace the structure</b>PluRel creates fresh rows and relationships; {escape(MODEL)}
fills their names and descriptions.</div>
</div>

<h2>The point: hidden values do not hide structure</h2>
<div class="graphs">{graphs}</div>
<p>Color denotes the application a component belongs to; node size denotes degree. Sensitive
attributes have changed in the middle graph, but every edge is still present:
<strong>{unchanged}</strong>. Pseudonymizing identifiers would change only the labels, not this
topology.</p>
<p>The synthetic graph replaces the rows and edges while keeping the schema, foreign-key
integrity, table sizes, and selected attribute distributions. It matches the source row counts,
<strong>{summary["synthetic_matches_source_row_counts"]}</strong>, and is not the source graph
relabelled,
<strong>{summary["synthetic_graph_differs_from_source"]}</strong>.</p>

<h2>What happened to the fields</h2>
<p>Every column gets one operation based on what it contains. The examples below come directly
from this run. The paper defines seven transformations; <em>Retain</em> marks deliberately unchanged
columns.</p>
{html_table(examples(source, anonymized))}

<h2>Stages</h2>
<p>The three tables keep the same position in every tab, so switching compares a table against
itself.</p>
<div class="stages">{switch}<div class="tabs">{labels}</div>{notes}{panels}</div>
<p>Between stage 2 and stage 3 the schema is induced. Heuristics list every column that is
present and unique, and every column whose values all appear in another table's candidate key;
{escape(MODEL)} picks the identifier of record for each table and keeps the references that are
real, rejecting {rejected} of the {len(induction["candidates"][1])} foreign key candidates. Each
choice is then checked against the data, and a choice that does not hold stops the run. The
chosen keys are {escape(", ".join(f"{t}.{c}" for t, c in chosen["primary_keys"].items()))}.</p>

<h2>Comparison</h2>
{html_table(measures)}
<p>The source and anonymized graph measures are equal throughout, which is the point of the
middle graph. Row counts and category shares are reproduced because generation is given them:
every fitted share came back unchanged,
<strong>{summary["synthetic_reproduces_every_category_share"]}</strong>. The highest degree and
the number of disconnected groups are not given to it and come from PluRel's own graph priors.
Foreign keys resolve at every stage, and the exported SQLite database passes its own check.</p>

<h2>Scope</h2>
<p>Seed {seed}. What crosses from the anonymized tables into generation is three row counts, the
share of each category, the range of each number and date, the shape of the owner portfolios,
and which port and network follow a protocol and a zone. All of it is recorded in
<code>summary.json</code>. No source row, identifier, pseudonym, or edge crosses over.</p>
<p>This demonstrates the mechanism, not a privacy guarantee. Pseudonyms leak equality, masking
leaks the network and the length of the address, distortion leaves the order of magnitude, and a
shifted date keeps its year. Nothing here establishes differential privacy or anonymity, and the
structural fidelity of the synthetic landscape is not measured. The input tables are invented,
so every file this run writes is safe to share.</p>
</html>"""
    (output / "report.html").write_text(page, encoding="utf-8")
    return summary

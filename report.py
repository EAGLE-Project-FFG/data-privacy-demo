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


def graph_svg(tables, graph_id="graph"):
    import networkx as nx

    width, height = 440, 360
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
    applications = {row["application_id"]: row for row in tables["applications"]}
    application_indices = {
        row["application_id"]: index for index, row in enumerate(tables["applications"])
    }
    components = {row["component_id"]: row for row in tables["components"]}
    application_nodes = {
        row["application_id"]: ("application", index)
        for index, row in enumerate(tables["applications"])
    }
    component_nodes = {
        row["component_id"]: ("component", index)
        for index, row in enumerate(tables["components"])
    }
    layout = nx.Graph()
    layout.add_nodes_from(application_nodes.values())
    layout.add_nodes_from(component_nodes.values())
    layout.add_edges_from(
        (application_nodes[row["application_id"]], component_nodes[row["component_id"]])
        for row in tables["components"]
    )
    layout.add_edges_from(
        (component_nodes[row["source_component_id"]], component_nodes[row["target_component_id"]])
        for row in tables["interfaces"]
    )
    # Position-based node identities keep source and anonymized layouts congruent
    # even though their identifiers differ.
    positions = nx.kamada_kawai_layout(layout)
    xs = [float(p[0]) for p in positions.values()]
    ys = [float(p[1]) for p in positions.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-9)
    scale = 280 / span
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    coords = {
        n: (width / 2 + (float(p[0]) - cx) * scale, height / 2 + (float(p[1]) - cy) * scale)
        for n, p in positions.items()
    }
    marker_id = f"arrow-{escape(graph_id)}"
    marks = [
        f'<defs><marker id="{marker_id}" viewBox="0 0 10 10" refX="11" refY="5" '
        'markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#667786"/></marker></defs>'
    ]
    for component in tables["components"]:
        a = application_nodes[component["application_id"]]
        b = component_nodes[component["component_id"]]
        marks.append(
            f'<path class="ownership-edge" d="M{coords[a][0]:.1f},{coords[a][1]:.1f} '
            f'L{coords[b][0]:.1f},{coords[b][1]:.1f}"/>'
        )
    for interface in tables["interfaces"]:
        source = components[interface["source_component_id"]]
        target = components[interface["target_component_id"]]
        a = component_nodes[source["component_id"]]
        b = component_nodes[target["component_id"]]
        path = f"M{coords[a][0]:.1f},{coords[a][1]:.1f} L{coords[b][0]:.1f},{coords[b][1]:.1f}"
        detail = escape(
            f'{source["name"]} → {target["name"]}\n'
            f'Protocol: {interface["protocol"]}\nPort: {interface["port"]}\n'
            f'{interface["description"]}',
            quote=True,
        )
        marks.append(
            f'<path class="interface-hit graph-detail" tabindex="0" d="{path}" '
            f'data-tooltip="{detail}" aria-label="{detail}"/>'
            f'<path class="interface-edge" d="{path}" marker-end="url(#{marker_id})"/>'
        )
    component_graph = graph(tables).to_undirected()
    degrees = dict(component_graph.degree())
    application_sizes = Counter(row["application_id"] for row in tables["components"])
    for index, application in enumerate(tables["applications"]):
        x, y = coords[application_nodes[application["application_id"]]]
        color = palette[index % len(palette)]
        detail = escape(
            f'{application["name"]}\nOwner: {application["owner"]}\n'
            f'Criticality: {application["criticality"]}\n'
            f'Components: {application_sizes[application["application_id"]]}',
            quote=True,
        )
        application_name = str(application["name"])
        label = escape(
            application_name if len(application_name) <= 24 else application_name[:23] + "…"
        )
        label_x = x - 12 if x > width / 2 else x + 12
        anchor = "end" if x > width / 2 else "start"
        marks.append(
            f'<g class="application-node graph-detail" tabindex="0" data-tooltip="{detail}" '
            f'aria-label="{detail}"><circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{color}"/>'
            f'<text x="{label_x:.1f}" y="{y + 3:.1f}" text-anchor="{anchor}">{label}</text></g>'
        )
    for component in tables["components"]:
        x, y = coords[component_nodes[component["component_id"]]]
        application = applications[component["application_id"]]
        degree = degrees[component["component_id"]]
        detail = escape(
            f'{component["name"]}\nApplication: {application["name"]}\n'
            f'Type: {component["component_type"]}\nInterface degree: {degree}',
            quote=True,
        )
        marks.append(
            f'<circle class="component-node graph-detail" tabindex="0" cx="{x:.1f}" cy="{y:.1f}" '
            f'r="{4 + degree * 0.6:.1f}" '
            f'fill="{palette[application_indices[component["application_id"]] % len(palette)]}" '
            f'data-tooltip="{detail}" aria-label="{detail}"/>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Applications, components, and directed interfaces">'
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


def write_report(output, stages, seed, fitted):
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
    graphs = "".join(
        f"<figure>{graph_svg(stages[key], key)}<figcaption>{escape(label)}</figcaption></figure>"
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
.ownership-edge{{fill:none;stroke:#aeb8c2;stroke-width:1;stroke-dasharray:3 3}}
.interface-edge{{fill:none;stroke:#667786;stroke-width:1.25;pointer-events:none}}
.interface-hit{{fill:none;stroke:transparent;stroke-width:10;pointer-events:stroke;cursor:pointer}}
.interface-hit:hover+.interface-edge,.interface-hit:focus+.interface-edge{{stroke:#172b3a;stroke-width:2.5}}
.component-node{{stroke:#fff;stroke-width:1.5;cursor:pointer}}
.component-node:hover,.component-node:focus{{outline:none;stroke:#172b3a;stroke-width:2.5}}
.application-node{{cursor:pointer;outline:none}}.application-node circle{{stroke:#fff;stroke-width:2}}
.application-node:hover circle,.application-node:focus circle{{stroke:#172b3a;stroke-width:3}}
.application-node text{{font-size:8px;font-weight:600;paint-order:stroke;stroke:#fff;stroke-width:3;
stroke-linejoin:round;fill:#172b3a;pointer-events:none}}
.graph-legend{{display:flex;gap:18px;align-items:center;justify-content:center;flex-wrap:wrap;
color:#536474;font-size:.8rem;margin:8px 0 14px}}
.graph-legend span{{display:flex;align-items:center;gap:5px}}.graph-key{{display:inline-block}}
.graph-key.application{{width:12px;height:12px;border-radius:50%;background:#247ba0}}
.graph-key.component{{width:7px;height:7px;border-radius:50%;background:#247ba0}}
.graph-key.ownership{{width:20px;border-top:1px dashed #8d9aa8}}
.graph-key.interface{{font-size:1.25rem;line-height:.5;color:#667786}}
.graph-tooltip{{position:fixed;z-index:10;pointer-events:none;opacity:0;white-space:pre-line;
max-width:300px;background:#1b2b3a;color:#fff;border-radius:4px;padding:7px 10px;font-size:.8rem;
box-shadow:0 3px 10px #0003;transition:opacity .1s}}
.graph-tooltip.visible{{opacity:1}}
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
<div><b>1. Input CSV data</b>3 connected Tables with: 8 applications, 16 components, and 18 interfaces.</div>
<div><b>2. Protect attributes</b>Seven anonymization and pseudonymization transformations to hide
sensitive values (but leave every relationship intact).</div>
<div><b>3. Replace the structure</b>PluRel creates new rows and relationships. The LLM fills their names and descriptions.</div>
</div>

<h2>Topology</h2>
<div class="graphs">{graphs}</div>
<div class="graph-legend">
<span><i class="graph-key application"></i>Application</span>
<span><i class="graph-key component"></i>Component</span>
<span><i class="graph-key ownership"></i>Ownership</span>
<span><i class="graph-key interface">→</i>Interface</span>
</div>
<div class="graph-tooltip" role="tooltip"></div>

<h2>Applied Techniques</h2>
{html_table(examples(source, anonymized))}

<h2>Stages</h2>
<div class="stages">{switch}<div class="tabs">{labels}</div>{notes}{panels}</div>

<h2>Comparison</h2>
{html_table(measures)}
<script>
const tooltip=document.querySelector('.graph-tooltip');
const show=(node,x,y)=>{{
  tooltip.textContent=node.dataset.tooltip;
  tooltip.classList.add('visible');
  tooltip.style.left=`${{Math.max(8,Math.min(x+12,innerWidth-tooltip.offsetWidth-8))}}px`;
  tooltip.style.top=`${{Math.max(8,Math.min(y+12,innerHeight-tooltip.offsetHeight-8))}}px`;
}};
document.querySelectorAll('.graph-detail').forEach(node=>{{
  node.addEventListener('pointerenter',event=>show(node,event.clientX,event.clientY));
  node.addEventListener('pointermove',event=>show(node,event.clientX,event.clientY));
  node.addEventListener('pointerleave',()=>tooltip.classList.remove('visible'));
  node.addEventListener('focus',()=>{{const box=node.getBoundingClientRect();show(node,box.right,box.top)}});
  node.addEventListener('blur',()=>tooltip.classList.remove('visible'));
}});
</script>
</html>"""
    (output / "report.html").write_text(page, encoding="utf-8")
    return summary

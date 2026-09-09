"""The seven field operations of the paper, one per column.

`PLAN` assigns exactly one operation to every column of every table, and
`OPERATIONS` says what each one does. Both drive the run and the report, so the
documented rule and the executed rule cannot drift apart. A column that `PLAN`
does not mention is retained unchanged.

Pseudonymization is reversible: a hash of the value selects an entry from a word
list, the pair is recorded in `pseudonym_map.json`, and `Pseudonyms.restore`
reads it back. Deletion, removal, masking, distortion and date shifting are not
reversible. The latter two are deterministic so repeated source values receive
the same plausible replacement, but the original value is not recorded.
"""

import calendar
import datetime
import hashlib

# Word lists. Pseudonyms are drawn from these, never invented, so an alias
# carries no fragment of the value it replaced.
PEOPLE = [
    f"{first} {last}"
    for first in ("Alex", "Robin", "Sam", "Chris", "Jordan", "Casey", "Morgan", "Riley")
    for last in ("Doe", "Roe", "Public", "Bloggs", "Noakes", "Stiles")
]
PRODUCTS = [
    "Aurora",
    "Beacon",
    "Cascade",
    "Delta",
    "Ember",
    "Fathom",
    "Granite",
    "Harbor",
    "Juniper",
    "Kestrel",
    "Lantern",
    "Meridian",
    "Northwind",
    "Orchard",
    "Pinnacle",
    "Quarry",
    "Ridgeline",
    "Summit",
    "Tidewater",
    "Vantage",
]
HOSTS = [
    f"svc-{word}"
    for word in (
        "basalt",
        "cobalt",
        "cinder",
        "dune",
        "flint",
        "garnet",
        "halite",
        "indigo",
        "jasper",
        "kaolin",
        "lignite",
        "marl",
        "nacre",
        "obsidian",
        "pumice",
        "quartz",
        "rutile",
        "shale",
        "topaz",
        "umber",
        "verdite",
        "wolfram",
        "xenon",
        "zircon",
    )
]
WORD_LISTS = {"people": PEOPLE, "applications": PRODUCTS, "components": HOSTS}

DELETED = ""  # What Deletion writes; configurable per the paper.
MASK = "*"  # What Masking writes over each character it hides.
SPREAD = 0.2  # Distortion moves a number by at most this fraction of itself.

# One operation per column. Everything absent from this table is retained.
PLAN = {
    "applications": {
        "name": "pseudonymize_component",
        "owner": "pseudonymize_name",
        "cost_center": "remove_column",
        "annual_cost_eur": "distortion",
        "go_live_date": "shift_date",
        "description": "deletion",
    },
    "components": {
        "name": "pseudonymize_component",
        "ip_address": "masking",
        "description": "deletion",
    },
    "interfaces": {
        "description": "deletion",
    },
}

# Name, effect and what the operation leaves behind, for the report.
OPERATIONS = {
    "pseudonymize_name": (
        "Pseudonymize Name",
        "A hash of the person's name selects a replacement from a name word list.",
        "Equality only: that the same person is responsible for several applications.",
    ),
    "pseudonymize_component": (
        "Pseudonymize Component",
        "A hash of the system's name selects a replacement from a component word list.",
        "Equality only: which rows refer to the same system, not what it is called.",
    ),
    "deletion": (
        "Deletion",
        "Replaces the field's content with an empty string.",
        "Only that the column exists.",
    ),
    "remove_column": (
        "Remove Column",
        "Removes the column from the exported CSV.",
        "Nothing. The column is absent from every output.",
    ),
    "masking": (
        "Masking",
        f"Replaces each character after the leading octet with '{MASK}', preserving length.",
        "The network the host sits in, and the length of the address.",
    ),
    "distortion": (
        "Distortion",
        f"Moves the number by up to {SPREAD:.0%} of itself, by a hash of the value.",
        "The order of magnitude, and roughly the ranking between rows.",
    ),
    "shift_date": (
        "Shift Date",
        "Replaces the date with another valid date; day and month move independently.",
        "The year, so the age of the system stays readable.",
    ),
    "retain": (
        "Retain",
        "Leaves the value unchanged.",
        "The exact value. These columns identify nobody on their own.",
    ),
}


def digest(value):
    """A stable integer for a value, so every operation is deterministic."""
    return int.from_bytes(hashlib.blake2b(str(value).encode(), digest_size=8).digest(), "big")


class Pseudonyms:
    """Word-list pseudonyms: stable within a run, distinct, and reversible.

    The hash picks the entry, and the search steps forward from there when the
    entry is already taken. Probing is what keeps the mapping injective, and an
    injective mapping is what makes `restore` well defined.
    """

    def __init__(self):
        self.mapping = {}

    def replace(self, value, domain):
        if value in (None, ""):
            return None
        taken = self.mapping.setdefault(domain, {})
        if str(value) in taken:
            return taken[str(value)]
        words = WORD_LISTS[domain]
        for step in range(len(words)):
            candidate = words[(digest(value) + step) % len(words)]
            if candidate not in taken.values():
                taken[str(value)] = candidate
                return candidate
        raise ValueError(f"The {domain} word list has fewer entries than there are values")

    def restore(self, value, domain):
        return {alias: original for original, alias in self.mapping[domain].items()}[value]


def mask(value):
    """Hide every octet of an address but the first, one mask character per digit.

    The length is preserved, as in the paper, and so is the leading octet: the
    network a host sits in is technical context worth keeping, while the host
    itself is not. Retaining a prefix is a deliberate trade, not a free one.
    """
    head, *rest = str(value).split(".")
    return ".".join([head] + [MASK * len(part) for part in rest])


def distort(value, spread=SPREAD):
    """Move a number to another number within a predefined range around it."""
    offset = 2 * (digest(value) % 10_000) / 9_999 - 1  # in [-1, 1]
    return round(value * (1 + spread * offset))


def shift_date(value):
    """Replace a date with another valid date, varying the day and the month.

    Day and month are shifted by separate amounts, and the day is folded into
    the length of the month it lands in, so the result is always a real date.
    The year is kept: it carries the age of the system, which the landscape is
    analyzed for, and on its own it identifies nothing.
    """
    date = datetime.date.fromisoformat(str(value))
    key = digest(value)
    month = (date.month - 1 + key % 12) % 12 + 1
    length = calendar.monthrange(date.year, month)[1]
    day = (date.day - 1 + (key // 12) % 28) % length + 1
    return datetime.date(date.year, month, day).isoformat()


def apply_operation(operation, value, table, aliases):
    """One value, one operation. `remove_column` is handled by the caller."""
    if value in (None, ""):
        return None if operation != "deletion" else DELETED
    if operation == "pseudonymize_name":
        return aliases.replace(value, "people")
    if operation == "pseudonymize_component":
        return aliases.replace(value, table)
    if operation == "deletion":
        return DELETED
    if operation == "masking":
        return mask(value)
    if operation == "distortion":
        return distort(value)
    if operation == "shift_date":
        return shift_date(value)
    if operation == "retain":
        return value
    raise ValueError(f"Unknown operation {operation!r}")


def anonymize(tables):
    """Apply `PLAN` to every column, keeping row order and every join intact."""
    aliases = Pseudonyms()
    result = {}
    for table, rows in tables.items():
        plan = PLAN[table]
        result[table] = []
        for row in rows:
            item = {}
            for column, value in row.items():
                operation = plan.get(column, "retain")
                if operation == "remove_column":
                    continue
                item[column] = apply_operation(operation, value, table, aliases)
            result[table].append(item)
    return result, aliases


def examples(source, anonymized):
    """One row per operation, with the columns it covers and a before-and-after pair.

    Grouped by operation rather than by column, so the table reads like the
    paper's list of operations with this run's values filled in. The example
    comes from the first row of that column that is not empty.
    """
    columns = {}
    for table, rows in source.items():
        for column in rows[0]:
            columns.setdefault(PLAN[table].get(column, "retain"), []).append((table, column))
    rows = []
    for operation, (name, effect, survives) in OPERATIONS.items():
        covered = columns.get(operation)
        if not covered:
            continue
        # Prefer a column that is not an identifier, so a retained example shows
        # something more telling than an integer key mapping to itself.
        table, column = next((c for c in covered if not c[1].endswith("_id")), covered[0])
        source_rows = source[table]
        position, before = next(
            ((i, r[column]) for i, r in enumerate(source_rows) if r[column] not in (None, "")),
            (0, None),
        )
        after = (
            "(column removed)"
            if operation == "remove_column"
            else anonymized[table][position][column]
        )
        rows.append(
            {
                "Operation": name,
                "Columns": ", ".join(f"{t}.{c}" for t, c in covered),
                "Effect": effect,
                "Example": f"{before}  \u2192  {after if after not in (None, '') else '(empty)'}",
                "What survives": survives,
            }
        )
    return rows

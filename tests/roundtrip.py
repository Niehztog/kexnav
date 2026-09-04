"""Round-trip gate for the NAV3 reader/writer.

Reads every shipped nav file, parses it, serialises it back and demands the
bytes be identical. This is the objective test that the format in
``kexnav/nav3.py`` is understood well enough to *generate* nav files: anything
misread would either fail to parse, leave trailing bytes, or come back
different.

It also checks the structural invariants a generator must honour, and prints a
profile of every enum field across the corpus.

    python3 -m tests.roundtrip                  # the retail pak0.pak
    python3 -m tests.roundtrip --pak PATH       # a different pak
    python3 -m tests.roundtrip --dir PATH       # a directory of loose .nav files
"""

import argparse
import collections
import os
import sys

from kexnav import env, nav3, validate
from kexnav.pak import Pak

NAV_PREFIX = "bots/navigation/"


def collect(args):
    """Yield (name, bytes) for every nav file the arguments point at."""
    if args.dir:
        for root, _, files in os.walk(args.dir):
            for fn in sorted(files):
                if fn.lower().endswith(".nav"):
                    path = os.path.join(root, fn)
                    with open(path, "rb") as fp:
                        yield os.path.relpath(path, args.dir), fp.read()
        return
    with Pak(env.find_pak(args.pak)) as pak:
        for name in sorted(n for n in pak.names
                           if n.startswith(NAV_PREFIX) and n.endswith(".nav")):
            yield name[len(NAV_PREFIX):], pak.read(name)


def first_difference(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else None


def profile(nav, report):
    """Tally every enum field across the corpus. The structural checks live in
    ``kexnav.validate`` so that a generated file is held to the same rules; the
    type/traversal rule is skipped here because this is where that observation
    comes from in the first place."""
    for link in nav.links:
        report["link_type"][link.type_name] += 1
        report["link_flags"]["|".join(link.flag_names) or "0"] += 1
        report["type_vs_traversal"][(link.type_name, link.has_traversal)] += 1
    for t in nav.traversals:
        report["traversal_funnel"][funnel_state(t)] += 1
        if t.ladder_plane is None:
            report["ladder_plane"]["absent (v<4)"] += 1
        else:
            report["ladder_plane"]["non-zero" if any(t.ladder_plane) else "zero"] += 1
    for node in nav.nodes:
        report["node_flags"]["|".join(node.flag_names) or "0"] += 1
        report["node_radius"][node.radius] += 1
    report["heuristic"][round(nav.heuristic, 6)] += 1
    report["edicts_per_file"][len(nav.edicts)] += 1
    return validate.check(nav, skip=("type/traversal rule",))


def funnel_state(traversal):
    """A funnel has *two* spellings of "unset". 1e30 in every component is the
    documented one; an all-zero vector is the other, and it is not a point at
    the world origin -- all 98 of badlands.nav's ledge funnels are (0,0,0),
    and it accounts for every one of the 1412 ``WALK_OFF_LEDGE`` traversals
    that looked like they carried a funnel."""
    if all(c >= nav3.UNSET_COORD for c in traversal.funnel):
        return "unset (1e30)"
    if not any(traversal.funnel):
        return "unset (zero)"
    return "point"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m tests.roundtrip", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--pak", help="pak archive to read nav files from "
                                   "(default: the first kexnav.env.find_pak "
                                   "turns up)")
    src.add_argument("--dir", help="directory of loose .nav files instead")
    ap.add_argument("-v", "--verbose", action="store_true", help="one line per file")
    args = ap.parse_args(argv)

    report = collections.defaultdict(collections.Counter)
    skipped = []
    exact = failed = 0
    totals = collections.Counter()
    violations = []

    for name, raw in collect(args):
        try:
            nav = nav3.loads(raw)
        except nav3.UnsupportedVersion as exc:
            skipped.append((name, exc.version))
            continue
        except nav3.NavError as exc:
            print(f"  PARSE FAIL  {name}: {exc}")
            failed += 1
            continue

        again = nav3.dumps(nav)
        if again != raw:
            where = first_difference(raw, again)
            print(f"  MISMATCH    {name}: {len(raw)} in / {len(again)} out, "
                  f"first difference at byte {where}")
            print(f"              orig {raw[where:where+16].hex(' ')}")
            print(f"              ours {again[where:where+16].hex(' ')}")
            failed += 1
            continue

        exact += 1
        totals["nodes"] += len(nav.nodes)
        totals["links"] += len(nav.links)
        totals["traversals"] += len(nav.traversals)
        totals["edicts"] += len(nav.edicts)
        report["version"][nav.version] += 1
        for problem in profile(nav, report):
            violations.append(f"{name}: {problem}")
        if args.verbose:
            print(f"  ok  {name:<24} v{nav.version} {len(nav.nodes):>5} nodes "
                  f"{len(nav.links):>6} links {len(nav.traversals):>5} traversals "
                  f"{len(nav.edicts):>3} edicts")

    total = exact + failed
    print()
    print(f"round trip : {exact}/{total} byte-identical" + (f", {failed} FAILED" if failed else ""))
    if skipped:
        by_version = collections.Counter(v for _, v in skipped)
        detail = ", ".join(f"v{v}: {n}" for v, n in sorted(by_version.items()))
        print(f"skipped    : {len(skipped)} file(s) of an unmodelled version ({detail})")
    print(f"corpus     : {totals['nodes']} nodes, {totals['links']} links, "
          f"{totals['traversals']} traversals, {totals['edicts']} nav edicts")

    print("\nstructural invariants")
    if violations:
        print(f"  {len(violations)} violation(s):")
        for v in violations[:20]:
            print(f"    {v}")
        if len(violations) > 20:
            print(f"    ... and {len(violations) - 20} more")
    else:
        print("  clean: link ranges contiguous and in node order; every target,")
        print("  traversal and edict link index in range; every traversal used by")
        print("  exactly one link; ladder plane a unit vector, non-zero exactly")
        print("  for LADDER links; every count inside its u16 field")

    print("\nfield profile (whole corpus)")
    for key in ("version", "link_type", "link_flags", "type_vs_traversal", "node_flags",
                "node_radius", "traversal_funnel", "ladder_plane",
                "heuristic", "edicts_per_file"):
        counts = report[key]
        shown = sorted(counts.items(), key=lambda kv: -kv[1])[:10]
        rendered = ", ".join(f"{k}: {v}" for k, v in shown)
        more = "" if len(counts) <= 10 else f", (+{len(counts) - 10} more)"
        print(f"  {key:<18} {rendered}{more}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

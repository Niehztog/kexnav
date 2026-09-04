"""Gate for the AAS reader.

The nav side has a byte-exact round trip to prove it (``python3 -m
tests.roundtrip``). An AAS file cannot be round-tripped that cheaply --
nothing here writes one -- so this leans on the next best thing: **BSPC's own
reading of the same file**.

``bspc -aasinfo`` prints thirteen lump counts and a travel-type histogram.
This parses the file independently and demands every one of those numbers
match. That is an independent implementation agreeing on the parse, which is
the same standard the nav format was held to.

It also checks the structural invariants the converter relies on -- the two
CSR index runs, every cross-lump index in range, and the dummy record 0 -- and
verifies the ``AAS_PointAreaNum`` tree walk lands back in the area it started
from, since the oracle comparison in ``kexnav.py check`` depends on it.

    python3 -m tests.aascheck FILE_OR_DIR [...]
    python3 -m tests.aascheck --no-bspc ...         # skip the cross-check
"""

import argparse
import collections
import os
import re
import subprocess
import sys

from kexnav import aas, env

#: ``bspc -aasinfo`` prints its histogram in ``aasfile.h`` travel-type order,
#: one line per type, as ``<count> <name>``.
BSPC_TRAVEL_ORDER = [
    aas.TravelType.WALK, aas.TravelType.CROUCH, aas.TravelType.BARRIERJUMP,
    aas.TravelType.JUMP, aas.TravelType.LADDER, aas.TravelType.WALKOFFLEDGE,
    aas.TravelType.SWIM, aas.TravelType.WATERJUMP, aas.TravelType.TELEPORT,
    aas.TravelType.ELEVATOR, aas.TravelType.ROCKETJUMP, aas.TravelType.BFGJUMP,
    aas.TravelType.GRAPPLEHOOK, aas.TravelType.DOUBLEJUMP,
    aas.TravelType.RAMPJUMP, aas.TravelType.STRAFEJUMP,
    aas.TravelType.JUMPPAD, aas.TravelType.FUNCBOB,
]

_TOTAL_LINE = re.compile(r"^\s*(\w+) = (\d+)\s*$")
_TRAVEL_LINE = re.compile(r"^\s*(\d+) ([a-z ]+?)\s*$")


def bspc_aasinfo(bspc, path):
    """Run ``bspc -aasinfo`` and return (totals, travel histogram), or None if
    BSPC would not read the file -- it only accepts version 5.

    Invoked as ``cd <dir> && bspc -aasinfo <basename>``: the filename argument
    goes into a fixed-size buffer, and a long absolute path smashes the stack
    before anything is printed. BSPC also writes ``bspc.log`` into the working
    directory, which this keeps next to the input rather than in the tree.
    """
    try:
        out = subprocess.run([bspc, "-aasinfo", os.path.basename(path)],
                             cwd=os.path.dirname(os.path.abspath(path)),
                             capture_output=True, text=True,
                             timeout=120).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"could not run {bspc}: {exc}"
    if "ERROR" in out:
        return None, next((l.strip() for l in out.splitlines() if "ERROR" in l), "error")

    totals, travel = {}, {}
    for line in out.splitlines():
        m = _TOTAL_LINE.match(line)
        if m:
            totals[m.group(1)] = int(m.group(2))
            continue
        m = _TRAVEL_LINE.match(line)
        if m:
            travel[m.group(2).replace(" ", "")] = int(m.group(1))
    if not totals:
        return None, "no totals in bspc output"
    return (totals, travel), None


def travel_histogram(a):
    """Counts per travel type, keyed the way ``-aasinfo`` names them. Record 0
    is the dummy and BSPC does not count it."""
    counts = collections.Counter()
    for r in a.reachability[1:]:
        counts[r.travel_type] += 1
    return {t.name.lower(): counts.get(int(t), 0) for t in BSPC_TRAVEL_ORDER}


def check_invariants(a):
    """Return a list of human-readable violations."""
    problems = []

    if a.areas and any(a.areas[0].mins) or a.areas and a.areas[0].numfaces:
        problems.append("area 0 is not the expected zero dummy")
    if a.reachability and a.reachability[0].traveltype:
        problems.append("reachability 0 is not the expected zero dummy")

    # aas_areasettings_t indexes the reachability lump as a CSR run, in area
    # order. The converter walks it that way. The number of dummy records the
    # run starts past is not fixed -- BSPC leaves one, but the Gladiator
    # engine, which wrote the v3 reach lumps itself, left two on q2ctf1 -- so
    # take the start from the data and check only that it is contiguous
    # afterwards and consumes the lump. Area 0 is excluded throughout: it is
    # the dummy, and the v3 files have it claiming a reachability of its own.
    first = next((s.firstreachablearea for i, s in enumerate(a.areasettings)
                  if i and s.numreachableareas), None)
    if first is not None:
        running = first
        for i, s in enumerate(a.areasettings):
            if not i or not s.numreachableareas:
                continue
            if s.firstreachablearea != running:
                problems.append(f"area {i}: firstreachablearea "
                                f"{s.firstreachablearea}, expected {running}")
                running = s.firstreachablearea
            running += s.numreachableareas
        if running != len(a.reachability):
            problems.append(f"area reachability runs end at {running}, "
                            f"lump has {len(a.reachability)} records")

    running = 0
    for i, c in enumerate(a.clusters):
        if c.firstportal != running:
            problems.append(f"cluster {i}: firstportal {c.firstportal}, expected {running}")
            running = c.firstportal
        running += c.numportals
    if running != len(a.portalindex):
        problems.append(f"cluster portal runs cover {running} entries, "
                        f"portalindex has {len(a.portalindex)}")

    # only areanum is checkable -- facenum and edgenum are a per-travel-type
    # payload, see Reachability's docstring
    for i, r in enumerate(a.reachability):
        if not 0 <= r.areanum < len(a.areas):
            problems.append(f"reachability {i}: areanum {r.areanum} out of range")
    for i, area in enumerate(a.areas):
        if area.firstface + area.numfaces > len(a.faceindex):
            problems.append(f"area {i}: face run past the faceindex")
    for i, f in enumerate(a.faces):
        if not 0 <= f.planenum < len(a.planes):
            problems.append(f"face {i}: planenum {f.planenum} out of range")
        if f.firstedge + f.numedges > len(a.edgeindex):
            problems.append(f"face {i}: edge run past the edgeindex")
    for i, n in enumerate(a.nodes):
        if not 0 <= n.planenum < len(a.planes):
            problems.append(f"node {i}: planenum {n.planenum} out of range")
        for c in n.children:
            if c > 0 and c >= len(a.nodes):
                problems.append(f"node {i}: child {c} out of range")
            if c < 0 and -c >= len(a.areas):
                problems.append(f"node {i}: child area {-c} out of range")
    for i, p in enumerate(a.portalindex):
        if not 0 <= p < max(len(a.portals), 1):
            problems.append(f"portalindex {i}: portal {p} out of range")

    return problems


#: How many of an AAS file's sampled area centres may fail to resolve back to
#: their own area before it counts as a broken parse rather than a quirk of
#: the data.
#:
#: BSPC's ``center`` is the average of an area's boundary vertexes, which is
#: interior for a convex body -- but BSPC also *merges* areas
#: (``aas_areamerging.c``, and the log prints "N areas merged"), and a merged
#: area need not stay convex. Observed rate across the maps compiled here: 1
#: miss in 9368. So a couple of misses says nothing about the reader, while a
#: systematic failure would mean the tree walk or the plane lump is misread.
TREE_MISS_TOLERANCE = 0.01


def check_tree(a, limit=400):
    """``AAS_PointAreaNum`` on an area's own centre should return that area.
    See :data:`TREE_MISS_TOLERANCE` for why "should". Returns (hits, tested)."""
    grounded = a.grounded_areas()
    step = max(1, len(grounded) // limit)
    hits = tested = 0
    for num in grounded[::step]:
        tested += 1
        if a.point_area_num(a.areas[num].center) == num:
            hits += 1
    return hits, tested


def collect(paths):
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for fn in sorted(files):
                    if fn.lower().endswith(".aas"):
                        yield os.path.join(root, fn)
        else:
            yield p


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m tests.aascheck", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", default=["."], help=".aas files or directories")
    ap.add_argument("--bspc", help="BSPC binary for the cross-check "
                                   "(default: kexnav.env.find_bspc)")
    ap.add_argument("--no-bspc", action="store_true", help="skip the BSPC cross-check")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    bspc = None if args.no_bspc else env.find_bspc(args.bspc)
    if not args.no_bspc and not bspc:
        print("  note: no BSPC binary found, skipping the cross-check")

    report = collections.defaultdict(collections.Counter)
    parsed = failed = agreed = 0
    unchecked = []
    violations = []
    tree_hits = tree_tested = 0

    for path in collect(args.paths):
        name = os.path.basename(path)
        try:
            a = aas.load(path)
        except aas.AasError as exc:
            print(f"  PARSE FAIL  {name}: {exc}")
            failed += 1
            continue
        parsed += 1
        report["version"][a.version] += 1

        for p in check_invariants(a):
            violations.append(f"{name}: {p}")
        h, t = check_tree(a)
        tree_hits += h
        tree_tested += t
        if t and (t - h) > t * TREE_MISS_TOLERANCE:
            violations.append(f"{name}: point_area_num missed its own area on "
                              f"{t - h} of {t} sampled grounded areas, over the "
                              f"{TREE_MISS_TOLERANCE:.0%} tolerance")

        for r in a.reachability[1:]:
            report["travel"][r.travel_name] += 1
            if r.not_team1 or r.not_team2:
                report["travel_team_flags"][
                    f"{r.travel_name} notteam{'1' if r.not_team1 else ''}"
                    f"{'2' if r.not_team2 else ''}"] += 1
        for i, s in enumerate(a.areasettings):
            if not i:
                continue
            report["area_flags"]["|".join(s.flag_names) or "0"] += 1
            report["area_contents"]["|".join(s.content_names) or "0"] += 1
            report["presence"]["|".join(aas._flag_names(aas.PresenceType, s.presencetype)) or "0"] += 1
        for b in a.bboxes:
            report["bbox"][(b.presencetype, b.mins, b.maxs)] += 1

        if not bspc:
            continue
        info, why = bspc_aasinfo(bspc, path)
        if info is None:
            unchecked.append((name, why))
            continue
        totals, travel = info
        mine = a.totals()
        diffs = [f"{k}: bspc {v}, ours {mine.get(k)}"
                 for k, v in totals.items() if mine.get(k) != v]
        mine_travel = travel_histogram(a)
        diffs += [f"{k}: bspc {v}, ours {mine_travel.get(k)}"
                  for k, v in travel.items() if mine_travel.get(k) != v]
        if diffs:
            print(f"  DISAGREE    {name}: " + "; ".join(diffs))
            failed += 1
            continue
        agreed += 1
        if args.verbose:
            print(f"  ok  {name:<28} v{a.version} {mine['numareas']:>5} areas "
                  f"{len(a.grounded_areas()):>5} grounded "
                  f"{mine['reachabilitysize']:>5} reach")

    print()
    print(f"parsed     : {parsed} file(s)" + (f", {failed} FAILED" if failed else ""))
    if bspc:
        print(f"bspc agrees: {agreed}/{parsed} on all 13 lump counts and the "
              f"travel histogram")
    if unchecked:
        print(f"not cross-checked: {len(unchecked)} -- BSPC reads version 5 only")
        for n, why in unchecked[:3]:
            print(f"    {n}: {why}")
        if len(unchecked) > 3:
            print(f"    ... and {len(unchecked) - 3} more")

    print("\nstructural invariants")
    if violations:
        print(f"  {len(violations)} violation(s):")
        for v in violations[:20]:
            print(f"    {v}")
        if len(violations) > 20:
            print(f"    ... and {len(violations) - 20} more")
    else:
        print("  clean: reachability and portal index runs are contiguous and in")
        print("  order; every cross-lump index in range; record 0 a zero dummy;")
        print(f"  point_area_num returned the right area on "
              f"{tree_hits}/{tree_tested} sampled grounded areas "
              f"({tree_tested - tree_hits} miss(es), tolerance "
              f"{TREE_MISS_TOLERANCE:.0%} per file)")

    print("\nfield profile")
    for key in ("version", "travel", "travel_team_flags", "area_flags",
                "area_contents", "presence", "bbox"):
        counts = report[key]
        if not counts:
            continue
        shown = sorted(counts.items(), key=lambda kv: -kv[1])[:8]
        more = "" if len(counts) <= 8 else f", (+{len(counts) - 8} more)"
        print(f"  {key:<18} " + ", ".join(f"{k}: {v}" for k, v in shown) + more)

    return 1 if failed or violations else 0


if __name__ == "__main__":
    sys.exit(main())

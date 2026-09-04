"""Compare a generated nav file against Nightdive's hand-authored one for the
same map.

The oracle is large: every one of the 174 shipped nav files pairs 1:1 with a
BSP at the matching path in the same pak, so a generated file can be diffed
against a hand-authored one rather than merely inspected.

What is compared, and why each measure:

**Coverage** -- for every Nightdive node, how far to the nearest generated
node. This is the measure that matters. A generated graph that misses a room is
useless however tidy its histograms, and Nightdive's nodes are a hand-checked
sample of the positions a bot needs. Reported as a percentage within one node
radius (32) and within one node spacing.

**Density** -- the reverse distance, and the node count ratio. A generated
graph is *expected* to be denser than a hand-authored one: BSPC splits floor
that a human would have covered with one waypoint. This is reported, not
graded.

**Link type mix** -- side by side. The types BSPC cannot see show up here as
zeros, which is the honest way to display a gap.

**Spawn coverage** -- the share of the map's player spawn points that have a
node nearby. This one needs no oracle, so it is the coverage measure that works
on a map nobody has hand-authored, and it is sharper than it sounds: spawns are
spread deliberately across everywhere a player is meant to go, so a whole wing
missing from the AAS shows up here.

**100% is not the target, though**, which is why Nightdive's own figure is
printed beside it. Measured over the whole corpus, Nightdive's hand-authored
files put a node within one spacing of 2846 of 2930 spawns -- 97.1%, not 100%.
The misses cluster on the ``q64/*`` maps, which carry four spawns each that
Nightdive's own nav does not cover either. Read the two numbers together or the
metric will mislead.

**Invariants** -- the generated file is put through ``kexnav.validate``, the
same checks all 174 shipped files pass.

    python3 kexnav.py check q2dm1 q2dm3
    python3 kexnav.py check --all            # every stock map with a nav file
    python3 kexnav.py check --test           # four isolated-feature maps

Maps with no hand-authored nav file are reported too, on spawn coverage and
invariants alone -- which is how a generated file with no reference gets checked.
"""

import argparse
import collections
import math
import os
import shutil

from kexnav import aas, bsp, convert, env, nav3, validate
from kexnav.pak import Pak
from .generate import compile_aas, compile_best_aas

NAV_PREFIX = "bots/navigation/"

#: Entity classnames that put a player somewhere. Used for the coverage
#: measure that does not need an oracle.
SPAWN_CLASSNAMES = ("info_player_deathmatch", "info_player_start",
                    "info_player_coop", "info_player_team1",
                    "info_player_team2")

#: Nightdive's four isolated-feature test maps, each exercising one thing. The
#: right first targets, and the reason a ladder or barrier regression is
#: visible at all.
TEST_MAPS = ("test/mals_box", "test/mals_ladder_test",
             "test/mals_barrier_test", "test/mals_locked_door_test")


def nearest_distances(from_nodes, to_nodes):
    """For each origin in `from_nodes`, the distance to the closest in
    `to_nodes`. Brute force -- a few hundred nodes each way, so a spatial
    index would be more code than it saves."""
    out = []
    targets = [n.origin for n in to_nodes]
    for n in from_nodes:
        ox, oy, oz = n.origin
        best = None
        for tx, ty, tz in targets:
            d = (tx - ox) ** 2 + (ty - oy) ** 2 + (tz - oz) ** 2
            if best is None or d < best:
                best = d
        out.append(math.sqrt(best) if best is not None else float("inf"))
    return out


def _open_paks(entries):
    """Open every pak the arguments name, expanding a directory to the paks
    inside it. Later paks win, the way the engine's search order works."""
    paths = []
    for entry in (entries or [env.find_pak()]):
        if os.path.isdir(entry):
            paths += sorted(os.path.join(entry, f) for f in os.listdir(entry)
                            if f.lower().endswith(".pak"))
        else:
            paths.append(entry)
    return [Pak(p) for p in paths]


def _read(paks, name):
    """The bytes of `name` from the last pak that has it."""
    for pak in reversed(paks):
        if name in pak.entries:
            return pak.read(name)
    raise KeyError(name)


def _names(paks):
    seen = {}
    for pak in paks:
        for n in pak.names:
            seen[n] = True
    return seen.keys()


def spawn_distances(b, nav):
    """For each player spawn point, the distance to the nearest nav node.

    A spawn's ``origin`` is a *player-origin* position, so the nav-space
    equivalent is that origin less :data:`kexnav.convert.FLOOR_OFFSET` -- the
    same 23.47 that separates AAS space from nav space everywhere else.
    """
    if b is None or not nav.nodes:
        return []
    out = []
    for ent in b.by_classname(*SPAWN_CLASSNAMES):
        x, y, z = bsp.parse_vec3(ent.get("origin", ""))
        z -= convert.FLOOR_OFFSET
        out.append(min(math.dist((x, y, z), n.origin) for n in nav.nodes))
    return out


def quantiles(values):
    if not values:
        return {}
    v = sorted(values)
    def at(p):
        return v[min(len(v) - 1, int(len(v) * p))]
    return {"median": at(0.5), "p90": at(0.9), "max": v[-1]}


def summarise(distances, within):
    q = quantiles(distances)
    if not q:
        return "no nodes"
    parts = [f"{sum(1 for d in distances if d <= w) * 100.0 / len(distances):.0f}% "
             f"within {w:g}" for w in within]
    return (", ".join(parts)
            + f"   median {q['median']:.0f}, p90 {q['p90']:.0f}, max {q['max']:.0f}")


def degree_stats(nav):
    counts = [n.num_links for n in nav.nodes]
    if not counts:
        return "no nodes"
    q = quantiles(counts)
    isolated = sum(1 for c in counts if c == 0)
    return (f"median {q['median']}, p90 {q['p90']}, max {q['max']}"
            + (f", {isolated} with none" if isolated else ""))


#: How far outside the BSP's own world bounds a hand-authored node may sit
#: before the pair is called mismatched, and what share of them may.
#:
#: A nav file built for the map it is named after has **none**: measured across
#: all 174 shipped files against the BSP of the same name, exactly one pair
#: fails, and it fails hugely. ``mgu1m3``'s nav has 126 of its 263 nodes
#: outside the BSP's world -- its nodes reach x 1535 and y 3188 where the map
#: itself stops at 160 and 2864 -- so Nightdive shipped a nav for a different
#: revision of that map. Counting it as a coverage failure blames the generator
#: for something it cannot do anything about; it alone is 191 of the corpus's
#: 2227 missed nodes.
ORACLE_TOLERANCE = 64.0
ORACLE_MISMATCH_SHARE = 0.01


def oracle_mismatch(b, idnav, tolerance=ORACLE_TOLERANCE):
    """How many of `idnav`'s nodes fall outside the BSP's own world bounds."""
    if not b.models or not idnav.nodes:
        return 0
    world = b.models[0]
    return sum(1 for n in idnav.nodes
               if any(n.origin[i] < world.mins[i] - tolerance
                      or n.origin[i] > world.maxs[i] + tolerance
                      for i in range(3)))


def connectivity(nav):
    """``(weak components, largest strongly connected component)``.

    The measure that matters for a map with no oracle, and the one an AAS-level
    score gets wrong: a variant can add areas and reachabilities and still
    produce a *less* connected nav graph. Links are directional, so both
    numbers are needed -- the weak count says how many islands the map broke
    into, and the largest strongly connected component says how much of it a
    bot can actually round-trip.
    """
    n = len(nav.nodes)
    if not n:
        return 0, 0
    out = [[] for _ in range(n)]
    into = [[] for _ in range(n)]
    for i, node in enumerate(nav.nodes):
        for li in range(node.first_link, node.first_link + node.num_links):
            target = nav.links[li].target
            if target < n:
                out[i].append(target)
                into[target].append(i)

    both = [out[i] + into[i] for i in range(n)]
    seen = set()
    weak = 0
    for start in range(n):
        if start in seen:
            continue
        weak += 1
        stack = [start]
        seen.add(start)
        while stack:
            v = stack.pop()
            for w in both[v]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)

    # Kosaraju: one forward pass for the finish order, one reverse pass for the
    # components. Iterative, because a generated graph runs to tens of
    # thousands of nodes.
    order, visited = [], [False] * n
    for start in range(n):
        if visited[start]:
            continue
        stack = [(start, 0)]
        visited[start] = True
        while stack:
            v, k = stack.pop()
            if k < len(out[v]):
                stack.append((v, k + 1))
                w = out[v][k]
                if not visited[w]:
                    visited[w] = True
                    stack.append((w, 0))
            else:
                order.append(v)
    assigned = [False] * n
    largest = 0
    for v in reversed(order):
        if assigned[v]:
            continue
        stack, size = [v], 0
        assigned[v] = True
        while stack:
            x = stack.pop()
            size += 1
            for y in into[x]:
                if not assigned[y]:
                    assigned[y] = True
                    stack.append(y)
        largest = max(largest, size)
    return weak, largest


def report_generated(name, gen, stats, spawns, spacing, verbose=False):
    """Print what can be said about a map with no hand-authored nav file."""
    print(f"\n=== {name}   (no hand-authored nav file to compare against)")
    print(f"  generated      {len(gen.nodes)} nodes, {len(gen.links)} links, "
          f"{len(gen.traversals)} traversals, {len(gen.edicts)} nav edicts")
    if spawns:
        print(f"  spawn coverage {summarise(spawns, (convert.DEFAULT_RADIUS, spacing))}")
    print(f"  link degree    {degree_stats(gen)}")
    weak, largest = connectivity(gen)
    print(f"  connectivity   {weak} island(s); largest mutually reachable "
          f"component {largest}/{len(gen.nodes)}"
          + (f" ({largest * 100.0 / len(gen.nodes):.1f}%)" if gen.nodes else ""))
    print("  link types     " + ", ".join(
        f"{t}: {n}" for t, n in
        collections.Counter(l.type_name for l in gen.links).most_common()))
    problems = validate.check(gen)
    if problems:
        print(f"  INVARIANTS     {len(problems)} violation(s):")
        for p in problems[:8]:
            print(f"    {p}")
    else:
        print("  invariants     clean")
    return {
        "map": name, "id_nodes": 0, "gen_nodes": len(gen.nodes),
        "covered": 0, "problems": len(problems), "missing_types": [],
        "id_teleports": 0, "gen_teleports": sum(
            1 for l in gen.links if l.type == nav3.LinkType.TELEPORT),
        "synthesised": stats.nodes_synthesised if stats else 0,
        "spawns": spawns,
        "id_spawns": [],
        "mismatched": 0,
    }


def compare(name, idnav, gen, stats, spawns, id_spawns, spacing, verbose=False,
            mismatched=0):
    """Print the comparison for one map. Returns a dict of headline numbers."""
    print(f"\n=== {name}")
    if mismatched:
        print(f"  MISMATCHED     {mismatched} of {len(idnav.nodes)} hand-authored "
              f"nodes sit outside this BSP's own world bounds, so the nav file "
              f"was built for a different revision of the map. Coverage is "
              f"reported but left out of the totals.")
    print(f"  {'':<14} {'Nightdive':>10} {'generated':>10}")
    for label, a_, b_ in (("nodes", len(idnav.nodes), len(gen.nodes)),
                          ("links", len(idnav.links), len(gen.links)),
                          ("traversals", len(idnav.traversals), len(gen.traversals)),
                          ("nav edicts", len(idnav.edicts), len(gen.edicts))):
        ratio = f"  ({b_ / a_:.2f}x)" if a_ else ""
        print(f"  {label:<14} {a_:>10} {b_:>10}{ratio}")

    to_gen = nearest_distances(idnav.nodes, gen.nodes)
    to_id = nearest_distances(gen.nodes, idnav.nodes)
    print(f"  coverage       Nightdive node -> nearest generated:  "
          f"{summarise(to_gen, (convert.DEFAULT_RADIUS, spacing))}")
    print(f"  density        generated -> nearest Nightdive:       "
          f"{summarise(to_id, (convert.DEFAULT_RADIUS, spacing))}")
    if spawns:
        print(f"  spawn coverage spawn -> nearest generated:           "
              f"{summarise(spawns, (convert.DEFAULT_RADIUS, spacing))}")
        print(f"  {'':<14} spawn -> nearest Nightdive:           "
              f"{summarise(id_spawns, (convert.DEFAULT_RADIUS, spacing))}")
    print(f"  link degree    Nightdive: {degree_stats(idnav)}")
    print(f"  {'':<14} generated: {degree_stats(gen)}")
    for label, nv in (("Nightdive", idnav), ("generated", gen)):
        weak, largest = connectivity(nv)
        print(f"  connectivity   {label}: {weak} island(s); largest mutually "
              f"reachable component {largest}/{len(nv.nodes)}")

    id_types = collections.Counter(l.type_name for l in idnav.links)
    gen_types = collections.Counter(l.type_name for l in gen.links)
    print("  link types")
    for t in sorted(set(id_types) | set(gen_types),
                    key=lambda t: -(id_types[t] + gen_types[t])):
        mark = ""
        if id_types[t] and not gen_types[t]:
            mark = "   <- Nightdive has these, generator does not"
        elif gen_types[t] and not id_types[t]:
            mark = "   <- generator only"
        print(f"    {t:<22} {id_types[t]:>7} {gen_types[t]:>7}{mark}")

    problems = validate.check(gen)
    if problems:
        print(f"  INVARIANTS     {len(problems)} violation(s):")
        for p in problems[:8]:
            print(f"    {p}")
    else:
        print("  invariants     clean (the same checks all 174 shipped files pass)")

    if verbose and stats is not None:
        print(f"  conversion     {stats.areas_used} usable of {stats.areas} AAS "
              f"areas -> {stats.nodes_before_prune} nodes, "
              f"{stats.isolated_pruned} pruned isolated")
        if stats.dropped_travel:
            print(f"                 dropped travel types: "
                  f"{dict(stats.dropped_travel)}")
        if stats.dropped_endpoint or stats.dropped_duplicate:
            print(f"                 dropped {stats.dropped_endpoint} for an "
                  f"unusable endpoint, {stats.dropped_duplicate} duplicate "
                  f"node pairs")
        if stats.edicts_unresolved:
            print(f"                 {stats.edicts_unresolved} mover link(s) "
                  f"with no resolvable brush model")
        if stats.teleports or stats.teleporters_unresolved:
            print(f"                 {stats.teleports} teleport link(s), "
                  f"{stats.nodes_synthesised} node(s) synthesised "
                  f"outside AAS space")
            if stats.teleporters_unresolved:
                print(f"                 unresolved teleporters: "
                      f"{dict(stats.teleporters_unresolved)}")
        if stats.pushes or stats.pushers_unresolved:
            print(f"                 {stats.pushes} push link(s)")
            if stats.pushers_unresolved:
                print(f"                 unresolved push triggers: "
                      f"{dict(stats.pushers_unresolved)}")

    covered = sum(1 for d in to_gen if d <= spacing)
    return {
        "map": name,
        "id_nodes": len(idnav.nodes),
        "gen_nodes": len(gen.nodes),
        "covered": covered,
        "problems": len(problems),
        "missing_types": [t for t in id_types if not gen_types[t]],
        "id_teleports": id_types["TELEPORT"],
        "gen_teleports": gen_types["TELEPORT"],
        "synthesised": stats.nodes_synthesised if stats else 0,
        "coverage": to_gen,
        "spawns": spawns,
        "id_spawns": id_spawns,
        "mismatched": mismatched,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kexnav.py check", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("maps", nargs="*", help="map names, as they appear in the pak")
    ap.add_argument("--all", action="store_true",
                    help="every map in the pak. Maps with no hand-authored nav "
                         "file get the generated-only report")
    ap.add_argument("--test", action="store_true",
                    help="Nightdive's four isolated-feature test maps")
    ap.add_argument("--pak", action="append", metavar="PATH",
                    help="pak file, or a directory of them, to read both sides "
                         "from. Repeatable; later paks win, as in "
                         "kexnav.py generate")
    ap.add_argument("--seed-flood", action="store_true",
                    help="pass --seed-flood to the compile step; see "
                         "kexnav.py generate")
    ap.add_argument("--train-links", action="store_true",
                    help="pass --train-links to the compile step; see kexnav.py generate")
    ap.add_argument("--no-lift-links", dest="lift_links", action="store_false",
                    help="pass --no-lift-links to the compile step; see "
                         "kexnav.py generate")
    ap.add_argument("--door-movers", action="store_true",
                    help="pass --door-movers to the compile step; see "
                         "kexnav.py generate")
    ap.add_argument("--only-oracle", action="store_true",
                    help="skip maps that have no hand-authored nav file")
    ap.add_argument("--bspc", help="BSPC binary")
    ap.add_argument("--work", help="directory for cached .bsp/.aas intermediates")
    ap.add_argument("--spacing", type=float, default=convert.DEFAULT_SPACING)
    ap.add_argument("--min-separation", type=float,
                    default=convert.MIN_NODE_SEPARATION)
    ap.add_argument("--limit", type=int, help="stop after this many maps")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    bspc = env.find_bspc(args.bspc)
    if not bspc:
        ap.error("no BSPC binary found. " + env.bspc_hint())
    workdir = args.work or os.path.join("out", "cache")
    os.makedirs(workdir, exist_ok=True)

    paks = _open_paks(args.pak)
    have_nav = {n[len(NAV_PREFIX):-4] for pak in paks for n in pak.names
                if n.startswith(NAV_PREFIX) and n.endswith(".nav")}

    if args.test:
        wanted = list(TEST_MAPS)
    elif args.all:
        # every map in the paks, not only the ones with a nav file -- the
        # generated-only report is how a file with no reference gets checked
        wanted = sorted(n[len("maps/"):-4] for n in _names(paks)
                        if n.startswith("maps/") and n.endswith(".bsp"))
    else:
        wanted = []
        for m in args.maps:
            wanted.append(m if m in have_nav else
                          next((h for h in have_nav
                                if os.path.basename(h) == m), m))
    if args.only_oracle:
        wanted = [m for m in wanted if m in have_nav]
    if args.limit:
        wanted = wanted[:args.limit]
    if not wanted:
        ap.error("name at least one map, or pass --all or --test")

    rows = []
    skipped = []
    for m in wanted:
        # the pak-relative name, subdirectory and all -- six basenames in the
        # retail pak belong to two different maps (command / q64/command,
        # ware1 / old/ware1, lab, fact1, fact2, fact3), so keying the cache on
        # the basename made one of each pair overwrite the other's .aas
        base = m
        try:
            bsp_bytes = _read(paks, f"maps/{m}.bsp")
        except KeyError:
            skipped.append((base, "no matching BSP in the pak"))
            continue

        bsp_path = os.path.join(workdir, base + ".bsp")
        aas_path = os.path.join(workdir, base + ".aas")
        os.makedirs(os.path.dirname(bsp_path) or ".", exist_ok=True)
        if not os.path.isfile(bsp_path):
            with open(bsp_path, "wb") as fp:
                fp.write(bsp.strip_for_bspc(bsp_bytes))
        try:
            if not os.path.isfile(aas_path):
                produced = compile_best_aas(bspc, base, bsp_bytes, workdir,
                                            seed_flood=args.seed_flood,
                                            train_links=args.train_links,
                                            lift_links=args.lift_links,
                                            door_movers=args.door_movers)
                # compile_best_aas already places it at <workdir>/<base>.aas
                if os.path.abspath(produced) != os.path.abspath(aas_path):
                    shutil.copyfile(produced, aas_path)
            a = aas.load(aas_path)
            # the BSP BSPC actually compiled, not the one from the pak: a
            # mover variant carries duplicate brush models and its
            # reachabilities name them, so reading the original would leave
            # every edict on those links unresolved. compile_best_aas writes
            # the winner next to the .aas for exactly this reason.
            b = bsp.load(bsp_path) if os.path.isfile(bsp_path) \
                else bsp.loads(bsp_bytes)
            stats = convert.Stats()
            gen = convert.convert(a, b, spacing=args.spacing,
                                  min_separation=args.min_separation,
                                  stats=stats)
            idnav = (nav3.loads(_read(paks, f"{NAV_PREFIX}{m}.nav"))
                     if m in have_nav else None)
        except (RuntimeError, aas.AasError, bsp.BspError, nav3.NavError,
                convert.ConvertError, OSError) as exc:
            skipped.append((base, str(exc)))
            continue
        spawns = spawn_distances(b, gen)
        id_spawns = spawn_distances(b, idnav) if idnav is not None else []
        if idnav is None:
            rows.append(report_generated(m, gen, stats, spawns, args.spacing,
                                         args.verbose))
        else:
            mismatched = oracle_mismatch(b, idnav)
            if mismatched <= len(idnav.nodes) * ORACLE_MISMATCH_SHARE:
                mismatched = 0
            rows.append(compare(m, idnav, gen, stats, spawns, id_spawns,
                                args.spacing, args.verbose, mismatched))
    for pak in paks:
        pak.close()

    print(f"\n{'=' * 72}")
    print(f"compared   : {len(rows)} map(s)")
    if skipped:
        print(f"skipped    : {len(skipped)}")
        for name, why in skipped[:8]:
            print(f"    {name}: {why}")
        if len(skipped) > 8:
            print(f"    ... and {len(skipped) - 8} more")
    if rows:
        # a mismatched pair measures the wrong thing, so it stays out of the
        # totals -- see ORACLE_TOLERANCE
        graded = [r for r in rows if not r["mismatched"]]
        id_total = sum(r["id_nodes"] for r in graded)
        # only the maps that have an oracle: summing every generated node
        # against only the hand-authored ones would count the maps Nightdive
        # never authored on one side of the ratio and not the other
        gen_total = sum(r["gen_nodes"] for r in graded if r["id_nodes"])
        covered = sum(r["covered"] for r in graded)
        mismatched = [r for r in rows if r["mismatched"]]
        if mismatched:
            print("mismatched : " + ", ".join(
                f"{r['map']} ({r['mismatched']} of {r['id_nodes']} Nightdive "
                f"nodes outside the BSP)" for r in mismatched)
                + " -- nav file built for a different revision, left out of "
                  "the coverage total")
        if id_total:
            print(f"coverage   : {covered}/{id_total} Nightdive nodes "
                  f"({covered * 100.0 / id_total:.1f}%) have a generated node "
                  f"within {args.spacing:g} units")
        all_spawns = [d for r in rows for d in r["spawns"]]
        id_all = [d for r in rows for d in r["id_spawns"]]
        if all_spawns:
            near = sum(1 for d in all_spawns if d <= args.spacing)
            line = (f"spawns     : {near}/{len(all_spawns)} player spawn points "
                    f"({near * 100.0 / len(all_spawns):.1f}%) have a generated "
                    f"node within {args.spacing:g} units")
            if id_all:
                id_near = sum(1 for d in id_all if d <= args.spacing)
                line += (f"; Nightdive's own files manage "
                         f"{id_near * 100.0 / len(id_all):.1f}%")
            print(line)
        if id_total:
            print(f"density    : {gen_total} generated nodes against {id_total} "
                  f"hand-authored ({gen_total / id_total:.2f}x)")
        id_tel = sum(r["id_teleports"] for r in rows)
        gen_tel = sum(r["gen_teleports"] for r in rows)
        if id_tel or gen_tel:
            print(f"teleports  : {gen_tel} generated against {id_tel} "
                  f"hand-authored, {sum(r['synthesised'] for r in rows)} "
                  f"node(s) synthesised outside AAS space")
        worst = [r for r in sorted((r for r in graded if r["id_nodes"]),
                                   key=lambda r: -(r["id_nodes"] - r["covered"]))
                 if r["id_nodes"] - r["covered"]][:5]
        if worst:
            print("worst coverage (Nightdive nodes with no generated node "
                  f"within {args.spacing:g} units)")
            for r in worst:
                print(f"    {r['map']:<24} "
                      f"{r['id_nodes'] - r['covered']:>4} of "
                      f"{r['id_nodes']:>4} missed")
        # Only worth flagging where the generated file covers *fewer* spawns
        # than Nightdive's does. Several q64 maps carry spawns Nightdive's own
        # nav misses too, and listing those as generator failures would be
        # wrong.
        def covered(distances):
            return sum(1 for d in distances if d <= args.spacing)

        thin = [r for r in rows if r["spawns"] and r["id_spawns"]
                and covered(r["spawns"]) < covered(r["id_spawns"])]
        thin.sort(key=lambda r: covered(r["spawns"]) - covered(r["id_spawns"]))
        if thin:
            print("spawns the generated file misses that Nightdive's does not")
            for r in thin[:8]:
                print(f"    {r['map']:<24} {covered(r['spawns']):>4} covered "
                      f"against Nightdive's {covered(r['id_spawns']):>4}, "
                      f"of {len(r['spawns'])}")
        else:
            print("spawns     : no map where the generated file covers fewer "
                  "spawns than Nightdive's own")
        broken = [r for r in rows if r["problems"]]
        print(f"invariants : {len(rows) - len(broken)}/{len(rows)} clean"
              + (f", {len(broken)} FAILED" if broken else ""))
        missing = collections.Counter()
        for r in rows:
            missing.update(r["missing_types"])
        if missing:
            print("types Nightdive has that the generator did not emit, by "
                  "map count:")
            for t, n in missing.most_common():
                print(f"    {t:<22} {n} map(s)")

    print("\nknown gaps (kexnav.convert.GAPS)")
    for field, note in convert.GAPS:
        print(f"  {field}")
        print(f"      {note}")

    return 1 if any(r["problems"] for r in rows) or skipped else 0

"""Generate a NAV3 nav file for a map.

The whole pipeline, in one command:

    BSP  ->  bspc -bsp2aas  ->  .aas  ->  kexnav.convert  ->  .nav

BSPs come either from a pak or from the filesystem. The intermediate ``.aas``
is cached next to the output, because compiling one costs seconds while
converting it costs milliseconds -- so iterating on the converter does not
mean recompiling.

    # a stock map, straight out of the retail pak
    python3 kexnav.py generate q2dm1 -o out/

    # every map in a gamedir, from its own paks
    python3 kexnav.py generate --pak /path/to/gamedir/pak1.pak --all -o out/

    # a loose BSP
    python3 kexnav.py generate path/to/mymap.bsp -o out/

Drop the result at ``<gamedir>/bots/navigation/<mapname>.nav``. With ``--deploy
DIR`` this writes it there directly.
"""

import argparse
import collections
import os
import shutil
import subprocess
import tempfile

from kexnav import aas, bsp, convert, env, nav3, validate
from kexnav.pak import Pak

BSP_PREFIX = "maps/"


#: Extra switches to retry a failed compile with, in order.
#:
#: ``-nocsg`` disables BSPC's brush chopping. Some maps only compile without
#: it: a long-shipped, playable map can still abort with *"WARNING: entity
#: reached from outside / ERROR: **** leaked ****"* on a plain compile and
#: succeed with ``-nocsg``. The leak is an artefact of the chopping rather
#: than a genuine hole in the map.
#:
#: It is a fallback and not the default because it costs coverage: on maps
#: that compile cleanly either way, the plain pass measures about 10% more
#: grounded areas and reachabilities than ``-nocsg`` does.
BSPC_RETRIES = (("-nocsg",),)


def _run_bspc(bspc, directory, args, verbose):
    proc = subprocess.run([bspc] + list(args), cwd=directory,
                          capture_output=True, text=True)
    if verbose:
        for line in proc.stdout.splitlines():
            if line.startswith(("numareas", "reachabilitysize", "ERROR", "WARNING")):
                print(f"      bspc: {line}")
    return proc


def compile_aas(bspc, bsp_path, verbose=False):
    """Run ``bspc -bsp2aas`` and return the path of the ``.aas`` it wrote.

    Three BSPC quirks are handled here, all recorded in the wiki's
    GENERATING page:

    * The filename argument goes into a fixed-size buffer, so it is invoked
      from the file's own directory with a bare basename. A long absolute path
      smashes the stack before BSPC prints anything.
    * The 64-bit build **aborts at exit** with a stack-protector failure
      *after* writing a complete, byte-identical file. The return code is
      therefore not evidence of anything; whether the ``.aas`` appeared is.
    * Some maps only compile with :data:`BSPC_RETRIES`.
    """
    directory = os.path.dirname(os.path.abspath(bsp_path))
    base = os.path.basename(bsp_path)
    out = os.path.splitext(os.path.abspath(bsp_path))[0] + ".aas"
    log = os.path.join(directory, "bspc.log")

    attempts = [()] + [tuple(r) for r in BSPC_RETRIES]
    for i, extra in enumerate(attempts):
        proc = _run_bspc(bspc, directory, ["-bsp2aas", base] + list(extra), verbose)
        if os.path.isfile(out):
            if extra and verbose:
                print(f"      bspc: succeeded with {' '.join(extra)}")
            if os.path.isfile(log):
                os.remove(log)
            return out
        if verbose and i + 1 < len(attempts):
            print(f"      bspc: retrying with {' '.join(attempts[i + 1])}")

    if os.path.isfile(log):
        os.remove(log)
    detail = next((l.strip() for l in proc.stdout.splitlines() if "ERROR" in l), "")
    tried = ", ".join(" ".join(e) or "no switches" for e in attempts)
    raise RuntimeError(f"bspc wrote no .aas for {base}"
                       + (f": {detail}" if detail else f" (exit {proc.returncode})")
                       + f" -- tried {tried}")


def resolve_maps(args):
    """Yield (mapname, bsp bytes or path) for everything the arguments name.

    ``--pak`` may be repeated, and a directory is expanded to the ``pak*.pak``
    inside it: some gamedirs split their maps across multiple paks, so naming
    the directory has to be enough.
    """
    loose = [m for m in args.maps if m.lower().endswith(".bsp")]
    named = [m for m in args.maps if not m.lower().endswith(".bsp")]

    for path in loose:
        yield os.path.splitext(os.path.basename(path))[0], path

    if not named and not args.all:
        return

    pak_paths = []
    for entry in (args.pak or [env.find_pak()]):
        if os.path.isdir(entry):
            pak_paths += sorted(os.path.join(entry, f) for f in os.listdir(entry)
                                if f.lower().endswith(".pak"))
        else:
            pak_paths.append(entry)
    if not pak_paths:
        print("  SKIP  no pak files found")
        return

    # later paks win, the way the engine's search order works
    available = {}
    for pak_path in pak_paths:
        with Pak(pak_path) as pak:
            for n in pak.names:
                if n.startswith(BSP_PREFIX) and n.endswith(".bsp"):
                    available[n[len(BSP_PREFIX):-4]] = (pak_path, n)

    wanted = sorted(available) if args.all else named
    # resolve every name first, then read grouped by pak -- reopening a 1.7 GB
    # archive per map means re-reading its whole directory 222 times
    resolved = []
    for name in wanted:
        # a pak path may be nested, e.g. maps/test/mals_box.bsp
        key = name if name in available else next(
            (k for k in available if os.path.basename(k) == name), None)
        if key is None:
            print(f"  SKIP  {name}: no maps/{name}.bsp in "
                  + ", ".join(os.path.basename(p) for p in pak_paths))
            continue
        resolved.append((available[key][0], available[key][1], key))
    for pak_path in dict.fromkeys(p for p, _, _ in resolved):
        with Pak(pak_path) as pak:
            for owner, entry, key in resolved:
                if owner == pak_path:
                    # the pak-relative name, subdirectory and all: the engine
                    # loads bots/navigation/<mapname>.nav, and `mapname` for
                    # maps/q64/command.bsp is "q64/command". 33 of Nightdive's
                    # own 174 nav files sit in a subdirectory for exactly this
                    # reason, and six basenames are shared by two different
                    # maps.
                    yield key, pak.read(entry)


def bsp_variants(raw, seed_flood=False, train_links=False, lift_links=True,
                 door_movers=False):
    """The versions of a BSP worth handing BSPC, as two lists.

    Each one is a rewrite that gets BSPC to see something it otherwise would
    not; all of them also strip the lumps it cannot load. ``plain`` is always
    included so there is a floor to compare against, because every rewrite
    can also make BSPC refuse the map outright.
    """
    movers, count = bsp.add_movers(raw, trains=train_links, lifts=lift_links)
    base = movers if count else bsp.strip_for_bspc(raw)
    preferred = ([(f"movers({count})", movers)] if count else [])
    preferred.append(("plain", bsp.strip_for_bspc(raw)))

    compared = []
    if door_movers:
        # renamed *after* the mover rewrites, so plat_lift_entities never sees
        # a rotating door as a vertical func_door -- its `angle` is a rotation
        # axis, not a move direction
        doors, renamed = bsp.mark_doors_as_movers(base)
        if renamed:
            compared.append((f"doors({renamed})", doors))
    if seed_flood:
        seeded, seeds = bsp.add_flood_seeds(base)
        if seeds:
            compared.append((f"seeded({seeds})", seeded))
    return preferred, compared


def _scratch_name(name):
    """A map name flattened for use in a filename component."""
    return name.replace("/", "_").replace("\\", "_")


def compile_best_aas(bspc, name, raw, workdir, verbose=False, seed_flood=True,
                     train_links=False, lift_links=True, door_movers=False):
    """Compile the BSP variants worth trying and keep the best AAS.

    Scored on grounded areas first, then reachability count, so the result is
    **never worse than the plain compile** -- which matters because both
    rewrites can make BSPC fail: flood seeds can seed from the void, and so
    can a func_bobbing's origin. A map with nothing to translate costs one
    compile; one with a train costs two.

    See :func:`kexnav.bsp.add_train_bobbing` and
    :func:`kexnav.bsp.add_flood_seeds` for what each variant buys.
    """
    preferred, compared = bsp_variants(
        raw, seed_flood=seed_flood, train_links=train_links,
        lift_links=lift_links, door_movers=door_movers)
    scratch = tempfile.mkdtemp(prefix=f"{_scratch_name(name)}-", dir=workdir)
    try:
        best = None
        attempted = []
        index = 0

        def attempt(label, data):
            """Compile one variant and fold it into `best`."""
            nonlocal best, index
            attempted.append(label)
            path = os.path.join(scratch, f"v{index}.bsp")
            index += 1
            with open(path, "wb") as fp:
                fp.write(data)
            try:
                out = compile_aas(bspc, path, verbose=False)
            except RuntimeError as exc:
                if verbose:
                    print(f"      {label}: {str(exc)[:70]}")
                return False
            a = aas.load(out)
            score = (len(a.grounded_areas()), len(a.reachability))
            if verbose:
                print(f"      {label}: {score[0]} grounded areas, "
                      f"{score[1]} reachabilities")
            if best is None or score > best[0]:
                best = (score, out, label)
            return True

        # the first preferred variant that compiles is the baseline: they
        # cannot shrink the area graph, so there is nothing to compare
        for label, data in preferred:
            if attempt(label, data):
                break
        # everything in `compared` can shrink it, so all of them get compiled
        # and scored against that baseline
        for label, data in compared:
            attempt(label, data)

        if best is None:
            raise RuntimeError(f"bspc wrote no .aas for {name} -- tried "
                               + ", ".join(attempted))
        if verbose and len(attempted) > 1:
            print(f"      keeping the {best[2]} compile")
        # Move the winner out before the scratch directory goes away, and keep
        # the BSP it came from: a train variant carries extra brush models and
        # its reachabilities name them, so the converter has to read the same
        # file or the indices will not line up.
        kept = os.path.join(workdir, name + ".aas")
        os.makedirs(os.path.dirname(kept) or ".", exist_ok=True)
        shutil.move(best[1], kept)
        with open(os.path.join(workdir, name + ".bsp"), "wb") as fp:
            fp.write(dict(preferred + compared)[best[2]])
        return kept
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def generate(name, source, args, bspc, workdir):
    """Compile and convert one map. Returns (NavFile, Stats)."""
    if isinstance(source, str):
        with open(source, "rb") as fp:
            source = fp.read()

    cached = os.path.join(workdir, name + ".aas")
    bsp_path = os.path.join(workdir, name + ".bsp")
    os.makedirs(os.path.dirname(cached) or ".", exist_ok=True)
    if os.path.isfile(cached) and os.path.isfile(bsp_path) and not args.recompile:
        aas_path = cached
    else:
        # compile_best_aas also writes the winning variant's BSP next to the
        # .aas, because that is the file whose model indices its
        # reachabilities refer to
        aas_path = compile_best_aas(bspc, name, source, workdir,
                                    verbose=args.verbose,
                                    seed_flood=args.seed_flood,
                                    train_links=args.train_links,
                                    lift_links=args.lift_links,
                                    door_movers=args.door_movers)
        if os.path.abspath(aas_path) != os.path.abspath(cached):
            shutil.copyfile(aas_path, cached)
            aas_path = cached

    a = aas.load(aas_path)
    b = bsp.load(bsp_path)
    stats = convert.Stats()
    nav = convert.convert(a, b, heuristic=args.heuristic, version=args.nav_version,
                          spacing=args.spacing,
                          min_separation=args.min_separation, stats=stats)
    return nav, stats


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kexnav.py generate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("maps", nargs="*", help="map names to read from a pak, "
                                            "or paths to .bsp files")
    ap.add_argument("--pak", action="append", metavar="PATH",
                    help="pak file, or a directory of them, to take named maps "
                         "from. Repeatable; later paks win. Default: the retail "
                         "rerelease pak0.pak")
    ap.add_argument("--all", action="store_true", help="every map in the pak")
    ap.add_argument("-o", "--out", default="out", help="where to write .nav files")
    ap.add_argument("--deploy", metavar="GAMEDIR",
                    help="also install to GAMEDIR/bots/navigation/<map>.nav")
    ap.add_argument("--work", help="directory for cached .bsp/.aas intermediates "
                                   "(default: OUT/cache)")
    ap.add_argument("--bspc", help="BSPC binary (default: kexnav.env.find_bspc)")
    ap.add_argument("--recompile", action="store_true",
                    help="re-run bspc even if a cached .aas exists")
    ap.add_argument("--door-movers", action="store_true",
                    help="also present func_door_rotating and func_door_secret "
                         "to BSPC as func_door, so their volume stops being a "
                         "wall in the AAS. Off by default: it opens real "
                         "passages (xsewer1 gains 162 grounded areas and 1049 "
                         "reachabilities) yet the nav graph that comes out is "
                         "no better connected -- over seven maps the component "
                         "count went 60 to 64 and the share of Nightdive node "
                         "pairs mutually reachable 53.2%% to 52.5%%. See "
                         "kexnav.bsp.mark_doors_as_movers")
    ap.add_argument("--no-lift-links", dest="lift_links", action="store_false",
                    help="do not describe func_plat2 and vertical func_door "
                         "movers to BSPC as func_plat. On by default: BSPC "
                         "matches only the exact classname func_plat, and "
                         "adding the other two raises agreement with "
                         "Nightdive's own ride links from 12 of 216 to 136 "
                         "and recovers 69 of "
                         "136 otherwise-unreachable regions. See "
                         "kexnav.bsp.plat_lift_entities")
    ap.add_argument("--train-links", action="store_true",
                    help="also try a variant that describes each func_train "
                         "to BSPC as a func_bobbing, so it computes TRAIN ride "
                         "links. Off by default: measured against Nightdive's "
                         "files it reproduces 13 of their 54 TRAIN links and "
                         "recovers 6 "
                         "of 35 otherwise-unreachable regions, while emitting "
                         "334 links. See kexnav.bsp.add_train_bobbing")
    ap.add_argument("--seed-flood", action="store_true",
                    help="also try a variant with extra flood seeds at the "
                         "BSP's empty leaf centres. Recovers regions BSPC "
                         "discards on maps built from disjoint volumes, at the "
                         "cost of another compile per map. See "
                         "kexnav.bsp.add_flood_seeds")
    ap.add_argument("--spacing", type=float, default=convert.DEFAULT_SPACING,
                    help=f"node spacing inside an area, in units "
                         f"(default {convert.DEFAULT_SPACING:g})")
    ap.add_argument("--min-separation", type=float,
                    default=convert.MIN_NODE_SEPARATION,
                    help=f"drop a node this close to an existing one "
                         f"(default {convert.MIN_NODE_SEPARATION:g})")
    ap.add_argument("--heuristic", type=float, default=convert.DEFAULT_HEURISTIC,
                    help=f"A* weight in the header (default "
                         f"{convert.DEFAULT_HEURISTIC})")
    ap.add_argument("--nav-version", type=int, default=nav3.VERSION,
                    choices=nav3.SUPPORTED_VERSIONS,
                    help=f"nav file version to write (default {nav3.VERSION})")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    if not args.maps and not args.all:
        ap.error("name at least one map, or pass --all")

    bspc = env.find_bspc(args.bspc)
    if not bspc:
        ap.error("no BSPC binary found. " + env.bspc_hint())

    workdir = args.work or os.path.join(args.out, "cache")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(workdir, exist_ok=True)

    written = failed = 0
    totals = collections.Counter()
    types = collections.Counter()
    for name, source in resolve_maps(args):
        try:
            nav, stats = generate(name, source, args, bspc, workdir)
        except (RuntimeError, aas.AasError, bsp.BspError,
                convert.ConvertError, OSError) as exc:
            print(f"  FAIL  {name}: {exc}")
            failed += 1
            continue

        problems = validate.check(nav)
        if problems:
            print(f"  FAIL  {name}: generated file breaks "
                  f"{len(problems)} invariant(s):")
            for p in problems[:5]:
                print(f"        {p}")
            failed += 1
            continue

        data = nav3.dumps(nav)
        if nav3.dumps(nav3.loads(data)) != data:
            print(f"  FAIL  {name}: does not survive its own round trip")
            failed += 1
            continue

        path = os.path.join(args.out, name + ".nav")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as fp:
            fp.write(data)
        if args.deploy:
            target = os.path.join(args.deploy, "bots", "navigation", name + ".nav")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copyfile(path, target)

        written += 1
        for k in ("nodes", "links", "traversals", "edicts"):
            totals[k] += getattr(stats, k)
        types.update(stats.link_types)
        print(f"  ok    {name:<24} {stats.nodes:>5} nodes {stats.links:>6} links "
              f"{stats.traversals:>5} traversals {stats.edicts:>3} edicts  "
              f"{len(data):>8} bytes")
        if args.verbose:
            print(f"        from {stats.areas_used} usable of {stats.areas} AAS "
                  f"areas, {stats.reachabilities} reachabilities; "
                  f"pruned {stats.isolated_pruned} isolated")
            if stats.dropped_travel:
                print(f"        dropped travel types: {dict(stats.dropped_travel)}")
            print(f"        {dict(stats.link_types)}")

    print()
    print(f"written    : {written} nav file(s) to {args.out}/"
          + (f", {failed} FAILED" if failed else ""))
    if written:
        print(f"totals     : {totals['nodes']} nodes, {totals['links']} links, "
              f"{totals['traversals']} traversals, {totals['edicts']} nav edicts")
        print(f"link types : " + ", ".join(f"{k}: {v}" for k, v in types.most_common()))
    if args.deploy:
        print(f"deployed   : {args.deploy}/bots/navigation/")
    return 1 if failed else 0

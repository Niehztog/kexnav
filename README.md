# kexnav

Generates **NAV3** bot navigation files for Quake II (the KEX rerelease engine).

The engine's closed-source bot AI loads one `.nav` file per map from `bots/navigation/<mapname>.nav`, and a map without one gets no bot navigation at all. Nightdive Studios shipped nav files for the stock maps only, hand-authored a node at a time in the engine's built-in nav editor. `kexnav.py generate` builds one for any Quake II BSP instead, by compiling the map's AAS and converting it.

## How it works

```
BSP  -->  bspc -bsp2aas  -->  .aas  -->  kexnav.convert  -->  .nav
```

BSPC - Quake III's bot path compiler - decomposes a map into convex areas and runs a movement simulation over them to work out which pairs a player can get between, and how: walk, jump, ladder, lift, barrier climb. That classification is what a nav file needs and what floor sampling cannot produce, which is why the pipeline goes through AAS rather than tracing the map directly. Everything after the `.aas` is plain Python.

BSPC is used **unmodified**, so what it leaves open is Quake III's vocabulary: it matches Quake III's entity classnames, not Quake II's. Two mechanisms close that gap.

* Entities that only need renaming are **rewritten into the BSP** before BSPC sees it. A `func_plat2`, or a vertical `func_door` or `func_water`, is handed over as a `func_plat`, so BSPC's own elevator pass gives its lift ride links.
* Entities that cannot be renamed are **simulated in the converter**: teleporters and `trigger_push`. Both would need a synthetic entity carrying an `origin` key, and that is the one rewrite BSPC cannot take - an `origin` seeds its inside/outside flood, so a synthetic one seeds from the void and the compile leaks.

The reasoning behind every conversion decision, and the measurement behind every constant, is in the wiki's [GENERATING](https://github.com/Niehztog/kexnav/wiki/GENERATING) page.

## Requirements

Python 3, standard library only - no dependencies, no build step. You will also need:

* a **BSPC binary** - 2.1h, built native from [bnoordhuis/bspc](https://github.com/bnoordhuis/bspc);
* a **Quake II install** holding the maps you want to generate for.

`kexnav/env.py` looks where a stock install puts the pak and searches `PATH` for `bspc`, and every command takes an explicit override. To point it somewhere else for good, drop a `kexnav.local` beside `kexnav.py` - one `<what> <path>` pair per line, `~` expanded, repeat a key for a second location. Git ignores it, so your layout stays yours:

```
pak   ~/games/quake2-rerelease/baseq2/pak0.pak
bspc  ~/src/bspc/bspc
```

## Usage

Generate a nav file for a stock map, straight out of the retail pak:

```sh
python3 kexnav.py generate q2dm1 -o out/
```

Generate for a whole gamedir at once:

```sh
python3 kexnav.py generate --pak /path/to/gamedir --all -o out/
```

The engine looks for the result at `<gamedir>/bots/navigation/<mapname>.nav`, where `<mapname>` keeps any subdirectory the BSP had - `maps/q64/command.bsp` needs `bots/navigation/q64/command.nav`. `--deploy` puts it there for you:

```sh
python3 kexnav.py generate q2dm1 --deploy ~/gamedir
```

Then, optionally, check the result:

```sh
python3 kexnav.py check q2dm1
```

`check` compares node coverage, spawn-point coverage and link types against Nightdive's own hand-authored file where one exists. Where none does - which is the case for every map this tool is actually for - it reports spawn coverage and structural validity alone. Run `python3 kexnav.py <command> -h` for every flag a command takes; the four that change what comes out are tabulated in [GENERATING](https://github.com/Niehztog/kexnav/wiki/GENERATING#the-flags-that-change-what-comes-out).

## Results

The retail pak holds 222 maps and 174 hand-authored nav files. BSPC compiles 193 of the maps, 146 of which have a nav file to grade the output against:

```
$ python3 kexnav.py check --all
compared   : 193 map(s), 29 skipped
mismatched : mgu1m3 (126 of 263 Nightdive nodes outside the BSP) -- nav file
             built for a different revision, left out of the coverage total
coverage   : 51878/53914 Nightdive nodes (96.2%) have a generated node within
             128 units
spawns     : 2603/2767 player spawn points (94.1%) have a generated node within
             128 units; Nightdive's own files manage 96.9%
density    : 1.76x
invariants : 193/193 clean
```

Read the two spawn numbers together: 100% is not the target, because Nightdive's own hand-authored files do not reach it either.

Of the Nightdive nodes that are missed, **84% are somewhere BSPC's area graph does not cover at all** - and only about a tenth of those are tight spots beside open space, the rest being regions BSPC discarded outright. Node placement, spacing and merging together account for roughly a tenth of the gap. So the converter's own constants are not the lever here; `--seed-flood` is, and what it does and does not buy is measured in [GENERATING](https://github.com/Niehztog/kexnav/wiki/GENERATING#bspcs-flood-seeding-improved-not-fixed).

## Limitations

* **BSPC can silently lose whole regions of a map.** It decides what is "inside" by flooding from entity origins, so a region whose only entities sit too close to a wall is discarded. `--seed-flood` recovers most of the waypoints but not the routes to them; see [GENERATING](https://github.com/Niehztog/kexnav/wiki/GENERATING#bspcs-flood-seeding-improved-not-fixed).
* **29 of the retail pak's 222 maps do not compile**, 28 of them nav-paired. All are BSPC limits hit by old mission-pack and N64 maps, not by community maps.
* **`func_door_rotating` is baked into the AAS as a permanent wall.** BSPC gives `CONTENTS_MOVER` to the exact classname `func_door` and nothing else, so all 448 rotating doors in the pak seal the doorways they guard. `--door-movers` presents them as `func_door` and does open real passages, but the nav graph that comes out is no better connected, so it is off by default; see [GENERATING](https://github.com/Niehztog/kexnav/wiki/GENERATING#func_door_rotating-a-real-wall-that-should-not-be-and-it-still-does-not-pay).
* **`func_train` ride links are behind `--train-links`, off by default** - the measured cost and benefit are in [GENERATING](https://github.com/Niehztog/kexnav/wiki/GENERATING#func_train-translated-to-func_bobbing-and-off-by-default).
* **`func_door_rotating` and `func_button` ride links are not generated at all.** A rotating door travels in an arc about an origin brush, and neither of BSPC's mover descriptions - straight down, or straight along one axis - covers an arc.
* **`CROUCH` links are never generated**, because BSPC never emits `TRAVEL_CROUCH`: the constant appears in the bot's runtime movement code and nowhere in `be_aas_reach.c`. This costs less than it sounds. Of Nightdive's 156 `CROUCH` links in a 51-map sample, 117 have *both* endpoints outside AAS space entirely, and Nightdive itself uses plain `WALK` for 675 links that do touch a crouch-only area.

## Layout

```
kexnav.py             the single entry point: generate / check

kexnav/nav3.py         the NAV3 format: loads/dumps/load/dump
kexnav/aas.py          BSPC's AAS output -- areas, reachabilities, the BSP tree
kexnav/bsp.py          the two Quake II BSP lumps needed: entities and models
kexnav/convert.py      AAS -> NAV3, and every conversion constant with its evidence
kexnav/validate.py     the structural rules, shared by the gates and the generator
kexnav/pak.py          minimal Quake II PACK reader
kexnav/env.py          where the inputs live -- stock candidates plus kexnav.local
kexnav/cli/            the two commands' implementations

tests/test_kexnav.py   the unit tests
tests/roundtrip.py     dev-only gate: nav3.py against the retail corpus, byte-exact
tests/aascheck.py      dev-only gate: aas.py against bspc -aasinfo's own reading
```

The two gates in `tests/` are not part of the tool - they check the format model and the AAS reader against known-good data rather than touching a user's map:

```sh
python3 -m tests.roundtrip          # 174/174 shipped files re-serialise byte-identical
python3 -m tests.aascheck out/cache # bspc -aasinfo agrees on every lump count
```

## Documentation

To use the tool, this README is enough. Beyond it, in the [wiki](https://github.com/Niehztog/kexnav/wiki):

| page | what it covers | when you need it |
|------|----------------|------------------|
| [GENERATING](https://github.com/Niehztog/kexnav/wiki/GENERATING) | how the converter works and why each decision was made; the AAS -> NAV3 mapping; the known gaps | changing the converter, or judging its output |
| [FORMAT](https://github.com/Niehztog/kexnav/wiki/FORMAT) | the NAV3 file format, field by field, and the evidence for it | writing another reader or writer |
| [BACKGROUND](https://github.com/Niehztog/kexnav/wiki/BACKGROUND) | how bots work in the rerelease, and what is and is not moddable | orientation; not needed to use the tool |
| [DECISION](https://github.com/Niehztog/kexnav/wiki/DECISION) | why this generates nav files rather than replacing the bots | before proposing a different approach |

[CLAUDE.md](CLAUDE.md) is the orientation page for an AI agent session.

## License

MIT - see [LICENSE](LICENSE).

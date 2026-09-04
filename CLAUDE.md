# kexnav: orientation

Read this first, then follow the links you need. Everything a fresh session requires to take over is in this folder or in the repository's [wiki](https://github.com/Niehztog/kexnav/wiki); nothing depends on the conversation that produced it.

## What this project is

Tools for **NAV3**, the bot navigation format of the Quake II rerelease (KEX engine). The engine's closed-source bot AI loads one nav file per map from `bots/navigation/<mapname>.nav`. Maps without one get no bot navigation at all.

Nightdive Studios shipped nav files for the stock maps only, and authored them **by hand** in the engine's built-in nav editor. **The goal of this project was to make nav files generatable** - a BSPC -> AAS -> NAV3 converter, so that any Quake II BSP can have bot navigation without hand-placed waypoints.

The immediate motivation was a Quake II mod whose maps had no nav data, but the tool is not tied to any one mod and should not become so.

## Where it stands

**Finished.** The pipeline works end to end. Everything below is what it reports today, and the three gates are the format model, the AAS reader, and the writer against its own output:

```
$ python3 -m tests.roundtrip              # the format model
round trip : 174/174 byte-identical
corpus     : 66424 nodes, 224836 links, 12511 traversals, 1120 nav edicts

$ python3 -m tests.aascheck out/cache     # the AAS reader, vs bspc -aasinfo
bspc agrees: 193/193 on all 13 lump counts and the travel histogram

$ python3 kexnav.py generate --pak /path/to/gamedir --all -o out/
written    : 28 nav file(s)
totals     : 41340 nodes, 162723 links, 37782 traversals, 546 nav edicts

$ python3 -m tests.roundtrip --dir out/   # generated output, same gate
round trip : 28/28 byte-identical
```

Against Nightdive's own files, on the 193 stock maps BSPC can compile: **96.2% of Nightdive's 53914 nodes have a generated node within one spacing**, 94.1% of player spawns against Nightdive's own 96.9%, 1.76x the node count, and 193/193 files clean on the invariants. (53914 rather than 54177 because `mgu1m3`'s nav file was authored for a different revision of that map -- 126 of its 263 nodes sit outside the BSP's own world -- and `check` reports it separately instead of grading it.)

Of what is missed, **84% is geometry BSPC's area graph never covered** -- see [README.md](README.md#results). The converter's own constants are not the lever; `--seed-flood` is, and it is measured in [GENERATING](https://github.com/Niehztog/kexnav/wiki/GENERATING#bspcs-flood-seeding-improved-not-fixed).

## Layout

```
kexnav.py           the single entry point for actual use: generate / check

kexnav/nav3.py      the NAV3 format: Node/Link/Traversal/NavEdict/NavFile, enums, loads/dumps
kexnav/aas.py       BSPC's AAS output: areas, reachabilities, the BSP tree walk
kexnav/bsp.py       the two Quake II BSP lumps needed: entities and models
kexnav/convert.py   AAS -> NAV3, and every conversion constant with its evidence
kexnav/validate.py  the structural rules, shared by the gates and the generator
kexnav/pak.py       minimal Quake II PACK reader; reads entries on demand
kexnav/env.py       where the inputs live -- stock candidates, plus this machine's kexnav.local
kexnav/cli/         generate's and check's implementations, dispatched by kexnav.py

tests/test_kexnav.py  the unit test suite -- kexnav/bsp.py train translation, kexnav/validate.py
tests/roundtrip.py    dev-only gate: kexnav/nav3.py against the retail corpus, byte-exact
tests/aascheck.py     dev-only gate: kexnav/aas.py against bspc -aasinfo's own reading
```

Python 3, standard library only, no dependencies and no build step.

The reference documentation is **not in the tree**: it lives in the repository's [wiki](https://github.com/Niehztog/kexnav/wiki), a separate git repo at `git@github.com:Niehztog/kexnav.wiki.git`, and the local `docs/` folder it came from is untracked. [GENERATING](https://github.com/Niehztog/kexnav/wiki/GENERATING) is how the converter works and why, with the mapping table and the gaps; [FORMAT](https://github.com/Niehztog/kexnav/wiki/FORMAT) is the NAV3 spec, the complete field profile, and why it is trusted; [BACKGROUND](https://github.com/Niehztog/kexnav/wiki/BACKGROUND) is how bots work in the rerelease and what is and isn't moddable; [DECISION](https://github.com/Niehztog/kexnav/wiki/DECISION) is why this generates nav files rather than replacing the bots.

## Running things

```sh
python3 kexnav.py generate q2dm1 -o out/                        # generate one map
python3 kexnav.py generate --pak /path/to/gamedir --all -o out/ # a whole gamedir
python3 kexnav.py generate q2dm1 --deploy ~/gamedir             # install it in place
python3 kexnav.py generate q2dm1 --train-links -o out/          # plus func_train rides

python3 kexnav.py check --test                                  # the 4 feature maps
python3 kexnav.py check --all                                   # the whole oracle
```

Machine-specific paths **must not** be in the tree. `kexnav/env.py` walks a candidate list for the pak and the BSPC binary and every CLI takes an override, but the candidates there are only where a *stock* install puts things. The local paths live in `kexnav.local` beside `kexnav.py`, which git ignores - add one there rather than to `env.py`.

`generate` and `check` are the tool; `kexnav.py` deliberately stops there. Two more gates live in `tests/`, not in `kexnav.py`, because they don't touch a user's map at all - they test `kexnav/nav3.py` and `kexnav/aas.py` themselves against the retail corpus, the same role `tests/test_kexnav.py` plays for the rest of the library:

```sh
python3 -m tests.roundtrip                                      # format gate
python3 -m tests.roundtrip --dir out/                           # ... on generated files
python3 -m tests.aascheck out/cache                              # AAS gate
```

## Working agreements

* **The markdown is ASCII-only and not hard-wrapped.** One line per paragraph, per bullet, per table row; the reader's viewer decides where lines break. No em dashes, en dashes, minus signs or arrow glyphs either -- plain `-`, `->` and `x`. Both are deliberate, so do not re-wrap a file you edit and do not let an editor reintroduce a typographic dash. The Python still wraps at 79, as code should.
* **Headings separate their two halves with a colon, never with ` - `.** GitHub deletes an em dash from a slug but keeps a hyphen, so when the docs went ASCII-only every `#anchor` pointing at a dashed heading silently went one hyphen short and 18 intra-doc links broke without any renderer complaining. A colon is deleted from the slug like the em dash was, so the anchors stay stable. Check them after editing a heading -- compute the slug as lowercase, punctuation deleted, spaces to hyphens.
* **Re-run the gates after touching the model.** `python3 -m tests.roundtrip` after `kexnav/nav3.py`, `python3 -m tests.aascheck` after `kexnav/aas.py`, `kexnav.py check --test` after `kexnav/convert.py`. They are the only thing standing between a plausible-looking model and a wrong one, and they are fast. Treat a regression as a stop.
* **`kexnav/nav3.py` and `kexnav/aas.py` mirror their files, they do not interpret them.** Parallel arrays, raw indices, links not resolved into nodes, enum fields kept as plain ints with the enums used only for *naming*. A file may legally carry a value no enum lists, and coercing it would break the round trip. Enforcement lives in `kexnav/validate.py`; interpretation lives in `kexnav/convert.py`.
* **Every constant in `kexnav/convert.py` carries its evidence in a comment.** That is deliberate and worth maintaining: each one was measured against the 174-file corpus, and the comment is what lets the next session tell a measurement from a guess. If you change one, change the evidence with it.
* **Distinguish proven from inferred.** The record layout is proven - byte-exact round trip on 174 shipped and 28 generated files, corroborated by the engine's log and two independent implementations. Field *meanings* come from q2pro-ng's loader, several of them re-measured here. A handful of converter decisions are genuinely inferred and are listed as such in `kexnav.convert.GAPS`, which `kexnav.py check` prints. Keep that line visible.
* **Two Quake II entities are simulated in the converter rather than described to BSPC**, and both for the same reason: the rewrite BSPC would need is one that carries an `origin` key. Teleporters, and now `trigger_push` - BSPC recognises the classname and flags its areas, but reads the launch velocity off a *target entity's origin* the way Quake III builds a jump pad, and Q2's push has no target. So all 80 arena and 76 retail push triggers produced nothing. `kexnav/convert.py` runs the trajectory instead, which is cheap because AAS space is player-origin space: one `AAS_PointAreaNum` per step is already a swept box test. It models both idioms - a shaft that carries you at constant speed while you are inside it, and a vertical pad you *steer* off, the latter reproducing `AAS_HorizontalVelocityForJump` exactly. Reach for that pattern when a Q2 entity needs a *position* BSPC cannot be told about.
* The engine writes a timestamped log to `/mnt/c/Users/<you>/Saved Games/Nightdive Studios/Quake II/stdout.txt` on the Windows side. Reading it beats asking anyone to paste a console. Useful lines are tabulated in [BACKGROUND](https://github.com/Niehztog/kexnav/wiki/BACKGROUND#reading-the-engines-mind).
* Running the game itself needs Steam and takes over the desktop - ask rather than launching it unprompted.
* **BSPC so far is told about Quake II by rewriting the BSP it is handed.** Six rewrites live in `kexnav/bsp.py`: strip the lumps it cannot load, drop the entities the map marked `_nofill`, describe `func_plat2` and vertical `func_door` / `func_water` as `func_plat`, describe `func_train` as `func_bobbing`, present `func_door_rotating` as `func_door`, and seed its inside/outside flood from the BSP's own empty leaves. BSPC's gaps are mostly *recognition* gaps rather than capability gaps -- it already simulates the movement, it just matches Quake III's classnames -- so this reuses the simulation instead of reimplementing it, and BSPC's own filters reject what the geometry does not support.
* **A leak is worth chasing to the entity that caused it.** BSPC writes a `.lin` file whose last point is the origin it flooded out of; matching that against the entity lump names the culprit. That is how `_nofill` was found on `ware1`, which stopped it depending on the lossy `-nocsg` fallback. `compile_aas` deletes `bspc.log`, so diagnose by running BSPC by hand on `bsp.strip_for_bspc(raw)` in a scratch directory.
* **Never give a synthetic entity an `origin` key.** `FloodEntities` (`portals.c:843`) seeds BSPC's inside/outside decision from any entity that has one, so a key holding a geometric offset seeds from the void and BSPC reports `**** leaked ****` and writes nothing. Encode the offset in a duplicate brush model's bounds instead. This cost an afternoon to find twice.
* **An AAS-level score is a poor proxy for nav quality.** `compile_best_aas` ranks BSP variants on grounded areas then reachabilities, and on `rsewer2` and `rhangar2` that selected the `--door-movers` variant even though the nav graph it produced was *less* connected. If you add a variant, measure it at the nav level too -- `kexnav.cli.check.connectivity` gives the two numbers that matter, islands and the largest mutually reachable component. Re-measured for the 13 arena maps that have a `func_train` or a `func_door_rotating`: `--train-links --door-movers` buys 26 `TRAIN` links and one fewer island but **loses 20 `ELEVATOR` rides**, because one map's train variant trips BSPC's inline-model limit and falls back to a compile with no lift translation either. Both stay off.

## Things worth knowing before you form a plan

* The nav file is a **waypoint graph, not a navmesh**, despite the name. q2dm1 is 343 nodes at radius 32. Links are directional.
* Nav files carry **no reference to the BSP** they were built for.
* **Versions 2-6 differ in one layout detail only**: the traversal's fourth vec3 (the ladder plane) was added in v4, so older traversals are 36 bytes rather than 48. Everything else the version gates is interpretation of the link flags. Bumps have been additive.
* `GetPathToGoal` is used by **monsters as well as bots**, so nav data is not bot-only.
* **The load-bearing constant is 23.47.** AAS space is player-origin space (BSPC expands every brush by the player box), NAV3 space is floor space, and a nav node sits 23.47 below the AAS ground face. Get that wrong and everything is 24 units out. Measured four independent ways - [GENERATING](https://github.com/Niehztog/kexnav/wiki/GENERATING#coordinate-spaces).
* **AAS both over- and under-segments, and the converter corrects both.** A flat room is *one* convex area however large (Nightdive gave its own 320-unit `mals_box` nine nodes), while a BSP split leaves slivers a couple of units wide. Hence a lattice inside each area plus two node-merge rules with two different safety arguments - [GENERATING](https://github.com/Niehztog/kexnav/wiki/GENERATING#node-placement).
* **[q2pro-ng](https://github.com/skullernet/q2pro-ng)'s `src/server/nav.c` is the reference implementation** for field semantics. Consult it before reverse-engineering anything further - [FORMAT](https://github.com/Niehztog/kexnav/wiki/FORMAT) records what it settled.
* **`--seed-flood` raises the coverage number without raising what a bot can do**, and that is why it stays off. Over 40 stock maps it takes Nightdive-node coverage 91.8% to 94.2% and is never worse on any map -- but the largest mutually reachable component is *identical* on all 68 maps measured (40 stock, 28 arena), because a region BSPC's flood could not reach is one its reachability pass cannot join to the rest either. The waypoints come back, the route does not. On the arena set it buys nothing at all: not one spawn, not one node in the reachable component. Re-measure both numbers before arguing for a different default -- [GENERATING](https://github.com/Niehztog/kexnav/wiki/GENERATING#bspcs-flood-seeding-improved-not-fixed).
* **MuffMode's `sv nav_bake` is the only other generator**, and it is walk-only floor sampling; its own docs say jumps, ladders, teleports and lifts "cannot be inferred from floor sampling alone" and must be added by hand afterwards. That gap was this project's reason to exist, and it is now closed -- `func_train` included, though that one is behind `--train-links` and off by default because the measurement did not justify it. Push pads too, which MuffMode does not list but which BSPC also cannot infer on a Quake II map.
* BSPC 2.1h is used as an unmodified external binary, built native from a checkout alongside this repo; `kexnav.local` is where its path is recorded. Three quirks bite and are all handled in `kexnav/cli/generate.py`: a long filename argument smashes its stack, the 64-bit build aborts at exit *after* writing a good file, and brush chopping can fake a leak (`-nocsg` is the retry). It writes `bspc.log` into the current working directory.

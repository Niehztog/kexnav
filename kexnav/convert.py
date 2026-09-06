"""AAS -> NAV3. The converter.

BSPC has already done the hard part: it decomposed the map into convex areas
and ran a movement simulation to work out which pairs of areas a player can
get between, and how. This turns that graph into the waypoint graph the KEX
engine loads:

    AAS grounded area   ->  nav node
    AAS reachability    ->  nav link, with the travel type mapped to a link
                            type, plus a traversal for the types that need one

Everything the mapping rests on is either read out of BSPC's source or measured
against Nightdive's 174 hand-authored nav files. Where a value could not be
derived, this leaves the field at the sentinel Nightdive's own files use for
"unset" rather than inventing one; :data:`GAPS` lists those, and ``kexnav.py
check`` reports them.

Coordinate spaces
-----------------

The one thing that has to be right. BSPC expands every map brush by
``cfg.bboxes[0]``, whose ``mins.z`` is -24 (``AAS_ExpandMapBrush``,
``aas_map.c``), so **AAS space is player-origin space**: a point is in an open
area iff a player's origin can be there, and a grounded area's floor face sits
at real floor + 24.

**NAV3 space is floor space.** Measured on Nightdive's files: a node origin is
:data:`FLOOR_OFFSET` = 23.47 below the AAS ground face -- 216 of 222 locatable
nodes on q2dm1, 138 of 140 on q2dm3, exactly. Corpus-wide 88% of node origins
have a z fractional part of 0.53, which is floor + half a unit + a trace
epsilon. Traversal ``start``/``end`` are in the same space: 21064 of 22528
share that 0.53 fraction.

So every position crossing from AAS to NAV3 loses 23.47 in z, and nothing
else changes.
"""

import collections
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import aas, bsp, nav3, validate

#: AAS z minus nav z. See the module docstring -- this is measured, not
#: assumed, and it is *not* the 24 of ``PLAYER_MINS`` even though it is close;
#: the remaining 0.53 is half a unit plus a trace epsilon and Nightdive's
#: editor reproduces it to the hundredth.
FLOOR_OFFSET = 23.47

#: What ``node.radius`` gets. 32 on 63593 of 66424 corpus nodes, and the only
#: value MuffMode's generator emits. Deliberately *not* derived from the area's
#: extents: tested against Nightdive's files, there is no correlation --
#: radius-32 nodes sit in areas whose smaller horizontal extent runs from 8 to
#: 336 units, and the four radius-16 nodes on q2dm1 sit in a 480-unit-wide
#: area. It is a hand-tightened editor value, so a constant is the honest
#: choice.
DEFAULT_RADIUS = 32

#: How far apart nodes go inside one AAS area, in units.
#:
#: One node per area is not enough. AAS areas are convex, so a flat room is a
#: *single* area however large: Nightdive's own ``mals_box`` test map is one
#: 320-unit area and Nightdive gave it 9 nodes. Measured on the real maps, 51%
#: of q2dm1's floor and 52% of q2ctf1's sits in areas whose longer horizontal
#: extent is over 256 units, so leaving those as one node each would put a
#: single waypoint in the middle of every arena.
#:
#: 128 rather than a rounder number because it matches Nightdive's own density:
#: 343 nodes over q2dm1's 4699305 square units of floor is a node every ~117
#: units. MuffMode's sampler defaults to 96 and allows 48-256.
DEFAULT_SPACING = 128.0

#: How close two nodes may get before the second is dropped and its area
#: borrows the first.
#:
#: Needed because AAS over-segments: a BSP split leaves slivers of floor, and
#: one node per sliver puts waypoints a couple of units apart. Without this,
#: 25% to 51% of a generated map's nodes sat within 32 units of another, some
#: 1.2 units apart -- a bot would dither between them.
#:
#: 64 -- twice :data:`DEFAULT_RADIUS`, so two nodes' radii never overlap.
#: Chosen by sweeping it against the oracle rather than by taste. Over seven
#: stock maps, moving from 32 to 64 costs at most 0.7% of the Nightdive nodes
#: covered within one spacing while cutting the node count by a third:
#:
#: =========  ===============  ==========  ==========
#: map        Nightdive nodes  gen at 32   gen at 64
#: =========  ===============  ==========  ==========
#: base64     1167             2798        1854
#: city64     1233             2463        1544
#: q2ctf1      435              900         652
#: =========  ===============  ==========  ==========
#:
#: Coverage at 128 units stayed at 99.5/95.6/96.1% and 99.2/94.9/95.6%. Going
#: on to 96 units started to bite -- the share of Nightdive nodes with a
#: generated node within 64 fell from ~78% to ~63%.
#:
#: For reference, Nightdive's own spacing: the closest pair of nodes on q2dm1,
#: base64 and q2ctf1 is 33.8, 38.7 and 46.2 units, and 158 of the 174 shipped
#: files keep every pair at 32 or more. So this is a default in the spirit of
#: the corpus, not an invariant of it -- the other 16 files go tighter, down to
#: 17.18 on security.nav.
MIN_NODE_SEPARATION = 64.0

#: Node flags that carry over when one area borrows another's node.
#:
#: A borrowed node sits physically inside the area that created it, so flags
#: describing *that spot* -- ``UNDER_WATER``, ``CROUCH`` -- must keep coming
#: from its owner. Marking a dry node ``UNDER_WATER``, or making a bot crouch
#: on open floor because a crawlspace next door borrowed the node, would both
#: be wrong.
#:
#: These three are different: they say "something special is reachable from
#: here" rather than "this spot is like this", and the engine needs them set
#: for the node the link actually starts from. ``LADDER`` would belong here
#: too, but :func:`node_flags` never returns it -- a ladder's foot node gets
#: it directly, so there is nothing to borrow.
BORROWABLE_NODE_FLAGS = int(nav3.NodeFlags.TELEPORTER | nav3.NodeFlags.PUSHER
                            | nav3.NodeFlags.ELEVATOR)

#: The radius within which a node is merged *regardless* of connectivity.
#:
#: :data:`MIN_NODE_SEPARATION` merges only into a walk-connected area, because
#: at 64 units two nodes could be on opposite sides of a wall. Below 32 they
#: cannot be, and that is a property of the input rather than a guess: AAS is
#: built from brushes expanded by the player box, so free space clears every
#: solid surface by 16 horizontally, 24 above a floor and 32 below a ceiling.
#: Two points in AAS space with solid between them are therefore at least 32
#: units apart plus the thickness of whatever separates them.
#:
#: So this rule is what actually guarantees the floor on node spacing --
#: without it, two areas that share no walk reachability can still land nodes
#: on top of each other, which was measured at 0.0 units apart on city64.
CLEARANCE_MERGE_RADIUS = 32.0

#: ``TEAM_RED | TEAM_BLUE``. The value on 179776 of 191714 v6 links, and on
#: every link of every type's plurality. The other observed v6 combinations
#: add ``EXIT_AT_TARGET|WALK_ONLY``, which is per-link authoring.
DEFAULT_LINK_FLAGS = int(nav3.ALL_TEAMS)

#: A* weight. 0.8 in 171 of the 174 shipped files.
DEFAULT_HEURISTIC = 0.8

#: How far behind ``start`` the funnel run-up point sits, per link type.
#:
#: Measured across the corpus, and exact: for every jump-family link the funnel
#: lies on the reverse of the horizontal ``start``->``end`` axis at ``start``'s
#: own height, a whole number of units back. ``LONG_JUMP`` 32 (915 of 1008),
#: ``BARRIER_JUMP`` 8 (959 of 1098), ``ROCKET_JUMP`` 48 (21 of 22), ``TRAIN``
#: 16 (56 of 56). On-axis in 1008/1008, 1064/1098, 22/22 and 56/56
#: respectively, with ``funnel.z == start.z`` in every single case.
FUNNEL_RUNUP = {
    nav3.LinkType.LONG_JUMP: 32.0,
    nav3.LinkType.MANUAL_LONG_JUMP: 32.0,
    nav3.LinkType.BARRIER_JUMP: 8.0,
    nav3.LinkType.MANUAL_BARRIER_JUMP: 16.0,
    nav3.LinkType.ROCKET_JUMP: 48.0,
    nav3.LinkType.TRAIN: 16.0,
}

#: ``nav_edict.model`` is the inline brush model index **plus one**. Verified
#: on all 1120 edicts in the corpus: ``model - 1`` is an index owned by a
#: brush entity in the matching BSP, every time, and the owner is always a
#: mover or interactive brush (func_button 361, func_plat 203, func_plat2 199,
#: func_door 168, func_explosive 79, func_train 77, ...). FORMAT.md previously
#: recorded this field as the index itself, which is off by one.
EDICT_MODEL_BIAS = 1

#: A nav edict's box is the mover's brush model bounds grown by this on every
#: axis: 972 of the 987 corpus edicts whose model still sits at its brush
#: position match at exactly 2 units, 15 at 1.
EDICT_EXPAND = 2.0

#: AAS travel type -> NAV3 link type.
#:
#: The Q3-only moves are absent on purpose and their reachabilities are
#: dropped: ``BFGJUMP``, ``GRAPPLEHOOK``, ``DOUBLEJUMP``, ``RAMPJUMP`` and
#: ``STRAFEJUMP`` have no NAV3 link type and no Quake II move behind them.
#: ``INVALID`` is BSPC's own "temporarily not possible" marker.
TRAVEL_TO_LINK = {
    aas.TravelType.WALK: nav3.LinkType.WALK,
    aas.TravelType.CROUCH: nav3.LinkType.CROUCH,
    aas.TravelType.BARRIERJUMP: nav3.LinkType.BARRIER_JUMP,
    aas.TravelType.JUMP: nav3.LinkType.LONG_JUMP,
    aas.TravelType.LADDER: nav3.LinkType.LADDER,
    aas.TravelType.WALKOFFLEDGE: nav3.LinkType.WALK_OFF_LEDGE,
    # NAV3 has no swim link type and no swim move state; Nightdive's files
    # carry underwater nodes joined by ordinary WALK links, and mark the nodes
    # UNDER_WATER instead. 2060 corpus nodes have that flag.
    aas.TravelType.SWIM: nav3.LinkType.WALK,
    # "jump out of the water" -- a barrier climb with water under it.
    aas.TravelType.WATERJUMP: nav3.LinkType.BARRIER_JUMP,
    aas.TravelType.TELEPORT: nav3.LinkType.TELEPORT,
    aas.TravelType.ELEVATOR: nav3.LinkType.ELEVATOR,
    aas.TravelType.ROCKETJUMP: nav3.LinkType.ROCKET_JUMP,
    aas.TravelType.JUMPPAD: nav3.LinkType.PUSHER,
    aas.TravelType.FUNCBOB: nav3.LinkType.TRAIN,
}

#: Link types that carry a traversal, from the corpus type/traversal profile.
#: This is per-type and *not* "everything but WALK": ``TELEPORT``, ``PUSHER``,
#: ``CROUCH`` and ``PIVOT_AND_JUMP`` never carry one in 174 files, while the
#: jumps, ``LADDER``, ``ELEVATOR`` and ``TRAIN`` always do.
NEEDS_TRAVERSAL = frozenset({
    nav3.LinkType.LONG_JUMP,
    nav3.LinkType.MANUAL_LONG_JUMP,
    nav3.LinkType.BARRIER_JUMP,
    nav3.LinkType.MANUAL_BARRIER_JUMP,
    nav3.LinkType.ROCKET_JUMP,
    nav3.LinkType.WALK_OFF_LEDGE,
    nav3.LinkType.LADDER,
    nav3.LinkType.ELEVATOR,
    nav3.LinkType.TRAIN,
})

#: Quake II's teleporters, which BSPC cannot see.
#:
#: ``AAS_Reachability_Teleport`` looks for Q3's ``trigger_teleport``, or a
#: ``trigger_multiple`` aimed at a ``target_teleporter``, and additionally
#: requires the source area to carry ``AREACONTENTS_TELEPORTER``, which comes
#: from a *brush's* contents. Quake II's teleporter is the **point** entity
#: ``misc_teleporter``: it has no brush at all, and spawns its trigger at
#: runtime. So BSPC finds nothing -- 0 teleport reachabilities wherever one is
#: used, including the stock maps q2ctf1 and q2ctf4.
#:
#: This is synthesised here instead of patched into BSPC, because a teleport
#: needs no physics: it is an entity lookup plus two ``AAS_PointAreaNum``
#: calls, and a NAV3 ``TELEPORT`` link carries no traversal at all -- 0 of the
#: corpus's 54 do. Nothing BSPC can do that the converter cannot, so this
#: keeps the C tree unforked. See the wiki's GENERATING page.
TELEPORTER_CLASSNAMES = ("misc_teleporter", "trigger_teleport")

#: Where ``misc_teleporter`` puts its runtime trigger, relative to its own
#: origin: ``SP_misc_teleporter`` spawns it at the pad origin with
#: ``mins {-8,-8,8}``, ``maxs {8,8,24}``.
TELEPORTER_TRIGGER_MINS = (-8.0, -8.0, 8.0)
TELEPORTER_TRIGGER_MAXS = (8.0, 8.0, 24.0)

#: ``teleporter_touch`` sets ``other->s.origin = dest->s.origin`` and then
#: ``other->s.origin[2] += 10``, so this is where the player's *origin* lands.
TELEPORT_DEST_RISE = 10.0

#: Quake II's push triggers, which BSPC cannot use.
#:
#: ``AAS_Reachability_JumpPad`` does match the classname -- and ``aas_map.c``
#: does give the brush ``CONTENTS_JUMPPAD``, so the pad's areas come out
#: correctly flagged -- but ``AAS_GetJumpPadInfo`` then derives the launch
#: velocity from a **target entity's origin**, the way Quake III's jump pads
#: are built. Quake II's ``trigger_push`` has no target: it pushes along the
#: direction its ``angle`` selects at ``speed * 10`` units per second. So BSPC
#: prints *"trigger_push without target entity"* and emits nothing, on all 80
#: push triggers of the 28-map arena set and all 76 in the retail pak.
#:
#: Synthesised here rather than described to BSPC because the BSP rewrite that
#: would work -- inventing the target entity BSPC wants -- needs it to carry an
#: ``origin`` key, and ``FloodEntities`` seeds its inside/outside decision from
#: exactly those; see :func:`kexnav.bsp.add_train_bobbing`. The trajectory is
#: cheap to run here instead, because AAS space is player-origin space: a
#: single ``AAS_PointAreaNum`` per step is already a swept box test.
PUSH_CLASSNAMES = ("trigger_push",)

#: ``trigger_push_touch``: ``other->velocity = self->movedir * (self->speed *
#: 10)``, and ``SP_trigger_push`` defaults an unset ``speed`` to 1000.
PUSH_SPEED_SCALE = 10.0
PUSH_SPEED_DEFAULT = 1000.0

#: ``SPAWNFLAG_PUSH_START_OFF``. Without a ``targetname`` nothing can ever
#: switch it on, and ``SP_trigger_push`` turns the entity into ``SOLID_BSP``
#: instead -- a wall, not a push. One such trigger in the retail pak.
PUSH_SPAWNFLAG_START_OFF = 8

#: Quake II's gravity, and BSPC's: ``sv_gravity`` 800, matching the
#: ``phys_gravity`` default ``AAS_InitSettings`` uses for its own simulation.
PUSH_GRAVITY = 800.0

#: The player bounding box, used to decide when the origin being traced is
#: still inside the trigger's volume, since a trigger fires on a *box* overlap.
#:
#: Not taken from ``p_client.cpp`` on trust: it is also the box BSPC expanded
#: every brush by, so it has to be this one or the trace would disagree with
#: the space it is tracing through. ``bspc -aasinfo`` reports
#: ``bbox 0: (-16,-16,-24) (16,16,32)`` on all 97 compiled files.
PLAYER_MINS = (-16.0, -16.0, -24.0)
PLAYER_MAXS = (16.0, 16.0, 32.0)

#: How far one step of a push trajectory may advance, in units.
#:
#: The trajectory is traced as a *point* because AAS space is player-origin
#: space -- BSPC expanded every brush by the player box, so a point that is
#: inside an area is a position the player's box fits in. That makes tunnelling
#: the only failure mode, and 8 units leaves a factor of four against the 32
#: units of clearance the expansion guarantees around any solid.
PUSH_STEP = 8.0

#: Caps on one trajectory: seconds of flight, and total path length. The length
#: is what actually bites -- an arena set carries push triggers with ``speed``
#: 250000, a quarter of a million, which is not a move a bot can navigate and
#: which would otherwise integrate for a million steps.
PUSH_MAX_TIME = 6.0
PUSH_MAX_PATH = 16384.0

#: How far to slide the entry point back towards the volume's middle when the
#: upwind end is not somewhere a player fits.
#:
#: It usually is not: AAS space is player-origin space, so an origin has to
#: clear every solid surface by 16 units horizontally, and a trigger brush is
#: routinely built flush into the wall it starts at. Sliding in by a quarter
#: and then to the middle recovers the entry without giving up on the idea that
#: a wind tunnel is entered at one end.
PUSH_ENTRY_FRACTIONS = (0.0, 0.25, 0.5)


#: How finely the pad's column is walked. A trigger's own bounds are not a
#: guide to where a player can be inside it -- the box routinely starts below
#: the floor, and a shaft's air and its floor are different areas -- so the
#: column is scanned rather than probed at a fixed ladder of offsets. 4 units
#: because on ``mgdm1`` the only standable slice of one pad's column is eight
#: units thick, and an 8-unit walk stepped straight over it.
PUSH_COLUMN_STEP = 4.0

#: A pad this close to vertical is steered rather than aimed.
#:
#: Quake II's push replaces the player's whole velocity, so a straight-up pad
#: drops them back where they started -- yet Nightdive's own files hang four
#: ``PUSHER`` links off ``mgdm1``'s three vertical pads, two of them from the
#: same pad to two different ledges. The player steers in the air, and
#: ``AAS_Reachability_JumpPad`` models exactly that: below this much horizontal
#: launch velocity (``be_aas_reach.c:3670``) it stops predicting one trajectory
#: and instead asks, of every ground face above the pad, whether the player
#: could steer to it.
PUSH_AIR_CONTROL_MAX_HORIZONTAL = 100.0

#: The horizontal speed a steered landing may need, the literal bound at
#: ``be_aas_reach.c:3714``. Above it BSPC does not believe the player gets
#: there, and neither does this.
PUSH_AIR_CONTROL_SPEED = 150.0

#: ``phys_maxvelocity``, the ceiling ``AAS_HorizontalVelocityForJump`` applies
#: before it reports failure. 320 in ``cfgq3.c`` and in ``AAS_InitSettings``.
PUSH_MAX_HORIZONTAL = 320.0

#: What a synthesised link costs, for the cheapest-wins tie-break in
#: :func:`convert`. Deliberately far above any ``traveltime`` BSPC computes,
#: so where a teleporter or a push pad happens to join a node pair a player can
#: simply walk between, the walk is the link that survives.
ENTITY_LINK_COST = 1e6

# -- ladders ---------------------------------------------------------------
#
# A whole physical ladder becomes **one** ``LADDER`` link, from a node at its
# foot to a node on the floor at the top, and that is not a simplification of
# BSPC's output -- it is what the format wants. Measured over the corpus's 272
# ladder links:
#
#   * ``traversal.start`` equals the source node's origin and ``traversal.end``
#     the target's, to the last bit, in **272 of 272**. The traversal is not an
#     independent path; it restates the two nodes. A link whose traversal
#     begins somewhere the bot is not cannot be executed, which is what
#     BSPC's own endpoints produced: snapped to the nearest lattice node they
#     landed a median 50-80 units, and up to 368, from where the climb starts.
#   * the source node carries :attr:`~kexnav.nav3.NodeFlags.LADDER` in 254 and
#     the target does not in 246. Nightdive marks 277 nodes ``LADDER`` in
#     66424 -- one per ladder, not one per area a climb passes through.
#   * 269 of 272 go **up**, 268 of 270 source nodes have exactly one, and 262
#     have no reverse link at all: a bot descends by walking off the top, and
#     the foot is still reachable from the top by some other route on 268 of
#     the 272.
#
# BSPC instead reports a ladder as a chain of one reachability per pair of
# ladder areas, 64 units of climb each, in both directions, plus a
# ``TRAVEL_JUMP`` onto the bottom rung -- ``AAS_Reachability_Ladder`` in
# ``be_aas_reach.c``. Those chunk boundaries are mid-air positions on the
# ladder, which is a fine way to *simulate* a climb and a useless place for a
# waypoint. So the chain is collapsed here and the areas it passes through are
# left without nodes; see :func:`usable_areas`.

#: How close two ladder reachabilities must be, horizontally, to be the same
#: physical ladder. They share the ladder face's plane already -- that is the
#: first half of the grouping -- so this only has to separate two ladders on
#: one wall. Consecutive chunks of one ladder are stacked vertically and their
#: endpoints agree in xy to within the 3 units ``AAS_Reachability_Ladder``
#: pushes an end off the face, so the threshold has three orders of slack at
#: the bottom; at the top, no pair of Nightdive's own ladders on a shared wall
#: plane sits this close.
LADDER_COLUMN_RADIUS = 48.0

#: Sideways probe offsets when looking for the floor at a ladder's foot, in
#: units along the ladder face's normal. Signed, and tried in this order.
#:
#: The endpoints BSPC hands over sit *on* the ladder face -- ``area1point`` is
#: the shared edge's midpoint slid along the face -- so ``AAS_PointAreaNum``
#: is being asked about a boundary plane and answers "solid" often enough to
#: matter: 244 of 664 columns over the 193 stock maps found no floor at offset
#: 0 alone. Both signs are tried because which side of the face is open is not
#: knowable from the plane: on q2dm2 the free space is on the normal's far
#: side, on badlands the near one. A probe into solid costs a tree walk and
#: returns nothing, so guessing wrong is only ever slow.
LADDER_PROBE_OFFSETS = (0.0, 4.0, -4.0, 8.0, -8.0, 16.0, -16.0, 24.0, -24.0)

#: The same for the step-off point at the top, which needs to reach further:
#: Nightdive's own top nodes sit a median 45 units out from the ladder face
#: (p10 17, p90 82), because the node goes where you land, not on the lip.
LADDER_TOP_OFFSETS = (0.0, 8.0, -8.0, 16.0, -16.0, 24.0, -24.0, 32.0, -32.0,
                      40.0, 48.0, 56.0, 64.0)

#: How far below the top step-off point its floor may be. ``area2point`` is
#: 32 units above the top shared edge and the reachability's end another 16,
#: so the platform is ~48 down; 96 leaves room for a step without letting the
#: probe fall back down the shaft it just climbed.
LADDER_TOP_DROP = 96.0

#: How far below the bottom rung to look for the floor. Generous on purpose:
#: a ladder that starts well above its floor is normal -- Nightdive's own foot
#: nodes sit a median 80 units below the lowest AAS endpoint, and 200 at the
#: tenth percentile -- and the scan stops at the first floor it finds anyway.
LADDER_FOOT_DROP = 512.0

#: How far up the floor probe starts, and its step.
#:
#: The rise clears the ground plane the endpoint may be sitting exactly on.
#: The step is :data:`PUSH_COLUMN_STEP`'s, for the same reason and with the
#: same cost: an area only a few units thick is a real thing to land on and a
#: coarser walk steps over it. ``mals_ladder_test``'s upper floor is one --
#: area 18 spans z 260 to 264 -- and an 8-unit scan missed all four of that
#: map's ladders. Samples are also phased to the middle of a step, because
#: Quake geometry sits on integer bounds and ``AAS_PointAreaNum`` asked about
#: a boundary plane answers with the solid side.
LADDER_PROBE_RISE = 24.0
LADDER_PROBE_STEP = 4.0

#: Below this much climb a ladder is not worth a link -- a bot walks or steps
#: up 32 units without help, and a column this short is usually a ladder
#: texture on a doorway rather than a route. 19 of 664 stock columns.
LADDER_MIN_CLIMB = 32.0

#: ``node.radius`` for a ladder's foot node. Every one of Nightdive's 277
#: ``LADDER``-flagged nodes has it: 276 at 4 and one at 12, against 32 on
#: 63053 of 66424 nodes overall. ``Nav_NodeReached`` compares it against the
#: horizontal distance to the node, so 4 is the engine being told the bot must
#: be *at* the ladder, not near it, before the climb counts as begun.
LADDER_FOOT_RADIUS = 4

#: Whether to describe the climb back *down* as well.
#:
#: Nightdive almost never does: 269 of 272 ladder links go up, and 262 have no
#: reverse at all -- a bot gets down by walking off the top, and the foot is
#: still reachable from the top by some other route on 268 of the 272. So a
#: reverse link goes in only where that is *not* true and the descent would
#: otherwise be lost, which is 24 of the arena set's 324 climbs, both times on
#: a map whose shaft has no stairs.
#:
#: ``mgu3m1`` is the corpus's one bidirectional ladder and settles how to
#: write it: n499->n500 and n500->n499, both source nodes flagged ``LADDER``
#: at radius 4, and **the same plane vector on both** -- its dot product with
#: the direction of travel is +1.0 going up and -1.0 coming down, so the field
#: describes the ladder and not the move.
LADDER_REVERSE_WHEN_STRANDED = True

#: How close an existing node has to be to serve as a ladder endpoint instead
#: of a new one. Tight at the foot because the traversal starts there and a
#: node one lattice cell away is a node off the ladder; loose at the top
#: because that is where Nightdive's own node sits -- a median 45 units from
#: the face -- and a lattice node on the platform is the same place.
LADDER_FOOT_REUSE = 16.0
LADDER_TOP_REUSE = 64.0

#: Link types that get a nav edict, when a mover model can be identified. Every
#: one of the corpus's 514 ELEVATOR links has an edict, and 55 of 56 TRAINs.
#: The corpus also puts edicts on 480 WALK links -- doors and buttons on the
#: path -- see :data:`DOOR_EDICT_LINK_TYPES`.
NEEDS_EDICT = frozenset({nav3.LinkType.ELEVATOR, nav3.LinkType.TRAIN})

#: Fields and behaviours a generated file does not reproduce, each with what
#: Nightdive's own files do instead. Kept as data so ``kexnav.py check`` can
#: report it and so the list cannot quietly rot.
GAPS = (
    ("node.radius", "32 everywhere but a ladder's foot, which gets 4 the way "
                    "all 277 of Nightdive's LADDER nodes do. Nightdive "
                    "hand-tightens another 4% of nodes besides, and that part "
                    "is not reproduced: the value does not correlate with AAS "
                    "area extents"),
    ("which ladders exist",
     "one climb per group of BSPC LADDER reachabilities sharing a wall plane "
     "and a column, which is more than Nightdive waypoints -- 476 columns "
     "resolve over the 193 stock maps against their 272 links, and 181 of "
     "those 272 have a generated foot within 96 units (median 19). The "
     "surplus is real ladder surface BSPC found and a hand author skipped; "
     "the shortfall is mostly a column whose top or foot has no floor in the "
     "AAS at all, counted in Stats.ladders_unresolved"),
    ("ELEVATOR traversal.funnel", "left unset; Nightdive points it at an "
                                  "adjacent node's origin (510 of 513), which "
                                  "node being unexplained"),
    ("nav edicts on WALK links",
     "not emitted; Nightdive has 480. The AAS does say which mover a link "
     "crosses -- a func_door's volume becomes an area whose contents carry "
     "its brush model number -- but that is not what Nightdive's records are: "
     "measured over the corpus, 40 of 43 WALK edicts name an entity the link "
     "never passes "
     "through, and the owners are func_button 27 and func_explosive 8 rather "
     "than doors. They mark the thing to shoot or press, which needs route "
     "reasoning a converter does not do"),
    ("TRAIN links", "only with --train-links, which describes each func_train "
                    "to BSPC as a func_bobbing; off by default because the "
                    "measurement did not justify it. Nightdive has 56 across 174 maps"),
    ("push triggers outside the AAS",
     "a trigger_push whose volume BSPC lost, or whose landing has no node, "
     "yields nothing; Stats.pushers_unresolved counts them with the reason. "
     "Over the 28-map arena set that is 8 of 80 pads BSPC never modelled, 1 "
     "landing nowhere and 1 too far from any node -- the other 42 that make no "
     "link move a player less than one node apart, which is a wind effect "
     "rather than a route"),
    ("push links up a shaft that is one AAS area",
     "not emitted, because the node lattice is horizontal: a 350-unit updraft "
     "is one convex area with one ground face, so it holds a single node and "
     "the ride has nowhere to land but where it started. hangar2 and rhangar2 "
     "are both this. The connection is not lost -- BSPC's reachabilities out "
     "of that area still hang off the same node -- but it is described as a "
     "rocket jump rather than a ride"),
    ("teleport pads outside AAS space",
     "a node is synthesised at the entity origin and walk-linked to its "
     "nearest neighbour -- the position matches Nightdive's to the unit, but "
     "the walk link is inferred rather than traced"),
    ("PIVOT_AND_JUMP, MANUAL_* link types",
     "not emitted; they are editor-authored variants of moves BSPC reports as "
     "the plain type, and the corpus offers no rule that separates them. "
     "MANUAL_BARRIER_JUMP's geometry is BARRIER_JUMP's (horizontal median 98 "
     "against 77, height 34 against 32) and MANUAL_LONG_JUMP's p10-p90 spread "
     "sits inside LONG_JUMP's. 450 links across 174 maps"),
    ("node.flags CHECK_* bits", "not emitted; q2pro-ng treats them as a "
                                "runtime re-evaluation mask, so a generator "
                                "has nothing to compute. Measured rather than "
                                "assumed for CHECK_FOR_HAZARD, the one that "
                                "looks derivable: of Nightdive's 27 in a "
                                "40-map sample only 6 sit in a lava area, "
                                "against "
                                "1764 lava and slime areas in those maps"),
    ("nodes inside lava and slime",
     "emitted like any other liquid area -- 266 of the arena set's 41340 nodes "
     "and 154 of 33906 over 40 stock maps. AAS marks the contents and the "
     "nodes carry UNDER_WATER, but NAV3 has no travel-flag equivalent of the "
     "avoidance weight BSPC's own router applies, and dropping the areas would "
     "cut reachability rather than add it"),
)


@dataclass
class Stats:
    """What a conversion did, for the report and for the tests."""

    areas: int = 0
    areas_used: int = 0
    nodes_before_prune: int = 0
    reachabilities: int = 0
    nodes: int = 0
    links: int = 0
    traversals: int = 0
    edicts: int = 0
    dropped_travel: Dict[str, int] = field(default_factory=collections.Counter)
    dropped_endpoint: int = 0
    dropped_duplicate: int = 0
    link_types: Dict[str, int] = field(default_factory=collections.Counter)
    node_flags: Dict[str, int] = field(default_factory=collections.Counter)
    isolated_pruned: int = 0
    nodes_merged: int = 0
    edicts_unresolved: int = 0
    teleports: int = 0
    pushes: int = 0
    ladders: int = 0
    ladders_reversed: int = 0
    nodes_synthesised: int = 0
    teleporters_unresolved: Dict[str, int] = field(default_factory=collections.Counter)
    pushers_unresolved: Dict[str, int] = field(default_factory=collections.Counter)
    ladders_unresolved: Dict[str, int] = field(default_factory=collections.Counter)


class ConvertError(Exception):
    pass


def _sub_z(point, dz=FLOOR_OFFSET):
    return (point[0], point[1], point[2] - dz)


def _polygon_centroid(points):
    """Area-weighted centroid of a polygon in the xy plane, and the mean z.

    Area-weighted rather than a vertex average: BSPC's areas come out of a BSP
    split and often have several vertexes bunched along one edge, which drags a
    plain average off the middle of the floor.
    """
    cx = cy = a2 = 0.0
    for i, p in enumerate(points):
        q = points[(i + 1) % len(points)]
        cross = p[0] * q[1] - q[0] * p[1]
        a2 += cross
        cx += (p[0] + q[0]) * cross
        cy += (p[1] + q[1]) * cross
    z = sum(p[2] for p in points) / len(points)
    if abs(a2) < 1e-6:            # degenerate, e.g. a face seen edge-on
        return (sum(p[0] for p in points) / len(points),
                sum(p[1] for p in points) / len(points), z)
    return (cx / (3.0 * a2), cy / (3.0 * a2), z)


def node_origin(a, areanum):
    """Where a single node for this AAS area goes, in nav space.

    Horizontally the centroid of the area's ground face, which is the middle
    of the floor a player can stand on -- not ``area.center``, which is the
    middle of the convex *volume* and drifts off the floor in areas with a
    sloped or partial ceiling. Vertically the ground face's own height, less
    :data:`FLOOR_OFFSET`.

    Falls back to ``area.center`` with ``area.mins.z`` for areas with no
    ground face, which is every liquid and ladder area.
    """
    area = a.areas[areanum]
    face = a.ground_face(areanum)
    if face is not None:
        c = _polygon_centroid(a.face_points(face))
        return (c[0], c[1], c[2] - FLOOR_OFFSET)
    return (area.center[0], area.center[1], area.mins[2] - FLOOR_OFFSET)


def _point_in_polygon_xy(point, polygon):
    """Ray-crossing test in the xy plane."""
    x, y = point[0], point[1]
    inside = False
    n = len(polygon)
    for i in range(n):
        ax, ay = polygon[i][0], polygon[i][1]
        bx, by = polygon[(i + 1) % n][0], polygon[(i + 1) % n][1]
        if (ay > y) != (by > y):
            t = (y - ay) / (by - ay)
            if x < ax + t * (bx - ax):
                inside = not inside
    return inside


def area_node_origins(a, areanum, spacing=DEFAULT_SPACING):
    """Node positions for one AAS area, in nav space, with their lattice
    coordinates so neighbours can be linked.

    Returns a list of ``(origin, (i, j))`` where ``(i, j)`` is a cell on a
    *global* lattice -- global so that the lattice does not restart per area
    and adjacent areas' nodes stay aligned. Areas smaller than the spacing
    fall back to a single centroid node, whose lattice coordinate is None.

    **Why the lattice needs no inset from the walls.** AAS is built from map
    brushes expanded by the player bounding box (``AAS_ExpandMapBrush``), so a
    point inside an area is by construction a position a player's *origin* can
    occupy. The ground face polygon is therefore already the walkable set, and
    a grid point inside it is already clear of the geometry. Each candidate is
    confirmed against ``AAS_PointAreaNum`` as well, which catches the rare
    non-convex ground face and costs a tree walk.
    """
    face = a.ground_face(areanum)
    if face is None:
        return [(node_origin(a, areanum), None)]

    points = a.face_points(face)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ground_z = sum(p[2] for p in points) / len(points)

    out = []
    i0, i1 = math.floor(min(xs) / spacing), math.ceil(max(xs) / spacing)
    j0, j1 = math.floor(min(ys) / spacing), math.ceil(max(ys) / spacing)
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            x, y = i * spacing, j * spacing
            if not _point_in_polygon_xy((x, y), points):
                continue
            # a hair above the floor plane, so the tree walk cannot land on
            # the boundary itself
            if a.point_area_num((x, y, ground_z + 1.0)) != areanum:
                continue
            out.append(((x, y, ground_z - FLOOR_OFFSET), (i, j)))

    if not out:
        return [(node_origin(a, areanum), None)]
    return out


def node_flags(a, areanum, elevator_targets=()):
    """``nav_node_flags_t`` for an area, from its AAS flags and contents.

    Only the flags that describe what the *place* is get set. The ``CHECK_*``
    bits are omitted deliberately -- q2pro-ng groups them as a mask of things
    the engine re-evaluates at runtime, so there is nothing static to compute
    -- and so is ``DISABLED``, which no shipped file uses.

    ``LADDER`` is deliberately absent: it belongs to the one node at a
    ladder's foot, which :func:`convert` sets it on directly, and not to every
    area a climb touches. Nightdive marks 277 nodes with it in 66424, and
    q2pro-ng's ``nav.c`` excludes a ``LADDER`` node from its node searches, so
    spending the flag on ordinary floor costs more than the description is
    worth.
    """
    s = a.areasettings[areanum]
    flags = 0
    if s.contents & (aas.AreaContents.WATER | aas.AreaContents.SLIME
                     | aas.AreaContents.LAVA):
        flags |= nav3.NodeFlags.UNDER_WATER
    if s.contents & aas.AreaContents.TELEPORTER:
        flags |= nav3.NodeFlags.TELEPORTER
    if s.contents & aas.AreaContents.JUMPPAD:
        flags |= nav3.NodeFlags.PUSHER
    if areanum in elevator_targets:
        flags |= nav3.NodeFlags.ELEVATOR
    # PRESENCE_NORMAL absent means a player only fits here crouched.
    if not s.presencetype & aas.PresenceType.NORMAL:
        flags |= nav3.NodeFlags.CROUCH
    return int(flags)


def link_flags(reach):
    """``nav_link_flags_t`` for a reachability.

    Team availability carries across: AAS stores it as ``NOTTEAM1`` /
    ``NOTTEAM2`` exclusions in the top byte of ``traveltype``, NAV3 as
    ``TEAM_RED`` / ``TEAM_BLUE`` inclusions, so an exclusion clears the
    corresponding bit. The team *correspondence* (AAS team 1 = red) is
    inferred, not measured -- no shipped nav file pairs with an AAS file that
    has team flags set -- but it only arises on team-locked doors, and both
    readings leave a symmetric map symmetric.
    """
    flags = DEFAULT_LINK_FLAGS
    if reach.not_team1:
        flags &= ~int(nav3.LinkFlags.TEAM_RED)
    if reach.not_team2:
        flags &= ~int(nav3.LinkFlags.TEAM_BLUE)
    return flags


def funnel_for(link_type, start, end):
    """The traversal funnel, or the unset sentinel.

    For the jump family this is a run-up point: :data:`FUNNEL_RUNUP` units
    behind ``start``, along the reverse of the horizontal ``start``->``end``
    axis, at ``start``'s height. Anything else is left unset, which is what
    Nightdive's files do for every ``LADDER`` and for 7699 of 9111
    ``WALK_OFF_LEDGE`` traversals.
    """
    back = FUNNEL_RUNUP.get(link_type)
    if back is None:
        return (nav3.UNSET_COORD,) * 3
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 1e-3:             # a purely vertical move has no run-up axis
        return (nav3.UNSET_COORD,) * 3
    return (start[0] - dx / length * back,
            start[1] - dy / length * back,
            start[2])


def ladder_plane_for(a, reach):
    """The unit normal of the ladder face a ``LADDER`` reachability climbs.

    ``reach.facenum`` is a *signed* face index for ladders -- the same
    convention as the area's own face index, where a positive entry means the
    plane normal points into the area -- so the sign selects which side the
    climber is on. Measured on Nightdive's files the plane is a unit vector in
    all 264 ladder traversals, and horizontal in all of them (x or y aligned in
    249, oblique in 15), so it is the wall normal rather than a plane equation.

    Which *way* it points is not decided here -- see :func:`oriented_plane`.
    """
    facenum = reach.facenum
    if not facenum or abs(facenum) >= len(a.faces):
        return None
    normal = a.planes[a.faces[abs(facenum)].planenum].normal
    if facenum < 0:
        normal = (-normal[0], -normal[1], -normal[2])
    length = math.sqrt(sum(c * c for c in normal))
    if length < 1e-6:
        return None
    return (normal[0] / length, normal[1] / length, normal[2] / length)


def oriented_plane(normal, start, end):
    """`normal` signed the way Nightdive writes a ladder plane.

    The convention is measured, and it is not the one AAS hands over. On all
    237 corpus ladder traversals whose two nodes differ horizontally, the
    plane's dot product with the horizontal ``start``->``end`` direction is
    ``+1.00`` (165 exactly, 43 at 0.99, the rest above 0.8): it points the way
    the climber steps off at the top, out of the wall.

    ``AAS_Reachability_Ladder`` orients the same face the other way -- it
    reaches the step-off with ``VectorMA(end, -15, plane1->normal, end)`` --
    and the dot product measured against BSPC's own reachabilities comes back
    ``-1.0`` on 619 of 621 upward ones. So :func:`ladder_plane_for` alone is
    180 degrees out, which is a field the engine steers by.

    Falls back to reversing the AAS normal when the two nodes are stacked
    vertically and there is no horizontal direction to agree with.
    """
    dx, dy = end[0] - start[0], end[1] - start[1]
    if math.hypot(dx, dy) < 1.0:
        return (-normal[0], -normal[1], -normal[2])
    if dx * normal[0] + dy * normal[1] < 0.0:
        return (-normal[0], -normal[1], -normal[2])
    return normal


def _ladder_face_plane(a, reach):
    """The unsigned plane index of a ladder reachability's face, for grouping.

    AAS stores planes in opposed pairs -- verified over 12000 pairs in the
    stock cache, ``planes[n ^ 1].normal == -planes[n].normal`` every time --
    so dropping the low bit names the wall irrespective of which side of it a
    given reachability was written from.
    """
    facenum = abs(reach.facenum)
    if not facenum or facenum >= len(a.faces):
        return None
    return a.faces[facenum].planenum & ~1


def ladder_columns(a):
    """Group ``LADDER`` reachabilities into physical ladders.

    Two of them belong to the same ladder when they lie on the same wall --
    :func:`_ladder_face_plane` -- and come within
    :data:`LADDER_COLUMN_RADIUS` of each other horizontally. Union-find over
    that relation, so a chain of chunks joins up however long the climb.

    Returns a list of ``[(area number, reachability), ...]``.
    """
    lad = [(i, r) for i in range(1, len(a.areasettings))
           for r in a.area_reachabilities(i)
           if r.travel_type == aas.TravelType.LADDER]
    parent = list(range(len(lad)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_plane = collections.defaultdict(list)
    for k, (i, r) in enumerate(lad):
        by_plane[_ladder_face_plane(a, r)].append(k)
    for plane, ks in by_plane.items():
        if plane is None:
            continue
        for x in range(len(ks)):
            for y in range(x + 1, len(ks)):
                ra, rb = lad[ks[x]][1], lad[ks[y]][1]
                near = min(math.dist(u[:2], v[:2])
                           for u in (ra.start, ra.end)
                           for v in (rb.start, rb.end))
                if near <= LADDER_COLUMN_RADIUS:
                    fx, fy = find(ks[x]), find(ks[y])
                    if fx != fy:
                        parent[fx] = fy

    groups = collections.defaultdict(list)
    for k, entry in enumerate(lad):
        groups[find(k)].append(entry)
    return list(groups.values())


def _ground_below(a, point, usable, limit):
    """``(area, nav origin)`` where a player released at `point` comes to rest.

    `point` is a player origin in AAS space. The scan starts
    :data:`LADDER_PROBE_RISE` above it so a point sitting exactly on a ground
    plane resolves to the area above rather than the solid below, and stops at
    the first grounded area whose own floor is at or under the probe -- the
    height test is what keeps a tall convex shaft from reporting the floor of
    the storey above. A liquid area is remembered as a fallback, because the
    bottom of a ladder is water often enough to matter.

    ``(0, None)`` when nothing is found.
    """
    fallback = (0, None)
    z = point[2] + LADDER_PROBE_RISE
    z = (math.floor(z / LADDER_PROBE_STEP) + 0.5) * LADDER_PROBE_STEP
    while z >= point[2] - limit:
        num = a.point_area_num((point[0], point[1], z))
        if num and num in usable:
            s = a.areasettings[num]
            if s.grounded:
                face = a.ground_face(num)
                ground = z
                if face is not None:
                    pts = a.face_points(face)
                    ground = sum(q[2] for q in pts) / len(pts)
                if ground <= z + 1.0:
                    return num, (point[0], point[1], ground - FLOOR_OFFSET)
            elif not fallback[0] and s.liquid:
                fallback = (num, (point[0], point[1], z - FLOOR_OFFSET))
        z -= LADDER_PROBE_STEP
    return fallback


def _ladder_floor(a, point, normal, offsets, usable, limit):
    """The first floor found probing sideways from `point` off a ladder face.

    See :data:`LADDER_PROBE_OFFSETS` for why the offsets are signed and why
    zero is not enough on its own.
    """
    for d in offsets:
        found = _ground_below(a, (point[0] + normal[0] * d,
                                  point[1] + normal[1] * d, point[2]),
                              usable, limit)
        if found[0]:
            return found
    return (0, None)


def ladder_pairs(a, usable):
    """One climb per physical ladder: the foot and the top of each.

    Yields ``(foot area, foot origin, top area, top origin, ladder plane,
    cost)`` in nav space, plus a counter of why the rest were dropped. The
    plane is oriented from the two points resolved *here* rather than from the
    nodes they end up on: the top point is directly out from the ladder face
    by construction, while a reused lattice node can sit off to one side and
    make :func:`oriented_plane`'s sign turn on a few degrees.

    The foot is the lowest ``start`` of the column's reachabilities dropped to
    the floor, and the top the highest ``end`` -- the two ends of what BSPC
    simulated, each resolved to a place a player stands rather than a place
    the climb passes through.
    """
    unresolved = collections.Counter()
    out = []
    for group in ladder_columns(a):
        normal = None
        for _, r in group:
            normal = ladder_plane_for(a, r)
            if normal:
                break
        if normal is None:
            unresolved["no ladder face to take a plane from"] += 1
            continue

        # A reachability leaving a ladder area starts inside it; one leaving
        # the floor at the bottom does not, so prefer the former.
        starts = [r.start for i, r in group if a.areasettings[i].ladder]
        foot = _ladder_floor(a, min(starts or [r.start for _, r in group],
                                    key=lambda p: p[2]),
                             normal, LADDER_PROBE_OFFSETS, usable,
                             LADDER_FOOT_DROP)
        if not foot[0]:
            unresolved["no floor under the ladder foot"] += 1
            continue

        # The step-off: the highest end of an upward reachability, which for
        # the top of the chain is already 16 units above the last rung and 15
        # out from the face. The floor it lands on has to be above the foot,
        # or the probe has fallen back down the shaft it just climbed.
        ends = [r.end for _, r in group if r.end[2] > r.start[2]]
        hi = max(ends or [r.end for _, r in group], key=lambda p: p[2])
        top = (0, None)
        for d in LADDER_TOP_OFFSETS:
            found = _ground_below(a, (hi[0] + normal[0] * d,
                                      hi[1] + normal[1] * d, hi[2]),
                                  usable, LADDER_TOP_DROP)
            if found[0] and found[1][2] > foot[1][2]:
                top = found
                break
        if not top[0]:
            unresolved["no floor at the ladder top"] += 1
            continue
        if top[1][2] - foot[1][2] < LADDER_MIN_CLIMB:
            unresolved[f"climb under {LADDER_MIN_CLIMB:g} units"] += 1
            continue

        cost = float(sum(r.traveltime for _, r in group)) or 1.0
        plane = oriented_plane(normal, foot[1], top[1])
        out.append((foot[0], foot[1], top[0], top[1], plane, cost))
    return out, unresolved


#: Heights to probe above a teleporter entity's origin, in AAS space.
#:
#: A ``misc_teleporter``'s origin is **already a player-origin position**, not
#: a floor position -- measured against Nightdive's files, its nav node sits at
#: exactly ``origin.z - 23.47``: xdm2's three pads at z 128/344/184 give
#: Nightdive nodes at 104.53/320.53/160.53, and its destinations likewise.
#: Since AAS space is player-origin space, the origin therefore lands *on* an
#: area's ground face, and ``AAS_PointAreaNum`` at exactly that plane resolves
#: to the solid side. Hence the epsilon first, and a short ladder after it for
#: a pad standing on a step.
_STAND_PROBES = (1.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0)

#: How far below a landing point to look for the floor, and in what steps.
#: ``AAS_Reachability_Teleport`` traces 64 units down from the destination for
#: the same reason -- a ``misc_teleporter_dest`` is often placed above the
#: floor it drops you onto.
_FALL_LIMIT = 512.0
_FALL_STEP = 8.0


def _locate_standing(a, point, usable):
    """``(area, origin)`` for a player standing at an entity's own position.

    See :data:`_STAND_PROBES` -- the position is already in player-origin
    space, so this only needs to clear the boundary plane it sits on. The
    probed point comes back too, because a push trajectory has to start from
    a position AAS agrees a player can occupy, not from the brush face.
    """
    for dz in _STAND_PROBES:
        probe = (point[0], point[1], point[2] + dz)
        num = a.point_area_num(probe)
        if num and num in usable:
            return num, probe
    return 0, point


def _locate_standing_area(a, point, usable):
    """The area a player standing at a teleporter entity's origin occupies."""
    return _locate_standing(a, point, usable)[0]


def _locate_landing_area(a, point, usable):
    """The area a player dropped at `point` ends up in.

    `point` is already a player origin. If it is in mid-air -- which a
    ``misc_teleporter_dest`` above its floor produces -- fall until a grounded
    area turns up, taking any usable area as a fallback so a destination
    inside water or on a ladder still resolves.
    """
    fallback = 0
    z = point[2]
    dropped = 0.0
    while dropped <= _FALL_LIMIT:
        num = a.point_area_num((point[0], point[1], z))
        if num and num in usable:
            if a.areasettings[num].grounded:
                return num
            fallback = fallback or num
        z -= _FALL_STEP
        dropped += _FALL_STEP
    return fallback


def teleport_pairs(a, b, usable):
    """Every teleporter in the map, as ``(src_area, src_point, dst_area,
    dst_point)``.

    Either area is 0 when AAS does not model that spot, which happens often
    enough to matter: on 10 stock maps, 6 pads and 8 destinations of 42 land
    outside AAS space. A teleporter in a recess narrower than 32 units
    disappears when BSPC expands the brushes by the player box, and a
    ``misc_teleporter_dest`` is routinely placed in open air above its floor.
    The caller decides what to do about it; the positions are always given.

    Returns (pairs, unresolved) where `unresolved` counts teleporters that
    could not be turned into a link, with the reason. This is why that count
    is reported rather than warned about: some mods resolve a teleporter's
    destination through game-mode-specific logic at runtime rather than a
    fixed ``target``, so no offline tool can pair them -- one 28-map set left
    73 of 349 teleporter entities this way.
    """
    unresolved = collections.Counter()
    if b is None:
        return [], unresolved

    by_targetname = {}
    for ent in b.entities:
        name = ent.get("targetname")
        if name and name not in by_targetname:
            by_targetname[name] = ent

    pairs = []
    for ent in b.by_classname(*TELEPORTER_CLASSNAMES):
        target = ent.get("target")
        if not target:
            unresolved["no target key"] += 1
            continue
        dest = by_targetname.get(target)
        if dest is None:
            unresolved["target names nothing"] += 1
            continue
        dest_origin = bsp.parse_vec3(dest.get("origin", ""))

        model = b.model_index(ent)
        if model is not None and model < len(b.models):
            # a brush trigger_teleport: stand in the middle of its footprint
            mins, maxs = b.model_bounds(model)
            src_point = ((mins[0] + maxs[0]) / 2.0,
                         (mins[1] + maxs[1]) / 2.0, mins[2])
        else:
            src_point = bsp.parse_vec3(ent.get("origin", ""))
        src_area = _locate_standing_area(a, src_point, usable)

        dst_area = _locate_landing_area(
            a, (dest_origin[0], dest_origin[1],
                dest_origin[2] + TELEPORT_DEST_RISE), usable)
        if src_area and src_area == dst_area:
            unresolved["source and destination in one area"] += 1
            continue
        pairs.append((src_area, src_point, dst_area, dest_origin))
    return pairs, unresolved


def push_velocity(ent):
    """The velocity a ``trigger_push`` gives a player, or None.

    ``G_SetMovedir`` (``g_utils.cpp``) reads the entity's ``angles``, with the
    ``angle`` key spawning as ``(0, angle, 0)``, and treats two of them
    specially: ``(0,-1,0)`` means straight up and ``(0,-2,0)`` straight down.
    Anything else is ``AngleVectors``' forward vector. ``trigger_push_touch``
    then scales it by ``speed * 10``.
    """
    if "angles" in ent:
        angles = bsp.parse_vec3(ent.get("angles", ""))
    else:
        angles = (0.0, bsp.float_key(ent, "angle"), 0.0)
    if angles == (0.0, -1.0, 0.0):
        movedir = (0.0, 0.0, 1.0)
    elif angles == (0.0, -2.0, 0.0):
        movedir = (0.0, 0.0, -1.0)
    else:
        pitch = math.radians(angles[0])
        yaw = math.radians(angles[1])
        movedir = (math.cos(yaw) * math.cos(pitch),
                   math.sin(yaw) * math.cos(pitch),
                   -math.sin(pitch))
    speed = bsp.float_key(ent, "speed") or PUSH_SPEED_DEFAULT
    scale = speed * PUSH_SPEED_SCALE
    velocity = (movedir[0] * scale, movedir[1] * scale, movedir[2] * scale)
    if max(abs(c) for c in velocity) < 1.0:
        return None
    return velocity


def _origin_box(mins, maxs):
    """The trigger's volume expressed in *origin* space.

    A trigger fires on a box overlap, so the set of player origins touching it
    is its own box grown by the player box reflected -- ``mins - PLAYER_MAXS``
    to ``maxs - PLAYER_MINS``.
    """
    return (tuple(mins[i] - PLAYER_MAXS[i] for i in range(3)),
            tuple(maxs[i] - PLAYER_MINS[i] for i in range(3)))


def _inside(point, box):
    lo, hi = box
    return all(lo[i] <= point[i] <= hi[i] for i in range(3))


def push_entry_point(mins, maxs, velocity):
    """Where a player enters a push trigger, given which way it pushes.

    The upwind end of the volume along the push's **dominant** horizontal axis,
    its middle on the other -- a wind tunnel is entered at one end, and a
    trigger angled five degrees off the x axis is still entered along x, not at
    the corner. Vertically the floor of the volume, or its ceiling for a
    downward push.

    A pad the player is steered off rather than aimed by -- the same
    :data:`PUSH_AIR_CONTROL_MAX_HORIZONTAL` test the landing search uses -- is
    entered in the middle, because its slight horizontal lean says nothing
    about which side a player walks on from.
    """
    point = [(mins[i] + maxs[i]) / 2.0 for i in range(3)]
    if math.hypot(velocity[0], velocity[1]) > PUSH_AIR_CONTROL_MAX_HORIZONTAL:
        axis = 0 if abs(velocity[0]) >= abs(velocity[1]) else 1
        point[axis] = maxs[axis] if velocity[axis] < 0 else mins[axis]
    point[2] = maxs[2] if velocity[2] < -1.0 else mins[2]
    return tuple(point)


def _push_column(a, point, height, usable):
    """Walk up a pad's column: ``(first open position, first usable area)``.

    The two answers are different and both are needed. The launch has to start
    from anywhere AAS calls open, mid-air included, because that is where the
    push acts; the *link* has to start from a node, so it needs the first area
    that can hold one.
    """
    launch = None
    area = 0
    dz = 0.0
    limit = max(height, 0.0) + _STAND_PROBES[-1]
    while dz <= limit:
        probe = (point[0], point[1], point[2] + dz)
        num = a.point_area_num(probe)
        if num:
            if launch is None:
                launch = probe
            if not area and num in usable:
                area = num
        dz += PUSH_COLUMN_STEP
    return launch, area


def push_launch_point(a, mins, maxs, velocity, usable=()):
    """``(entry, launch, area)`` for a push trigger.

    `entry` is where the bot has to be to get pushed, `launch` the first
    position inside the volume AAS agrees a player can occupy -- the same
    column when the trigger is roomy, and one further towards its middle when
    it is not -- and `area` the first area up that column a node can sit in.
    `launch` is None when no part of the volume is in the AAS at all.
    """
    entry = push_entry_point(mins, maxs, velocity)
    middle = [(mins[i] + maxs[i]) / 2.0 for i in range(2)]
    for frac in PUSH_ENTRY_FRACTIONS:
        probe = (entry[0] + (middle[0] - entry[0]) * frac,
                 entry[1] + (middle[1] - entry[1]) * frac, entry[2])
        launch, area = _push_column(a, probe, maxs[2] - mins[2], usable)
        if launch is not None:
            return probe, launch, area
    return entry, None, 0


def push_trajectory(a, start, velocity, box):
    """Follow a pushed player until something stops them.

    Returns ``(last position, last position still inside the trigger)``, both
    somewhere AAS agrees a player can be. While the origin is inside `box` the
    trigger re-applies its velocity every frame -- which is what makes a tall
    shaft carry a player up at a constant speed rather than launching them
    ballistically -- and outside it only gravity acts, because Quake II's air
    acceleration does nothing for a player not pressing a direction.

    Both points are wanted because a shaft has two useful answers: where the
    ride ends, and where the flight after it does. On a thin pad they are the
    same place and the caller's node-level check drops the duplicate.

    The trace is a point test, which is a *box* test here: AAS space is
    player-origin space. See :data:`PUSH_STEP`.
    """
    vx, vy, vz = velocity
    point = start
    last = inside = start
    elapsed = travelled = 0.0
    while elapsed < PUSH_MAX_TIME and travelled < PUSH_MAX_PATH:
        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        if speed < 1.0:
            break
        dt = PUSH_STEP / speed
        point = (point[0] + vx * dt, point[1] + vy * dt, point[2] + vz * dt)
        elapsed += dt
        travelled += PUSH_STEP
        if not a.point_area_num(point):
            break                     # solid, or off the end of the tree
        last = point
        if _inside(point, box):
            inside = point
            vx, vy, vz = velocity
        else:
            vz -= PUSH_GRAVITY * dt
    return last, inside


def horizontal_velocity_for_jump(zvel, start, end, gravity=PUSH_GRAVITY):
    """The horizontal speed needed to steer from `start` to `end`, or None.

    ``AAS_HorizontalVelocityForJump`` (``be_aas_move.c:1034``), reproduced:
    the flight lasts the rise ``zvel / g`` plus the fall from the apex to the
    target, and the horizontal speed is the horizontal distance over that.
    None where the target is above the apex, or needs more than
    :data:`PUSH_MAX_HORIZONTAL`.
    """
    apex = start[2] + zvel * zvel / (2.0 * gravity)
    drop = apex - end[2]
    if drop < 0:
        return None                    # the apex does not reach it
    flight = math.sqrt(drop / (0.5 * gravity)) + zvel / gravity
    if flight <= 0:
        return None
    speed = math.hypot(end[0] - start[0], end[1] - start[1]) / flight
    return None if speed > PUSH_MAX_HORIZONTAL else speed


def ground_centres(a, areas):
    """``{area number: ground face centroid}`` in AAS space, for the grounded
    areas among `areas`.

    Built once per map rather than per pad: the steered-landing search asks
    every grounded area whether a player could be thrown to it, and
    ``ground_face`` walks the area's faces and measures each one.
    """
    out = {}
    for areanum in areas:
        if not a.areasettings[areanum].grounded:
            continue
        face = a.ground_face(areanum)
        if face is not None:
            out[areanum] = _polygon_centroid(a.face_points(face))
    return out


def _steered_landings(a, launch, velocity, box, usable, exclude, centres):
    """Areas a player could steer to after a near-vertical push.

    Mirrors the second half of ``AAS_Reachability_JumpPad``: every ground face
    above the pad whose horizontal offset the player can cover inside the
    flight, confirmed by re-running the trajectory with that steering and
    checking it really lands there.
    """
    out = []
    for areanum, centre in centres.items():
        if areanum in exclude or centre[2] < launch[2]:
            continue                   # BSPC only steers upward
        speed = horizontal_velocity_for_jump(velocity[2], launch, centre)
        if speed is None or speed >= PUSH_AIR_CONTROL_SPEED:
            continue
        dx, dy = centre[0] - launch[0], centre[1] - launch[1]
        length = math.hypot(dx, dy)
        if length < 1e-3:
            continue
        steered = (dx / length * speed, dy / length * speed, velocity[2])
        landing, _ = push_trajectory(a, launch, steered, box)
        if _locate_landing_area(a, landing, usable) == areanum:
            out.append((areanum, landing))
    return out


def push_pairs(a, b, usable):
    """Every push trigger in the map, as ``(src_area, src_point, landings)``,
    where `landings` is a list of ``(dst_area, dst_point)``.

    One trigger can have several: a near-vertical pad is steered, so it has one
    landing per ground face the player could steer to, which is how Nightdive's
    own ``mgdm1`` hangs two ``PUSHER`` links off one pad.

    Returns (pads, unresolved), `unresolved` counting the triggers that could
    not be turned into a link and why. The commonest reason is honest rather
    than a defect: an arena set carries pads with ``speed`` 250000, which throw
    a player clean out of the level and land nowhere a bot could use.
    """
    unresolved = collections.Counter()
    if b is None:
        return [], unresolved

    triggers = b.by_classname(*PUSH_CLASSNAMES)
    if not triggers:
        return [], unresolved
    centres = ground_centres(a, usable)

    pads = []
    for ent in triggers:
        pad = _push_pad(a, b, ent, usable, unresolved, centres)
        if pad is not None:
            pads.append(pad)
    return pads, unresolved


def _push_pad(a, b, ent, usable, unresolved, centres):
    """``(src_area, src_point, [(dst_area, dst_point), ...])`` for one push
    trigger, or None with `unresolved` told why.

    Grouped by trigger rather than flattened because a steered pad has one
    source and many destinations, and both the work of resolving that source
    and the count of what went wrong belong to the trigger, not to each
    landing."""
    model = b.model_index(ent)
    if not model or model >= len(b.models):
        unresolved["not a brush entity"] += 1
        return None
    spawnflags = int(bsp.float_key(ent, "spawnflags"))
    if spawnflags & PUSH_SPAWNFLAG_START_OFF and not ent.get("targetname"):
        unresolved["START_OFF with nothing to switch it on"] += 1
        return None
    velocity = push_velocity(ent)
    if velocity is None:
        unresolved["no push direction"] += 1
        return None

    mins, maxs = b.model_bounds(model)
    entry, launch, src_area = push_launch_point(a, mins, maxs, velocity, usable)
    if launch is None:
        unresolved["pad volume is not in the AAS"] += 1
        return None
    box = _origin_box(mins, maxs)

    # Where the bot has to stand to be pushed: inside the pad if the trigger
    # sits on the floor, and on whatever is under it if the trigger's box
    # starts in mid-air, which is how a shaft's is usually built.
    if not src_area:
        src_area = _locate_landing_area(a, entry, usable)

    # No same-area guard, unlike teleport_pairs: an area is convex but it is
    # not small, and a push that runs the length of one still moves the bot
    # between two of its nodes. The node-level check in convert()'s offer()
    # rejects the ones that do not.
    landings = []
    landing, ridden = push_trajectory(a, launch, velocity, box)
    dst_area = _locate_landing_area(a, landing, usable)
    if dst_area:
        landings.append((dst_area, landing))
    # Where the ride itself ends, which is a different place from where the
    # flight after it does whenever the trigger is a shaft rather than a pad:
    # the retail pak's tall ones lift a player 350 to 400 units and then drop
    # them back down the same hole.
    ride_area = _locate_landing_area(a, ridden, usable)
    if ride_area and ride_area != dst_area:
        landings.append((ride_area, ridden))

    # Upward as well as near-vertical: AAS_HorizontalVelocityForJump squares
    # the launch velocity, so a downward pad would come back with a plausible
    # answer for a flight that never rises.
    if (velocity[2] > 0
            and abs(velocity[0]) <= PUSH_AIR_CONTROL_MAX_HORIZONTAL
            and abs(velocity[1]) <= PUSH_AIR_CONTROL_MAX_HORIZONTAL):
        landings += _steered_landings(a, launch, velocity, box, usable,
                                      {src_area, dst_area}, centres)

    if not landings:
        unresolved["nowhere to land"] += 1
        return None
    return src_area, entry, landings


def usable_areas(a):
    """Area numbers that can hold a node.

    Grounded areas are the backbone. Liquid areas come too -- Nightdive's files
    carry 2060 ``UNDER_WATER`` nodes, so swimming is navigated, and dropping
    them would orphan every ``SWIM`` and ``WATERJUMP`` reachability.

    And **any area BSPC gave a reachability to or from**, whatever its flags
    say. That is not a loosening: BSPC's movement simulation established that
    a player gets there, which is a stronger statement than a flag. It is also
    small and it pays -- over the 28-map arena set it adds 55 areas to 53721
    and recovers 119 reachabilities that had nowhere to attach, 62 of them
    ``ELEVATOR`` rides, because the standing spot on a raised plat is
    frequently a mover area with no ground face of its own.

    What is left out is the area a climb *passes through*: flagged ``LADDER``,
    with no ground face and no liquid, it is a slice of mid-air held up by a
    ladder brush, and a node there is a waypoint a bot cannot stand on.
    Nightdive has none -- 277 ``LADDER`` nodes in 66424, one per ladder, all of
    them at a foot. The climb over them is put back whole by
    :func:`ladder_pairs`, so this drops the waypoints and not the route.
    """
    endpoint = set()
    for i in range(1, len(a.areasettings)):
        for r in a.area_reachabilities(i):
            endpoint.add(i)
            if r.areanum:
                endpoint.add(r.areanum)
    out = []
    for i, s in enumerate(a.areasettings):
        if not i:
            continue                      # area 0 is the dummy
        if s.ladder and not (s.grounded or s.liquid):
            continue
        if s.grounded or s.liquid or i in endpoint:
            out.append(i)
    return out


#: Lattice offsets a node links to inside its own area: the 4 orthogonal
#: neighbours and the 4 diagonals. Convexity makes all 8 safe -- both
#: endpoints are inside one convex area, so the segment between them is too --
#: and the diagonals stop a lattice from forcing staircase paths across open
#: floor.
_LATTICE_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (1, -1), (-1, 1), (-1, -1))


def convert(a, b=None, heuristic=DEFAULT_HEURISTIC, version=nav3.VERSION,
            spacing=DEFAULT_SPACING, min_separation=MIN_NODE_SEPARATION,
            prune_isolated=True, stats=None):
    """Build a :class:`kexnav.nav3.NavFile` from an AAS file.

    `b` is the matching :class:`kexnav.bsp.BspFile`, needed only for nav edict
    bounding boxes; without it, elevator and train links are still emitted but
    carry no edict.
    """
    st = stats if stats is not None else Stats()
    st.areas = len(a.areas)
    st.reachabilities = len(a.reachability)

    areas = usable_areas(a)
    st.areas_used = len(areas)

    # Which areas an elevator delivers into, so their nodes can be flagged.
    elevator_targets = {r.areanum for r in a.reachability
                        if r.travel_type == aas.TravelType.ELEVATOR}

    # -- nodes: one lattice of them per area -------------------------------
    origins = []                      # node index -> nav-space origin
    flags = []                        # node index -> nav_node_flags_t
    radii = []                        # node index -> node.radius
    area_nodes = {}                   # area number -> [node index, ...]
    lattice = {}                      # (area number, i, j) -> node index
    lattice_by_area = collections.defaultdict(list)   # area -> [(i, j), ...]
    # two merge rules, with two different justifications -- see
    # MIN_NODE_SEPARATION and CLEARANCE_MERGE_RADIUS
    near = _Proximity(min_separation)
    clearance = _Proximity(min(CLEARANCE_MERGE_RADIUS, min_separation))
    node_areas = []                   # node index -> areas that use it
    walk_neighbours = _walk_neighbours(a, areas)
    for num in areas:
        indices = []
        area_flags = node_flags(a, num, elevator_targets)
        allowed = walk_neighbours[num] | {num}
        for origin, cell in area_node_origins(a, num, spacing):
            index = near.find(origins, origin, node_areas, allowed)
            if index is None:
                index = clearance.find(origins, origin)
            if index is None:
                index = len(origins)
                origins.append(origin)
                flags.append(area_flags)
                radii.append(DEFAULT_RADIUS)
                node_areas.append({num})
                near.add(index, origin)
                clearance.add(index, origin)
            else:
                # This area borrows a node that physically sits in another.
                # Only the capability flags carry over -- see
                # BORROWABLE_NODE_FLAGS.
                flags[index] |= area_flags & BORROWABLE_NODE_FLAGS
                node_areas[index].add(num)
                st.nodes_merged += 1
            if index not in indices:
                indices.append(index)
            if cell is not None:
                lattice[(num, cell[0], cell[1])] = index
                lattice_by_area[num].append(cell)
        area_nodes[num] = indices
    st.nodes_before_prune = len(origins)

    # -- links -------------------------------------------------------------
    # Gathered per source node, because firstLink is a running sum and so the
    # link array has to come out grouped by node. Keyed by (src, dst) to
    # dedupe: several reachabilities can collapse onto one node pair once the
    # endpoints are snapped, and the corpus has no duplicate node pair.
    chosen = {}

    def offer(src, dst, link_type, reach, cost, plane=None):
        """Keep the cheapest connection between a node pair.

        `plane` is set only by a synthesised ladder, which has no
        reachability to carry its traversal and describes itself instead.
        """
        if src == dst:
            return False
        key = (src, dst)
        old = chosen.get(key)
        if old is not None:
            if old[2] <= cost:
                return False
            st.dropped_duplicate += 1
        chosen[key] = (link_type, reach, cost, plane)
        return True

    # intra-area: the lattice. Cost 0 so a real reachability never loses to
    # one of these, and vice versa -- a walk across open floor is the cheapest
    # thing there is.
    for num, cells in lattice_by_area.items():
        for i, j in cells:
            index = lattice[(num, i, j)]
            for di, dj in _LATTICE_NEIGHBOURS:
                other = lattice.get((num, i + di, j + dj))
                if other is not None:
                    offer(index, other, nav3.LinkType.WALK, None, 0.0)

    # inter-area: the reachabilities
    for num in areas:
        for r in a.area_reachabilities(num):
            travel = r.travel_type
            if travel == aas.TravelType.LADDER:
                # One climb, synthesised below, replaces the whole chain of
                # them -- see the ladder constants.
                continue
            link_type = TRAVEL_TO_LINK.get(travel)
            if link_type is None:
                st.dropped_travel[_travel_name(travel)] += 1
                continue
            if r.areanum not in area_nodes:
                st.dropped_endpoint += 1
                continue
            src = _nearest(origins, area_nodes[num], _sub_z(r.start))
            dst = _nearest(origins, area_nodes[r.areanum], _sub_z(r.end))
            if src is None or dst is None or src == dst:
                st.dropped_endpoint += 1
                continue
            # traveltime is BSPC's own cost for the move, so it settles ties
            # between two ways of joining the same node pair.
            offer(src, dst, link_type, r, float(r.traveltime) or 1.0)

    # Quake II's teleporters and push triggers, neither of which BSPC ever
    # reports. Cost is deliberately high so a walkable route between the same
    # two nodes wins -- both are a last resort when they duplicate a walk.
    pairs, unresolved = teleport_pairs(a, b, area_nodes)
    for src_area, src_point, dst_area, dst_point in pairs:
        src = _entity_endpoint(a, src_area, src_point, origins, flags,
                               radii, area_nodes, spacing, offer, st)
        dst = _entity_endpoint(a, dst_area, dst_point, origins, flags,
                               radii, area_nodes, spacing, offer, st)
        if src is None:
            unresolved["pad unreachable, no node within the spacing"] += 1
            continue
        if dst is None:
            unresolved["destination unreachable, no node within the spacing"] += 1
            continue
        if offer(src, dst, nav3.LinkType.TELEPORT, None, ENTITY_LINK_COST):
            st.teleports += 1
            flags[src] |= int(nav3.NodeFlags.TELEPORTER)
    st.teleporters_unresolved.update(unresolved)

    pads, unresolved = push_pairs(a, b, area_nodes)
    for src_area, src_point, landings in pads:
        src = _entity_endpoint(a, src_area, src_point, origins, flags,
                               radii, area_nodes, spacing, offer, st)
        if src is None:
            unresolved["pad unreachable, no node within the spacing"] += 1
            continue
        made = 0
        for dst_area, dst_point in landings:
            dst = _entity_endpoint(a, dst_area, dst_point, origins, flags,
                                   radii, area_nodes, spacing, offer, st)
            if dst is None:
                continue
            if offer(src, dst, nav3.LinkType.PUSHER, None, ENTITY_LINK_COST):
                st.pushes += 1
                made += 1
        if made:
            # Nightdive sets it on the source node of 5 of its 7 one-way PUSHER
            # links, and BSPC has already set it on any node inside the pad's
            # own area -- this covers the pad whose node was merged outwards.
            flags[src] |= int(nav3.NodeFlags.PUSHER)
        else:
            unresolved["no landing a link could attach to"] += 1
    st.pushers_unresolved.update(unresolved)

    # Ladders: one link for the whole climb, between two nodes that stand at
    # its ends, because that is the only shape the format describes -- see the
    # ladder constants. Cost is BSPC's own traveltime for the chain, so a
    # walkable route between the same two nodes still wins.
    climbs, unresolved = ladder_pairs(a, set(areas))
    made_climbs = []
    for foot_area, foot_origin, top_area, top_origin, plane, cost in climbs:
        foot = _ladder_endpoint(foot_area, foot_origin,
                                node_flags(a, foot_area, elevator_targets),
                                LADDER_FOOT_REUSE, origins, flags, radii,
                                area_nodes, offer, st)
        top = _ladder_endpoint(top_area, top_origin,
                               node_flags(a, top_area, elevator_targets),
                               LADDER_TOP_REUSE, origins, flags, radii,
                               area_nodes, offer, st)
        if foot is None or top is None:
            unresolved["an end has no node to attach to"] += 1
            continue
        if not offer(foot, top, nav3.LinkType.LADDER, None, cost, plane):
            unresolved["a cheaper link already joins the two ends"] += 1
            continue
        st.ladders += 1
        made_climbs.append((foot, top, plane, cost))
        flags[foot] |= int(nav3.NodeFlags.LADDER)
        radii[foot] = LADDER_FOOT_RADIUS
    st.ladders_unresolved.update(unresolved)

    # A climb whose foot nothing else can get back to. Since the link itself
    # makes the top reachable from the foot, "the foot is reachable from the
    # top" is the same question as "are they in one strongly connected
    # component", which is one pass over the graph rather than one search per
    # ladder. See LADDER_REVERSE_WHEN_STRANDED.
    if made_climbs and LADDER_REVERSE_WHEN_STRANDED:
        component = _strong_components(chosen, len(origins))
        for foot, top, plane, cost in made_climbs:
            if component[foot] == component[top]:
                continue
            if not offer(top, foot, nav3.LinkType.LADDER, None, cost, plane):
                continue
            st.ladders_reversed += 1
            flags[top] |= int(nav3.NodeFlags.LADDER)
            radii[top] = LADDER_FOOT_RADIUS

    # -- prune, renumber ---------------------------------------------------
    live = set()
    for (src, dst) in chosen:
        live.add(src)
        live.add(dst)
    if prune_isolated:
        keep = [i for i in range(len(origins)) if i in live]
    else:
        keep = list(range(len(origins)))
    st.isolated_pruned = len(origins) - len(keep)
    renumber = {old: new for new, old in enumerate(keep)}

    per_node = collections.defaultdict(list)
    for (src, dst), (link_type, reach, _cost, plane) in chosen.items():
        per_node[renumber[src]].append(
            (renumber[dst], link_type, reach, plane))
    for links in per_node.values():
        links.sort(key=lambda t: t[0])

    # -- emit --------------------------------------------------------------
    nav = nav3.NavFile(version=version, heuristic=heuristic)
    for old in keep:
        nav.nodes.append(nav3.Node(flags=flags[old], num_links=0, first_link=0,
                                   radius=radii[old], origin=origins[old]))
    for src, node in enumerate(nav.nodes):
        node.first_link = len(nav.links)
        links = per_node.get(src, ())
        node.num_links = len(links)
        for dst, link_type, reach, plane in links:
            traversal = nav3.NO_TRAVERSAL
            if plane is not None:
                # A synthesised ladder. Its traversal restates its two nodes,
                # which is what all 272 of Nightdive's do, to the bit.
                start, end = node.origin, nav.nodes[dst].origin
                ladder = (plane if version >= nav3.VERSION_LADDER_PLANE
                          else None)
                traversal = len(nav.traversals)
                nav.traversals.append(nav3.Traversal(
                    funnel=(nav3.UNSET_COORD,) * 3,
                    start=start, end=end, ladder_plane=ladder))
            elif reach is not None and link_type in NEEDS_TRAVERSAL:
                start, end = _sub_z(reach.start), _sub_z(reach.end)
                ladder = None
                if version >= nav3.VERSION_LADDER_PLANE:
                    ladder = (0.0, 0.0, 0.0)
                traversal = len(nav.traversals)
                nav.traversals.append(nav3.Traversal(
                    funnel=funnel_for(link_type, start, end),
                    start=start, end=end, ladder_plane=ladder))
            link_index = len(nav.links)
            nav.links.append(nav3.Link(
                target=dst, type=int(link_type),
                flags=link_flags(reach) if reach is not None else DEFAULT_LINK_FLAGS,
                traversal=traversal))
            st.link_types[nav3.LinkType(link_type).name] += 1
            if reach is not None and link_type in NEEDS_EDICT:
                edict = _edict_for(b, reach, link_index)
                if edict is not None:
                    nav.edicts.append(edict)
                else:
                    st.edicts_unresolved += 1

    for n in nav.nodes:
        st.node_flags["|".join(n.flag_names) or "0"] += 1
    st.nodes = len(nav.nodes)
    st.links = len(nav.links)
    st.traversals = len(nav.traversals)
    st.edicts = len(nav.edicts)
    _check_limits(nav)
    return nav


def _strong_components(chosen, count):
    """Strongly connected component index per node, over the chosen links.

    Kosaraju, iterative on both passes, because a generated graph runs to tens
    of thousands of nodes and the recursion limit is 1000.
    """
    out = [[] for _ in range(count)]
    into = [[] for _ in range(count)]
    for src, dst in chosen:
        out[src].append(dst)
        into[dst].append(src)

    order = []
    seen = [False] * count
    for start in range(count):
        if seen[start]:
            continue
        seen[start] = True
        stack = [(start, 0)]
        while stack:
            v, i = stack.pop()
            if i < len(out[v]):
                stack.append((v, i + 1))
                w = out[v][i]
                if not seen[w]:
                    seen[w] = True
                    stack.append((w, 0))
            else:
                order.append(v)

    component = [-1] * count
    label = 0
    for v in reversed(order):
        if component[v] != -1:
            continue
        component[v] = label
        stack = [v]
        while stack:
            x = stack.pop()
            for y in into[x]:
                if component[y] == -1:
                    component[y] = label
                    stack.append(y)
        label += 1
    return component


def _walk_neighbours(a, areas):
    """For each area, the areas joined to it by a plain walk, either way.

    This is what makes merging safe at *any* distance. Two nodes 64 units
    apart could be on opposite sides of a wall -- the brush expansion only
    guarantees 32 units of clearance around solid -- so proximity alone is not
    enough to justify one area borrowing another's node. A ``WALK`` or
    ``CROUCH`` reachability between the two areas is: BSPC established that a
    player can walk from one to the other, and both nodes lie inside their own
    convex area, so the whole route is walkable.
    """
    walkable = {aas.TravelType.WALK, aas.TravelType.CROUCH}
    neighbours = collections.defaultdict(set)
    for num in areas:
        for r in a.area_reachabilities(num):
            if r.travel_type in walkable:
                neighbours[num].add(r.areanum)
                neighbours[r.areanum].add(num)
    return neighbours


class _Proximity:
    """A spatial hash answering "is there already a usable node nearby?".

    A grid of `radius`-sized cells, so a lookup touches 27 buckets. Brute
    force would be quadratic, and a generated map can run to thousands of nodes.
    """

    def __init__(self, radius):
        self.radius = radius
        self.cells = collections.defaultdict(list)

    def _cell(self, point):
        r = self.radius
        return (int(math.floor(point[0] / r)), int(math.floor(point[1] / r)),
                int(math.floor(point[2] / r)))

    def add(self, index, point):
        self.cells[self._cell(point)].append(index)

    def find(self, origins, point, node_areas=None, allowed=None):
        """The nearest node within `radius` that `allowed` may borrow, or None.

        `allowed` is a set of area numbers; a node qualifies only if some area
        already using it is in that set. Pass neither to search on distance
        alone.
        """
        cx, cy, cz = self._cell(point)
        best = None
        best_d = self.radius
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for i in self.cells.get((cx + dx, cy + dy, cz + dz), ()):
                        d = math.dist(origins[i], point)
                        if d >= best_d:
                            continue
                        if allowed is not None and not (node_areas[i] & allowed):
                            continue
                        best, best_d = i, d
        return best


#: The biggest height change a synthesised walk link will span. Quake II's
#: ``sv_stepheight`` is 18; 24 allows the half-unit floor offsets and a
#: slightly mismeasured pad without admitting a link up a wall.
MAX_SYNTHETIC_STEP = 24.0


def _ladder_endpoint(areanum, origin, area_flags, reuse, origins, flags,
                     radii, area_nodes, offer, st):
    """The node a ladder link attaches to at `origin`, creating one if need be.

    Unlike :func:`_entity_endpoint` this does not settle for the nearest node
    in the area: a ladder's traversal *is* its two node origins, so a node
    further than `reuse` away is the wrong place to start or finish the climb
    and a dedicated one goes in, walk-linked both ways to the nearest node of
    the same area. That is safe at any distance inside one area -- AAS areas
    are convex, so the segment between two points in one of them is walkable,
    the same argument :func:`_walk_neighbours` rests on.

    Nightdive's own foot nodes are dedicated in the same way: the nearest
    other node is a median 80 units off (minimum 17.2), and 126 of the 270
    have no link but the climb and one walk back.
    """
    siblings = area_nodes.get(areanum) or ()
    nearest = _nearest(origins, siblings, origin)
    if nearest is not None and math.dist(origins[nearest], origin) <= reuse:
        return nearest
    if nearest is None:
        return None
    index = len(origins)
    origins.append(origin)
    flags.append(area_flags)
    radii.append(DEFAULT_RADIUS)
    area_nodes[areanum] = list(siblings) + [index]
    offer(index, nearest, nav3.LinkType.WALK, None, 0.0)
    offer(nearest, index, nav3.LinkType.WALK, None, 0.0)
    st.nodes_synthesised += 1
    return index


def _entity_endpoint(a, areanum, point, origins, flags, radii, area_nodes,
                     spacing, offer, st):
    """The node index a synthesised link should attach to at `point`.

    Inside a usable area, that is simply the nearest node in it. Outside one,
    a node is *created* at the entity's own position and joined to the nearest
    existing node by a pair of ``WALK`` links.

    Creating it is what Nightdive does: q2ctf1's teleporter pad sits at
    ``(-320, 288, -56)``, which AAS reports as solid at every height, and
    Nightdive's own node 249 is at ``(-320.69, 281.90, -79.47)`` -- the pad
    position less :data:`FLOOR_OFFSET`, to the unit. The pad is somewhere a
    player stands by construction, so the position is trustworthy even where
    AAS lost it.

    The **walk link is inferred**, though, not derived: nothing here can trace,
    so it rests on the neighbour being within one node spacing and
    :data:`MAX_SYNTHETIC_STEP` in height. ``Stats.nodes_synthesised`` counts
    them so a report can say how much of a file leans on it.
    """
    if areanum:
        return _nearest(origins, area_nodes[areanum], point)

    origin = (point[0], point[1], point[2] - FLOOR_OFFSET)
    neighbour = _nearest(origins, range(len(origins)), origin)
    if neighbour is None:
        return None
    other = origins[neighbour]
    distance = math.dist(origin, other)
    if distance > spacing or abs(origin[2] - other[2]) > MAX_SYNTHETIC_STEP:
        return None
    if distance <= CLEARANCE_MERGE_RADIUS:
        # Close enough to be the same place, and synthesising here would break
        # the separation floor that CLEARANCE_MERGE_RADIUS exists to hold.
        return neighbour

    index = len(origins)
    origins.append(origin)
    flags.append(0)
    radii.append(DEFAULT_RADIUS)
    offer(index, neighbour, nav3.LinkType.WALK, None, 0.0)
    offer(neighbour, index, nav3.LinkType.WALK, None, 0.0)
    st.nodes_synthesised += 1
    return index


def _nearest(origins, indices, point):
    """The index in `indices` whose origin is closest to `point`."""
    best = None
    best_d = None
    for i in indices:
        o = origins[i]
        d = ((o[0] - point[0]) ** 2 + (o[1] - point[1]) ** 2
             + (o[2] - point[2]) ** 2)
        if best_d is None or d < best_d:
            best, best_d = i, d
    return best


def _travel_name(travel):
    try:
        return aas.TravelType(travel).name
    except ValueError:
        return f"<{travel}>"


def _edict_for(b, reach, link_index):
    """A nav edict for an elevator or train link, or None.

    The model number comes straight out of the reachability -- BSPC packs the
    mover's inline brush model into ``facenum`` for exactly these two travel
    types -- resolved through :func:`kexnav.bsp.canonical_model`, and the box
    out of the BSP, grown by :data:`EDICT_EXPAND`.

    The box is the mover's box *at its brush position*, which is where
    Nightdive's own edicts have it for 685 of the 987 measurable corpus
    records. The rest sit displaced, because a plat spawns lowered and a train
    spawns at its first path corner; reproducing that would mean running the
    game's spawn logic, so it is left as is.
    """
    if b is None:
        return None
    # A func_bobbing synthesised by bsp.add_train_bobbing names a duplicate
    # model that exists only to carry different bounds; the engine needs the
    # entity's real one.
    model = bsp.canonical_model(b, reach.mover_model)
    if not model or model >= len(b.models):
        return None
    mins, maxs = b.model_bounds(model)
    e = EDICT_EXPAND
    return nav3.NavEdict(
        link=link_index,
        model=model + EDICT_MODEL_BIAS,
        mins=(mins[0] - e, mins[1] - e, mins[2] - e),
        maxs=(maxs[0] + e, maxs[1] + e, maxs[2] + e))


def _check_limits(nav):
    """Refuse to return a graph too big for the format's u16 fields.

    Delegates to :func:`kexnav.validate.check_format_limits` rather than
    restating the rule -- one implementation, shared with the round-trip gate.
    The largest shipped file is base64.nav at 1167 nodes and 4138 links, so
    there is plenty of headroom, but a generated graph is finer than a
    hand-authored one and failing loudly beats writing a corrupt file.
    """
    problems = validate.check_format_limits(nav)
    if problems:
        raise ConvertError("; ".join(problems))

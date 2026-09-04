"""Minimal reader for Quake II BSP files -- the two lumps the converter needs.

AAS carries no entities and no brush model bounds, but a NAV3 file needs both:
a nav edict ties a link to a mover's *inline brush model* and its bounding box,
and the entity list is where teleporters and trains live at all -- BSPC looks
for Q3's entity names and so finds neither in a Q2 map.

Only the models and entities lumps are read; the geometry lumps are the AAS's
job. Nothing here writes.

Two idents exist in the rerelease's ``maps/``: 210 of its 222 BSPs are plain
``IBSP`` version 38, which is what Quake II always was, and 12 are ``QBSP``
version 38, an extended variant the rerelease added -- all of them ``mgu*``
mission maps, and BSPC rejects them outright. The lump directory is the same
shape in both, so this reads either; whether the *contents* of a QBSP lump are
laid out identically is not established here, and the converter needs BSPC to
have accepted the map anyway.
"""

import re
import struct
from dataclasses import dataclass
from typing import Dict, List, Tuple

IDENT = b"IBSP"
#: The rerelease's extended variant. Same header shape, same version number.
IDENT_EXTENDED = b"QBSP"
VERSION = 38

NUM_LUMPS = 19
LUMP_ENTITIES = 0
LUMP_PLANES = 1
LUMP_FACES = 6
LUMP_MODELS = 13

_HEADER = struct.Struct("<4si")
_LUMP = struct.Struct("<ii")
#: ``dmodel_t``: mins, maxs, origin, headnode, firstface, numfaces.
_MODEL = struct.Struct("<9f3i")
#: ``dplane_t``: normal, dist, type.
_PLANE = struct.Struct("<4fi")
#: ``dface_t``: planenum, side, firstedge, numedges, texinfo, styles[4], lightofs.
_FACE = struct.Struct("<Hhihh4Bi")

Vec3 = Tuple[float, float, float]


class BspError(Exception):
    pass


@dataclass
class Model:
    """An inline brush model. Model 0 is the world; 1 and up are the movers,
    referenced by entities as ``model`` ``"*1"``, ``"*2"``..."""

    mins: Vec3
    maxs: Vec3
    origin: Vec3
    headnode: int
    firstface: int
    numfaces: int


#: One ``"key" "value"`` pair inside an entity block.
_KEYVAL = re.compile(rb'"((?:[^"\\]|\\.)*)"\s*"((?:[^"\\]|\\.)*)"')


def parse_entities(blob):
    """Parse the entity lump into a list of dicts, in file order.

    The lump is the map's entity text, ``{ "key" "value" ... }`` blocks. Keys
    repeat legally -- Quake II entities may carry several ``target`` keys --
    so a repeated key becomes a space-joined value rather than being dropped,
    which is how the engine's own parser behaves for the keys that matter here.
    """
    entities = []
    depth = 0
    start = 0
    for i, ch in enumerate(blob):
        if ch == 0x7B:  # {
            if depth == 0:
                start = i + 1
            depth += 1
        elif ch == 0x7D:  # }
            depth -= 1
            if depth == 0:
                ent = {}
                for m in _KEYVAL.finditer(blob, start, i):
                    k = m.group(1).decode("latin-1")
                    v = m.group(2).decode("latin-1")
                    ent[k] = f"{ent[k]} {v}" if k in ent else v
                if ent:
                    entities.append(ent)
            elif depth < 0:
                raise BspError(f"unbalanced '}}' in the entity lump at byte {i}")
    if depth:
        raise BspError("unterminated entity block in the entity lump")
    return entities


def parse_vec3(value, default=(0.0, 0.0, 0.0)):
    """A ``"0 128 -64"`` entity value as a tuple. Returns `default` if it is
    not three numbers."""
    try:
        parts = [float(p) for p in value.replace(",", " ").split()]
    except (ValueError, AttributeError):
        return default
    return tuple(parts[:3]) if len(parts) >= 3 else default


@dataclass
class BspFile:
    ident: bytes = IDENT
    version: int = VERSION
    models: List[Model] = None
    entities: List[Dict[str, str]] = None

    def model_bounds(self, index):
        """(mins, maxs) of an inline brush model, as a nav edict needs them."""
        m = self.models[index]
        return m.mins, m.maxs

    def by_classname(self, *classnames):
        """Every entity whose ``classname`` is one of these, in file order."""
        want = set(classnames)
        return [e for e in self.entities if e.get("classname") in want]

    def model_index(self, entity):
        """The inline brush model number an entity references through its
        ``model`` key (``"*3"`` -> 3), or None if it has no brush model."""
        model = entity.get("model", "")
        if model.startswith("*"):
            try:
                return int(model[1:])
            except ValueError:
                return None
        return None


#: Lumps BSPC loads but never uses for AAS, and whose size limits the
#: rerelease's remastered maps blow past.
#:
#: ``Q2_LoadBSPFile`` copies every lump into a fixed-size array and calls
#: ``Error()`` on overflow, so ``base64``, ``city64`` and ``rdm7`` are simply
#: refused: *"exceeded max size for lump 7 size 7763148 > maxsize 3276800"*.
#: Lump 7 is the lightmaps, and ``MAX_MAP_LIGHTING`` is 0x320000 -- a 1997
#: budget against a 2023 remaster's lightmaps.
#:
#: Nothing in AAS generation reads lightmaps or vis: the areas come from the
#: brush lumps and the reachabilities from a movement simulation over them.
#: So :func:`strip_for_bspc` empties these rather than raising BSPC's limits,
#: which keeps the C tree unforked.
BSPC_UNUSED_LUMPS = (
    3,    # LUMP_VISIBILITY, MAX_MAP_VISIBILITY 0x280000
    7,    # LUMP_LIGHTING,   MAX_MAP_LIGHTING   0x320000
    16,   # LUMP_POP
)


def _repack(data, drop=(), entity_text=None, replace=None):
    """Rewrite a BSP: empty the lumps in `drop`, replace any lump given in
    `replace` (and the entity lump via `entity_text`), and repack everything
    else 4-byte aligned as a compiler would.

    The one place that writes a BSP. Every rewrite here goes through it, so
    they compose -- :func:`add_flood_seeds` and :func:`add_train_bobbing` both
    strip as well, and neither has to restate the packing.
    """
    ident, version = _HEADER.unpack_from(data, 0)
    if ident not in (IDENT, IDENT_EXTENDED):
        raise BspError(f"bad ident {ident!r}, expected {IDENT!r} or "
                       f"{IDENT_EXTENDED!r}")
    lumps = [_LUMP.unpack_from(data, _HEADER.size + i * _LUMP.size)
             for i in range(NUM_LUMPS)]

    head = bytearray(_HEADER.size + NUM_LUMPS * _LUMP.size)
    _HEADER.pack_into(head, 0, ident, version)
    body = bytearray()
    base = len(head)
    for i, (ofs, length) in enumerate(lumps):
        if i in drop:
            # a zero-length lump still needs a plausible offset
            _LUMP.pack_into(head, _HEADER.size + i * _LUMP.size, base, 0)
            continue
        if i == LUMP_ENTITIES and entity_text is not None:
            payload = bytes(entity_text)
        elif replace and i in replace:
            payload = replace[i]
        else:
            payload = data[ofs:ofs + length]
        _LUMP.pack_into(head, _HEADER.size + i * _LUMP.size,
                        base + len(body), len(payload))
        body += payload
        while len(body) % 4:
            body += b"\0"
    return bytes(head + body)


def strip_for_bspc(data, drop=BSPC_UNUSED_LUMPS):
    """Rewrite a BSP with the lumps in `drop` emptied and every ``_nofill``
    entity removed, repacking the rest.

    Returns the original bytes unchanged if neither was needed, so a map BSPC
    already accepts is handed over untouched. See :data:`NOFILL_KEY`.
    """
    ident, _ = _HEADER.unpack_from(data, 0)
    if ident not in (IDENT, IDENT_EXTENDED):
        raise BspError(f"bad ident {ident!r}, expected {IDENT!r} or "
                       f"{IDENT_EXTENDED!r}")
    lengths = [_LUMP.unpack_from(data, _HEADER.size + i * _LUMP.size)[1]
               for i in range(NUM_LUMPS)]
    ofs, length = _LUMP.unpack_from(
        data, _HEADER.size + LUMP_ENTITIES * _LUMP.size)
    entities = data[ofs:ofs + length]
    if NOFILL_KEY not in entities:
        if not any(lengths[i] for i in drop):
            return data
        return _repack(data, drop=drop)
    return _repack(data, drop=drop,
                   entity_text=drop_nofill(entities.rstrip(b"\0")) + b"\0")


#: ``dleaf_t`` -- contents, cluster, area, mins[3], maxs[3], then four index
#: pairs. The bounds are shorts, so a leaf is located to the unit.
_LEAF = struct.Struct("<ihh6h4H")
LUMP_LEAFS = 8

#: BSPC's ``MAX_MAP_ENTITIES``. Exceeding it is fatal --
#: ``ERROR: num_entities == MAX_MAP_ENTITIES`` -- so the seed count is capped
#: against the entities the map *already* has, not against a fixed number. A
#: big single-player map can carry over a thousand of its own.
BSPC_MAX_ENTITIES = 2048

#: Headroom left below the limit, since BSPC's own passes may add entities.
_ENTITY_MARGIN = 16

#: A leaf thinner than this in any axis is skipped as a seed site -- its
#: bounding-box centre is unlikely to be inside the leaf itself.
_MIN_SEED_LEAF = 4


def _entity_text(data):
    """The entity lump without its terminator, and how many more entities may
    be added before BSPC's ``MAX_MAP_ENTITIES`` is reached."""
    ofs, length = _LUMP.unpack_from(data, _HEADER.size + LUMP_ENTITIES * _LUMP.size)
    text = drop_nofill(data[ofs:ofs + length].rstrip(b"\0"))
    return bytearray(text), BSPC_MAX_ENTITIES - _ENTITY_MARGIN - text.count(b"{")


#: The key a mapper sets on an entity that sits **outside** the sealed world,
#: telling the compiler not to decide what is inside the map from it.
#:
#: Quake II's own qbsp honours it. BSPC's ``FloodEntities`` (``portals.c:843``)
#: does not -- it floods from every entity with a non-zero origin, full stop --
#: so one such entity is enough to make it report
#: ``WARNING: entity reached from outside`` / ``**** leaked ****`` and write no
#: ``.aas`` at all. ``ware1`` carries ten ``path_corner``\ s marked this way and
#: is exactly that case: the plain compile leaks and only the ``-nocsg`` retry
#: produces anything.
#:
#: Dropping the ten lets it compile plainly, which is the point -- ``-nocsg``
#: is a fallback that costs about 10% of the grounded areas. Measured on
#: ``ware1``: 885 nodes and 3315 links at 98.3% coverage becomes 814 and 2986
#: at 98.7%. A small gain in coverage, a real one in not depending on the
#: fallback, and three maps in the retail pak are affected (``ware1``,
#: ``rmine1``, ``q2dm5``).
NOFILL_KEY = b"_nofill"


def drop_nofill(text):
    """The entity text with every ``_nofill`` entity's block removed.

    Textual rather than parse-and-re-render, so every other byte of the lump
    survives untouched: the converter reads this same rewritten BSP back for
    its edicts and train segments.
    """
    if NOFILL_KEY not in text:
        return text
    out = bytearray()
    depth = 0
    start = 0
    for i, ch in enumerate(text):
        if ch == 0x7B:                # {
            if depth == 0:
                out += text[start:i]
                start = i
            depth += 1
        elif ch == 0x7D:              # }
            depth -= 1
            if depth == 0:
                block = text[start:i + 1]
                keep = True
                for m in _KEYVAL.finditer(block):
                    if m.group(1) == NOFILL_KEY and m.group(2) not in (b"0", b""):
                        keep = False
                        break
                if keep:
                    out += block
                start = i + 1
    out += text[start:]
    return bytes(out)


def empty_leaf_centres(data, limit=1000):
    """Bounding-box centres of the BSP's empty leaves.

    These are authoritative "inside the map" positions: Quake II's own
    compiler already ran its fill, so a leaf with `contents` 0 is enclosed
    space rather than the void. See :func:`add_flood_seeds` for what they are
    for.

    A leaf is convex but its bounding box is not the leaf, so a centre can
    land in a neighbour or in solid. That is harmless for the intended use --
    the seed simply fails to place -- so no further filtering is done beyond
    skipping leaves too thin to be worth trying.
    """
    ofs, length = _LUMP.unpack_from(data, _HEADER.size + LUMP_LEAFS * _LUMP.size)
    if length % _LEAF.size:
        raise BspError(f"leafs lump is {length} bytes, not a multiple of "
                       f"{_LEAF.size}")
    out = []
    for i in range(length // _LEAF.size):
        v = _LEAF.unpack_from(data, ofs + i * _LEAF.size)
        if v[0]:
            continue                       # not empty
        mins, maxs = v[3:6], v[6:9]
        if min(b - a for a, b in zip(mins, maxs)) < _MIN_SEED_LEAF:
            continue
        centre = tuple((a + b) / 2.0 for a, b in zip(mins, maxs))
        if any(centre):                    # FloodEntities skips the origin
            out.append(centre)
        if len(out) >= limit:
            break
    return out


def add_flood_seeds(data, limit=1000, drop=BSPC_UNUSED_LUMPS):
    """Rewrite a BSP with an extra point entity at each empty leaf centre.

    **What this is for.** BSPC decides which of its leaves are "outside" by
    flooding from entity origins (`FloodEntities`, `portals.c:843`) and filling
    whatever the flood did not reach. It floods the *expanded* brush tree,
    where an entity close to a surface sits inside solid and places no seed --
    so a region of the map whose only entities are lights on its walls can be
    discarded wholesale. On `mals_ladder_test`, whose rooms are joined only by
    teleporters and are therefore disjoint volumes, BSPC keeps one volume and
    fills the other: 57 of 110 non-solid leaves filled, and an AAS covering
    x -416..16 of a map that spans -432..400.

    Seeding from the BSP's own empty leaves gives every disjoint volume a
    chance at a seed. Measured: `mals_ladder_test` goes from 10 grounded areas
    to 19, `q2dm1` from 588 to 611 with 60 more reachabilities, `q2ctf1` from
    849 to 854.

    **It is an improvement, not a fix.** On `mals_ladder_test` the recovered
    areas raise the share of Nightdive's nodes with a generated node within 32
    units from 5% to 18%, and leave the within-128 figure and the spawn
    coverage unchanged at 36% and 0% -- the missing rooms stay missing.
    Whatever else is discarding them is not solved by seeding, and remains an
    open question.

    **It is not safe to use unconditionally**, which is why nothing here calls
    it by default. On `q2dm3` the seeded compile fails outright -- a seed
    reaches the outside node and BSPC reports a leak.
    `kexnav.py generate --seed-flood`
    therefore compiles both ways and keeps whichever yields more grounded
    areas, so the result is never worse than the plain compile.

    The classname is `light` deliberately: `FloodEntities` seeds from any
    entity with a non-zero origin, while every reachability pass matches on
    specific classnames, none of which is that.

    The seed count is capped against what the map already carries -- a big
    single-player map has over a thousand entities of its own, and going past
    :data:`BSPC_MAX_ENTITIES` is fatal. With no room left this returns the
    plain :func:`strip_for_bspc` output and a count of 0.
    """
    entities, room = _entity_text(data)
    if room <= 0:
        return strip_for_bspc(data, drop=drop), 0
    centres = empty_leaf_centres(data, limit=min(limit, room))
    entities += b"".join(
        b'{\n"classname" "light"\n"origin" "%d %d %d"\n}\n'
        % (int(x), int(y), int(z)) for x, y, z in centres) + b"\0"
    return _repack(data, drop=drop, entity_text=entities), len(centres)


def loads(data):
    """Parse the entities and models lumps out of a BSP."""
    if len(data) < _HEADER.size + NUM_LUMPS * _LUMP.size:
        raise BspError(f"too short to hold a header ({len(data)} bytes)")
    ident, version = _HEADER.unpack_from(data, 0)
    if ident not in (IDENT, IDENT_EXTENDED):
        raise BspError(f"bad ident {ident!r}, expected {IDENT!r} or {IDENT_EXTENDED!r}")
    if version != VERSION:
        raise BspError(f"unsupported BSP version {version}, expected {VERSION}")

    lumps = [_LUMP.unpack_from(data, _HEADER.size + i * _LUMP.size)
             for i in range(NUM_LUMPS)]
    for i, (ofs, length) in enumerate(lumps):
        if ofs < 0 or length < 0 or ofs + length > len(data):
            raise BspError(f"lump {i} at {ofs} length {length} runs outside a "
                           f"{len(data)}-byte file")

    ofs, length = lumps[LUMP_MODELS]
    if length % _MODEL.size:
        raise BspError(f"models lump is {length} bytes, not a multiple of "
                       f"{_MODEL.size}")
    models = []
    for i in range(length // _MODEL.size):
        v = _MODEL.unpack_from(data, ofs + i * _MODEL.size)
        models.append(Model(v[0:3], v[3:6], v[6:9], v[9], v[10], v[11]))

    ofs, length = lumps[LUMP_ENTITIES]
    entities = parse_entities(data[ofs:ofs + length])

    return BspFile(ident=ident, version=version, models=models, entities=entities)


def load(path):
    with open(path, "rb") as fp:
        return loads(fp.read())


#: ``func_train`` spawnflags that change how a ``path_corner`` positions it,
#: from ``g_func.cpp``. Without either, the train's *mins* corner is placed at
#: the corner's origin; ``USE_ORIGIN`` puts its entity origin there instead.
TRAIN_FIX_OFFSET = 16
TRAIN_USE_ORIGIN = 32

#: Ignore a travel segment shorter than this. A ``func_bobbing`` with a
#: near-zero ``height`` describes no movement, and BSPC substitutes its own
#: default of 32 for a zero one, which would invent travel that is not there.
_MIN_TRAVEL = 1.0

#: How far a segment may deviate from an axis and still count as axis-aligned.
_AXIS_EPSILON = 0.5


def train_travel_segments(b):
    """Every single-axis travel segment of every ``func_train``, expressed as
    the ``func_bobbing`` parameters that describe the same movement.

    Returns ``(model, origin, height, axis)`` tuples, deduplicated.

    **Why express it as a func_bobbing.** BSPC has no idea what a
    ``func_train`` is -- it handles Quake III's ``func_bobbing``, which is a
    brush oscillating along one axis, and computes the reachabilities for
    riding it in ``AAS_Reachability_FuncBobbing`` with the same movement
    simulation it uses everywhere else. A two-corner Quake II ``func_train``
    *is* that: a brush shuttling between two points. So rather than
    reimplementing the ride, this translates the train into the description
    BSPC already understands and lets BSPC decide -- including deciding
    *against* it, since it requires the position above the platform's top to
    be real open space at both extremes.

    The translation, from ``be_aas_reach.c``: BSPC takes the mover's box as
    ``model.mins/maxs`` plus the entity's ``origin``, calls its centre ``mid``,
    and oscillates ``mid`` by ``height`` along the axis that ``spawnflags``
    selects -- 1 for x, 2 for y, neither for z. So for a segment between two
    corners, ``height`` is half the distance travelled and ``origin`` is
    whatever puts ``mid`` at the midpoint of the travel.

    A train with more than two corners becomes one segment per consecutive
    pair, plus the pair that closes the loop. Segments that are not
    axis-aligned are dropped: 85% of the 4242 segments in the retail pak are
    axis-aligned, so this keeps most of them, and a diagonal or curving move
    has no ``func_bobbing`` equivalent to translate to.
    """
    by_targetname = {}
    for e in b.entities:
        name = e.get("targetname")
        if name and name not in by_targetname:
            by_targetname[name] = e

    out = []
    for train in b.by_classname("func_train"):
        model = b.model_index(train)
        if model is None or model <= 0 or model >= len(b.models):
            continue
        box = b.models[model]
        centre = tuple((box.mins[i] + box.maxs[i]) / 2.0 for i in range(3))
        flags = _int_key(train, "spawnflags")

        # walk the path_corner chain, recording where the train's box sits
        offsets = []
        seen = set()
        cur = train.get("target")
        while cur and cur in by_targetname and cur not in seen:
            seen.add(cur)
            corner = parse_vec3(by_targetname[cur].get("origin", ""))
            if flags & TRAIN_USE_ORIGIN:
                offset = corner
            else:
                offset = tuple(corner[i] - box.mins[i] for i in range(3))
                if flags & TRAIN_FIX_OFFSET:
                    # shifts the whole path by one unit, so it moves the
                    # midpoint and leaves the axis and height alone
                    offset = tuple(c - 1.0 for c in offset)
            offsets.append(offset)
            cur = by_targetname[cur].get("target")
            if len(offsets) >= MAX_PATH_CORNERS:
                break
        if len(offsets) < 2:
            continue

        pairs = list(zip(offsets, offsets[1:]))
        if len(offsets) > 2:
            pairs.append((offsets[-1], offsets[0]))     # close the loop
        for first, second in pairs:
            moving = [i for i in range(3)
                      if abs(second[i] - first[i]) > _AXIS_EPSILON]
            if len(moving) != 1:
                continue
            axis = moving[0]
            height = abs(second[axis] - first[axis]) / 2.0
            if height < _MIN_TRAVEL:
                continue
            mid = tuple(centre[i] + (first[i] + second[i]) / 2.0 for i in range(3))
            out.append((model, tuple(mid[i] - centre[i] for i in range(3)),
                        height, axis))

    seen = set()
    unique = []
    for record in out:
        key = (record[0], tuple(round(c, 2) for c in record[1]),
               round(record[2], 2), record[3])
        if key not in seen:
            seen.add(key)
            unique.append(record)
    return unique


#: A path_corner chain longer than this is a conveyor loop, not a ride.
MAX_PATH_CORNERS = 32

#: ``spawnflags`` value selecting each bob axis, from ``be_aas_reach.c``.
_BOB_AXIS_SPAWNFLAG = (1, 2, 0)


def _int_key(entity, key):
    try:
        return int(float(entity.get(key) or 0))
    except ValueError:
        return 0


#: How close to vertical a face's normal must be to count as standable,
#: matching BSPC's own ``phys_maxsteepness`` of 0.7 in ``aas_cfg.h``.
_UPWARD = 0.7


def _model_lump_offset(data):
    return _LUMP.unpack_from(data, _HEADER.size + LUMP_MODELS * _LUMP.size)[0]


def model_ride_surface(data, model):
    """The height a player riding brush model `model` would stand on.

    Not ``maxs.z``, which is what BSPC assumes for a ``func_bobbing``. A Quake
    II train is often a *car* -- a hollow box whose roof is its ``maxs.z`` and
    whose walkable floor is a face well below that -- so this returns the
    **lowest** upward-facing surface in the model, the floor you walk in onto.
    A plain slab platform has exactly one upward face, its top, and gets the
    same answer either way.

    Found by measuring against Nightdive's own nav files rather than by
    reasoning. ``q2dm2``'s train spans z 616..792 with upward faces at 636,
    648, 752 and 792; the lowest is 636, its spawn position adds 4, and BSPC
    puts a rider 24 above the surface -- so a node on it belongs at 640 + 24 -
    23.47 = **640.53**, which is Nightdive's node z for that train to the
    hundredth. ``q2dm3`` agrees.

    Falls back to ``maxs.z`` for a model with no upward face at all.
    """
    box = _MODEL.unpack_from(data, _model_lump_offset(data) + model * _MODEL.size)
    mins_z, maxs_z = box[2], box[5]
    plane_ofs, plane_len = _LUMP.unpack_from(
        data, _HEADER.size + LUMP_PLANES * _LUMP.size)
    face_ofs, _ = _LUMP.unpack_from(data, _HEADER.size + LUMP_FACES * _LUMP.size)
    num_planes = plane_len // _PLANE.size

    best = None
    first, count = box[10], box[11]
    for i in range(first, first + count):
        planenum, side = _FACE.unpack_from(data, face_ofs + i * _FACE.size)[:2]
        if not 0 <= planenum < num_planes:
            continue
        _nx, _ny, nz, dist, _type = _PLANE.unpack_from(
            data, plane_ofs + planenum * _PLANE.size)
        if side:
            nz, dist = -nz, -dist
        if nz < _UPWARD:
            continue
        # a plane whose normal is near vertical has dist == its height
        if mins_z - 1.0 <= dist <= maxs_z + 1.0 and (best is None or dist < best):
            best = dist
    return maxs_z if best is None else best


#: BSPC's ``MAX_MAP_MODELS``. Duplicate models are capped below it.
BSPC_MAX_MODELS = 1024

#: Headroom left below the model limit.
_MODEL_MARGIN = 8


def add_movers(data, trains=False, lifts=False, drop=BSPC_UNUSED_LUMPS):
    """Rewrite a BSP so BSPC computes ride reachabilities for Quake II movers
    it does not recognise. Returns ``(bytes, count)``.

    `lifts` covers ``func_plat2`` and vertical ``func_door`` -- described as
    ``func_plat``, see :func:`plat_lift_entities`. `trains` covers
    ``func_train`` -- described as ``func_bobbing``, see
    :func:`train_travel_segments`. Both kinds go into one rewrite so that a
    map with both costs a single compile.

    Each synthetic entity references a **duplicate brush model** where the
    bounds need moving or correcting, and carries **no ``origin`` key**.
    That second point is load-bearing twice over: Quake II brush models have
    absolute bounds so BSPC's default of ``(0,0,0)`` is already right, and an
    origin key would seed ``FloodEntities`` (``portals.c:843``) from a
    geometric offset rather than a position in the map -- which makes BSPC
    report ``**** leaked ****`` and write nothing at all. Because nothing here
    seeds that flood, BSPC's inside/outside decision is untouched and the area
    graph comes out identical to the plain compile; verified on q2dm1, q2dm3,
    base1, fact1, hangar2, jail3, mine2 and power1, where only the
    reachability count rises.
    """
    parsed = loads(data) if not isinstance(data, BspFile) else data
    plats = plat_lift_entities(parsed) if lifts else []
    segments = train_travel_segments(parsed) if trains else []
    if not plats and not segments:
        return strip_for_bspc(data, drop=drop), 0

    ofs, length = _LUMP.unpack_from(data, _HEADER.size + LUMP_MODELS * _LUMP.size)
    models = bytearray(data[ofs:ofs + length])
    next_model = length // _MODEL.size
    entities, entity_room = _entity_text(data)
    room = min(entity_room, BSPC_MAX_MODELS - _MODEL_MARGIN - next_model)
    if room <= 0:
        return strip_for_bspc(data, drop=drop), 0

    surfaces = {}
    def ride_surface(model):
        if model not in surfaces:
            surfaces[model] = model_ride_surface(data, model)
        return surfaces[model]

    def duplicate(model, mins, maxs):
        nonlocal models
        v = _MODEL.unpack_from(data, ofs + model * _MODEL.size)
        models += _MODEL.pack(*mins, *maxs, *v[6:9], v[9], v[10], v[11])
        return next_model + (len(models) - length) // _MODEL.size - 1

    emitted = 0
    for model, rise, travel, _classname in plats:
        if emitted >= room:
            break
        v = _MODEL.unpack_from(data, ofs + model * _MODEL.size)
        top = ride_surface(model) + rise
        reference = model
        if rise or abs(top - v[5]) > 0.01:
            reference = duplicate(model, (v[0], v[1], v[2] + rise),
                                  (v[3], v[4], top))
        entities += (b'{\n"classname" "func_plat"\n"model" "*%d"\n'
                     b'"height" "%.2f"\n}\n' % (reference, travel))
        emitted += 1

    for model, offset, height, axis in segments:
        if emitted >= room:
            break
        v = _MODEL.unpack_from(data, ofs + model * _MODEL.size)
        # BSPC puts a func_bobbing's rider at maxs.z + 24, so maxs.z has to be
        # the surface they stand on. Only that component is substituted: the
        # horizontal extents stay the real footprint, which BSPC uses for the
        # platform's edge verts, and mins.z cancels out of its arithmetic.
        reference = duplicate(
            model,
            tuple(v[i] + offset[i] for i in range(3)),
            (v[3] + offset[0], v[4] + offset[1], ride_surface(model) + offset[2]))
        entities += (b'{\n"classname" "func_bobbing"\n"model" "*%d"\n'
                     b'"height" "%.2f"\n"spawnflags" "%d"\n}\n'
                     % (reference, height, _BOB_AXIS_SPAWNFLAG[axis]))
        emitted += 1

    entities += b"\0"
    return _repack(data, drop=drop, entity_text=entities,
                   replace={LUMP_MODELS: bytes(models)}), emitted


def add_train_bobbing(data, drop=BSPC_UNUSED_LUMPS):
    """Rewrite a BSP so BSPC computes ride reachabilities for its trains.

    Returns ``(bytes, count)``. The count is 0 and the plain strip comes back
    when there is no translatable train, so the caller can skip a redundant
    compile.

    For each segment from :func:`train_travel_segments` this appends **a
    duplicate brush model** whose bounds are the train's box recentred on the
    midpoint of that segment's travel, and a ``func_bobbing`` referencing it.
    The duplicate shares ``headnode``, ``firstface`` and ``numfaces`` with the
    original, so it is the same geometry under different bounds; BSPC reads
    those bounds through ``CM_ModelBounds`` and takes their centre as the
    point to oscillate.

    **Why a duplicate model instead of an ``origin`` key**, which is how a
    real ``func_bobbing`` would express the same offset: because that key is
    fatal here. ``FloodEntities`` (``portals.c:843``) seeds BSPC's
    inside/outside decision from *any* entity with a non-zero ``origin``, and
    a bobbing entity's origin is a geometric offset rather than a position in
    the map -- so it seeds from the void and BSPC reports ``**** leaked ****``
    and writes nothing at all. Measured: with the ``origin`` form, the compile
    failed on ``badlands``, ``boss1``, ``cool1``, ``fact1`` and ``fact2``, five
    of the first six maps tried. Encoding the offset in the bounds means the
    entity needs no ``origin`` key, so it seeds nothing.
    """
    return add_movers(data, trains=True, drop=drop)


def canonical_model(b, index):
    """The lowest-numbered model sharing `index`'s geometry.

    :func:`add_train_bobbing` appends duplicate models that differ from the
    original only in their bounds, so a reachability can name a synthetic one.
    Matching on ``headnode`` maps it back to the entity's real brush model --
    which is what a nav edict has to carry, since the engine looks the mover
    up by it.
    """
    if not 0 <= index < len(b.models):
        return index
    head = b.models[index].headnode
    for i, m in enumerate(b.models):
        if m.headnode == head:
            return i
    return index


#: ``func_door`` keys, from ``g_func.cpp``. Its travel is
#: ``|movedir . size| - lip`` along the direction ``angle`` selects, where the
#: two special angles mean vertical.
DOOR_ANGLE_UP = -1
DOOR_ANGLE_DOWN = -2
DOOR_LIP_DEFAULT = 8.0

#: ``func_plat2``'s lip defaults to 0, not 8 -- ``SP_func_plat2`` has the
#: ``st.lip`` default commented out -- and when ``height`` is given it
#: subtracts the lip from it, which ``func_plat`` does not.
PLAT2_LIP_DEFAULT = 0.0

#: ``func_water``'s lip likewise: ``SP_func_water`` sets no default, so an
#: unset ``lip`` is 0 rather than a door's 8. It is otherwise a ``func_door``
#: exactly -- same ``G_SetMovedir``, same ``|movedir . size| - lip`` travel.
WATER_LIP_DEFAULT = 0.0


def float_key(entity, key, default=0.0):
    """A numeric entity key, or `default` when it is absent or unparsable."""
    try:
        return float(entity.get(key))
    except (TypeError, ValueError):
        return default


def plat_lift_entities(b):
    """Movers BSPC's ``func_plat`` pass could handle but never sees.

    Returns ``(model, rise, travel, classname)`` tuples, where `rise` is how
    far above its brush position the mover's *top* position sits and `travel`
    is the distance between its two positions.

    ``AAS_Reachability_Elevator`` matches ``classname`` against ``"func_plat"``
    with ``strcmp``, so two very common Quake II movers get no ride
    reachability at all:

    * **``func_plat2``**, the Rogue variant. Geometrically the same mover --
      ``pos1`` is the brush position and ``pos2`` is below it -- and
      Nightdive's own files hang **199** ``ELEVATOR`` links off one. Only the
      lip arithmetic differs, so this computes the travel and states it
      outright rather than leaving BSPC to infer it.
    * **a vertical ``func_door``**, which is how Quake II builds a great many
      lifts. Nightdive's files hang **78** ``ELEVATOR`` links and 7 ``TRAIN``
      links off one. ``angle`` -1 opens upward from the brush position, -2
      opens downward from it; horizontal doors are not lifts and are skipped.
    * **a vertical ``func_water``**, which ``SP_func_water`` builds as a door
      in every respect that matters here. 78 of the retail pak's 117 are
      vertical, and a rising water block is ridden the same way a plat is --
      Nightdive's own edicts name a ``func_water`` too. Only the lip default
      differs.

    Every vertical mover in both the retail pak and the 28-map arena set
    selects its direction with the ``angle`` key rather than ``angles``: 1345
    ``func_door``, 78 ``func_water``, 49 ``func_plat2``, 0 the other way. So
    reading ``angle`` alone is not a shortcut, it is the whole population.

    Most vertical doors are *doors*, retracting into a ceiling, not lifts --
    1345 of the pak's 2557 are vertical while Nightdive treats only about 85 as
    ridable. Nothing here tries to tell them apart, because BSPC already does:
    its elevator pass wants a grounded or swimmable area near *both* plat
    positions, and a door that vanishes into a ceiling has nothing walkable at
    the top.
    """
    out = []
    for ent in b.entities:
        classname = ent.get("classname")
        model = b.model_index(ent)
        if model is None or model <= 0 or model >= len(b.models):
            continue
        box = b.models[model]
        size_z = box.maxs[2] - box.mins[2]

        if classname == "func_plat2":
            lip = float_key(ent, "lip", PLAT2_LIP_DEFAULT)
            height = float_key(ent, "height")
            travel = (height - lip) if height else (size_z - lip)
            rise = 0.0                       # the brush sits at the top
        elif classname in ("func_door", "func_water"):
            angle = float_key(ent, "angle", 0.0)
            if angle not in (DOOR_ANGLE_UP, DOOR_ANGLE_DOWN):
                continue
            lip = float_key(ent, "lip", DOOR_LIP_DEFAULT
                            if classname == "func_door" else WATER_LIP_DEFAULT)
            travel = size_z - lip
            # -1 opens upward, so the open position is the top one and it is
            # `travel` above the brush; -2 opens downward and the brush
            # position is already the top
            rise = travel if angle == DOOR_ANGLE_UP else 0.0
        else:
            continue

        if travel < _MIN_TRAVEL:
            continue
        out.append((model, rise, travel, classname))
    return out


def add_plat_lifts(data, drop=BSPC_UNUSED_LUMPS):
    """Rewrite a BSP with a ``func_plat`` per mover from
    :func:`plat_lift_entities`, so BSPC computes elevator reachabilities.

    Returns ``(bytes, count)``.

    BSPC reads a plat's ride height as ``maxs.z`` of the model bounds at its
    *top* position, so where those bounds need moving or correcting -- a door
    that opens upward, or a mover whose walkable face is not its box top, see
    :func:`model_ride_surface` -- a duplicate model carries the corrected
    bounds and the synthetic entity references that instead.

    The synthetic entity deliberately carries **no ``origin`` key**: Quake II
    brush models have absolute bounds, so BSPC's default of ``(0,0,0)`` is
    already right, and an origin key would seed ``FloodEntities`` from a point
    that need not be inside the map. That is the same trap
    :func:`add_train_bobbing` documents.
    """
    return add_movers(data, lifts=True, drop=drop)


#: Brush entities that seal a passage in the AAS but should not.
#:
#: ``AAS_PositionBrush`` (``aas_map.c:705``) assigns ``CONTENTS_MOVER`` -- which
#: *replaces* ``CONTENTS_SOLID``, so the volume stops being a wall and becomes
#: an ordinary area flagged as occupied by a mover -- to exactly one classname:
#: ``func_door``. Every other brush entity keeps its solid contents, so BSPC
#: bakes it into the AAS as a permanent wall wherever the compiler left it.
#:
#: For these two that is plainly wrong. A ``func_door_rotating`` swings out of
#: its own volume and a ``func_door_secret`` slides out of its own -- both are
#: doorways, and there are 448 and 13 of them in the retail pak, plus more
#: again across community maps. Presenting them as ``func_door`` is a
#: one-word rewrite that gets BSPC to treat their volume the way it already
#: treats a sliding door's.
#:
#: The other solid brush entities are left alone deliberately: ``func_wall``,
#: ``func_explosive`` and ``func_button`` really are walls until something
#: happens to them, and ``func_plat`` and ``func_train`` are handled by
#: describing their *movement* instead -- see :func:`add_movers`.
DOOR_MOVER_CLASSNAMES = ("func_door_rotating", "func_door_secret")


def mark_doors_as_movers(data, classnames=DOOR_MOVER_CLASSNAMES,
                         drop=BSPC_UNUSED_LUMPS):
    """Rewrite a BSP with `classnames` renamed to ``func_door``, so BSPC stops
    treating their volume as solid. Returns ``(bytes, count)``.

    Done as a byte substitution on the exact pair ``"classname" "<name>"``
    rather than by rebuilding the entity lump, because rebuilding it would
    have to re-emit keys that legally repeat -- a Quake II entity may carry
    several ``target`` keys -- and :func:`parse_entities` joins those. All 477
    occurrences across the two corpora' 127 affected maps use that exact
    spelling, so the substitution is unambiguous.

    **This one can leave the area graph smaller**, unlike the rewrites in
    :func:`add_movers`, because a door that stops being solid also stops being
    something to stand on: measured over seven maps, five gained a lot
    (``xsewer1`` +162 grounded areas and +1049 reachabilities, ``jail4`` +43
    and +546) while ``boss1`` lost 4 areas and ``rammo1`` 5. So the caller has
    to compile it alongside the plain version and keep the better -- which
    ``kexnav.py generate`` does.

    **And it does not pay off, which is why it is off by default.** Those extra
    areas and reachabilities are largely *inside* the door volumes, and once
    the converter has merged nodes the graph that comes out is no better
    connected: over seven maps the component count went from 60 to 64 and the
    share of Nightdive node pairs mutually reachable fell from 53.2% to 52.5%,
    with ``rsewer2``'s largest component dropping from 370 nodes to 321. Node
    coverage of Nightdive's own nodes stayed flat or slipped by a node or two.
    Worth keeping because the underlying finding is real and someone testing
    in-game could judge it properly, but not worth turning on blind.
    """
    ofs, length = _LUMP.unpack_from(data, _HEADER.size + LUMP_ENTITIES * _LUMP.size)
    text = data[ofs:ofs + length]
    count = 0
    for name in classnames:
        target = b'"classname" "%s"' % name.encode()
        count += text.count(target)
        text = text.replace(target, b'"classname" "func_door"')
    if not count:
        return strip_for_bspc(data, drop=drop), 0
    return _repack(data, drop=drop, entity_text=text), count

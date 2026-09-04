"""Reader/writer for the KEX Quake II rerelease bot navigation format.

The engine's closed-source "Nav Manager" loads one of these per map from
``bots/navigation/<mapname>.nav``. It is a flat waypoint graph -- despite the
name, not a navmesh -- of fixed-size little-endian records with no compression,
no alignment padding and no checksum:

    header      24B  'NAV3', u32 version, u32 numNodes, u32 numLinks,
                     u32 numTraversals, f32 heuristic
    node[]       8B  u16 flags, u16 numLinks, u16 firstLink, u16 radius
    origin[]    12B  vec3, one per node, parallel to node[]
    link[]       6B  u16 target, u8 type, u8 flags, u16 traversal (0xFFFF = none)
    traversal[] 48B  vec3 funnel, vec3 start, vec3 end, vec3 ladderPlane
                36B  ... without ladderPlane below version 4
    u32 numEdicts
    edict[]     30B  u16 link, u32 model, vec3 mins, vec3 maxs
                     ... model is the inline brush model index *plus one*

The layout was recovered from the 174 nav files shipped in baseq2/pak0.pak and
cross-checked against the counts the engine prints while loading one. The
*names and semantics* below come from q2pro-ng's independent open-source
loader (``src/server/nav.c``), which had to get them right for its pathing to
work. Two of them were re-verified against the shipped corpus rather than
taken on trust -- the ladder plane and the pre-v6 team bits, both exact.

This model mirrors the file one-for-one -- parallel arrays, raw indices, no
resolution of links into nodes, enum fields left as plain ints -- so that a
parse/serialise cycle is byte-exact and can be used to prove the format
understanding before anything is generated from scratch. Interpretation
belongs in the layer above; the enums here are for naming, never coercion,
because a file may legally carry a value no enum lists.
"""

import struct
from dataclasses import dataclass, field
from enum import IntEnum, IntFlag
from typing import List, Optional, Tuple

MAGIC = b"NAV3"

#: Written by :func:`dumps` unless a NavFile says otherwise.
VERSION = 6

#: Versions this module models. 4, 5 and 6 share one record layout -- those
#: bumps added link types and node flags, not fields. 2 and 3 differ in exactly
#: one respect: their traversals have no ladder plane and so are 36 bytes.
#: The engine still loads every one of them.
SUPPORTED_VERSIONS = (2, 3, 4, 5, 6)

#: Version that added ``Traversal.ladder_plane``.
VERSION_LADDER_PLANE = 4

NO_TRAVERSAL = 0xFFFF
UNSET_COORD = 1e30


class LinkType(IntEnum):
    """``nav_link_type_t``. Values outside this enum are legal on disk."""

    WALK = 0
    LONG_JUMP = 1
    TELEPORT = 2
    WALK_OFF_LEDGE = 3
    PUSHER = 4
    BARRIER_JUMP = 5
    ELEVATOR = 6
    TRAIN = 7
    MANUAL_LONG_JUMP = 8
    CROUCH = 9
    LADDER = 10
    MANUAL_BARRIER_JUMP = 11
    PIVOT_AND_JUMP = 12
    ROCKET_JUMP = 13
    UNKNOWN = 14


class NodeFlags(IntFlag):
    """``nav_node_flags_t``."""

    TELEPORTER = 1 << 0
    PUSHER = 1 << 1
    ELEVATOR = 1 << 2
    LADDER = 1 << 3
    UNDER_WATER = 1 << 4
    CHECK_FOR_HAZARD = 1 << 5
    CHECK_HAS_FLOOR = 1 << 6
    CHECK_IN_SOLID = 1 << 7
    NO_MONSTERS = 1 << 8
    CROUCH = 1 << 9
    NO_POI = 1 << 10
    CHECK_IN_LIQUID = 1 << 11
    CHECK_DOOR_LINKS = 1 << 12
    DISABLED = 1 << 13


class LinkFlags(IntFlag):
    """``nav_link_flags_t``.

    Note bits 2 and 3: below version 6 these were the yellow and green team
    bits, which is why v4/v5 files carry flags value 15 (all four teams) where
    v6 files carry 3. The engine strips them for pre-v6 files.
    """

    TEAM_RED = 1 << 0
    TEAM_BLUE = 1 << 1
    EXIT_AT_TARGET = 1 << 2
    WALK_ONLY = 1 << 3
    EASE_INTO_TARGET = 1 << 4
    INSTANT_TURN = 1 << 5
    DISABLED = 1 << 6


#: ``TEAM_RED | TEAM_BLUE`` -- the flags value on almost every stock link.
ALL_TEAMS = LinkFlags.TEAM_RED | LinkFlags.TEAM_BLUE

_HEADER = struct.Struct("<4sIIIIf")
_NODE = struct.Struct("<4H")
_VEC3 = struct.Struct("<3f")
_LINK = struct.Struct("<HBBH")
_TRAVERSAL = struct.Struct("<12f")
_TRAVERSAL_NO_LADDER = struct.Struct("<9f")
_COUNT = struct.Struct("<I")
_EDICT = struct.Struct("<HI6f")

Vec3 = Tuple[float, float, float]


def _name(enum_cls, value):
    try:
        return enum_cls(value).name
    except ValueError:
        return f"<{value}>"


def _flag_names(flag_cls, value):
    if not value:
        return []
    names = [f.name for f in flag_cls if value & f]
    leftover = value & ~sum(int(f) for f in flag_cls)
    if leftover:
        names.append(f"<{leftover:#x}>")
    return names


class NavError(Exception):
    pass


class UnsupportedVersion(NavError):
    """Raised for a nav file whose version this module does not model."""

    def __init__(self, version):
        super().__init__(f"unsupported version {version}, "
                         f"expected one of {SUPPORTED_VERSIONS}")
        self.version = version


def traversal_struct(version):
    """The traversal record layout for a given file version."""
    return _TRAVERSAL if version >= VERSION_LADDER_PLANE else _TRAVERSAL_NO_LADDER


@dataclass
class Node:
    flags: int
    num_links: int
    first_link: int
    radius: int
    origin: Vec3

    @property
    def flag_names(self):
        return _flag_names(NodeFlags, self.flags)


@dataclass
class Link:
    target: int
    type: int
    flags: int
    traversal: int

    @property
    def has_traversal(self):
        return self.traversal != NO_TRAVERSAL

    @property
    def type_name(self):
        return _name(LinkType, self.type)

    @property
    def flag_names(self):
        return _flag_names(LinkFlags, self.flags)


@dataclass
class Traversal:
    funnel: Vec3
    start: Vec3
    end: Vec3
    #: Only present from version 4 on, and non-zero only for ladder links.
    ladder_plane: Optional[Vec3] = None

    @property
    def has_funnel(self):
        """Whether ``funnel`` holds a real point.

        There are *two* spellings of unset. ``1e30`` in every component is the
        documented one, on 8353 corpus traversals. An all-zero vector is the
        other, on 1412 -- and it is not a point at the world origin: all 98 of
        ``badlands.nav``'s ledge funnels read (0,0,0), and the zero form
        accounts for every ``WALK_OFF_LEDGE`` traversal in the corpus that
        appeared to carry one. Both mean the same thing to a reader.
        """
        if all(c >= UNSET_COORD for c in self.funnel):
            return False
        return any(self.funnel)


@dataclass
class NavEdict:
    #: Index of the link this entity belongs to.
    link: int
    #: The mover's inline brush model index, **plus one** -- so 0 would mean
    #: "no model". Verified on all 1120 corpus edicts: ``model - 1`` is always
    #: an index a brush entity owns in the matching BSP, and the owner is
    #: always a mover or interactive brush.
    model: int
    mins: Vec3
    maxs: Vec3


@dataclass
class NavFile:
    version: int = VERSION
    heuristic: float = 0.8
    nodes: List[Node] = field(default_factory=list)
    links: List[Link] = field(default_factory=list)
    traversals: List[Traversal] = field(default_factory=list)
    edicts: List[NavEdict] = field(default_factory=list)

    def node_links(self, index):
        """The links belonging to node `index`, sliced out of the flat array."""
        node = self.nodes[index]
        return self.links[node.first_link:node.first_link + node.num_links]

    def byte_size(self):
        return (_HEADER.size
                + len(self.nodes) * (_NODE.size + _VEC3.size)
                + len(self.links) * _LINK.size
                + len(self.traversals) * traversal_struct(self.version).size
                + _COUNT.size
                + len(self.edicts) * _EDICT.size)


def loads(data):
    """Parse a nav file. Raises NavError on anything unexpected, including
    trailing bytes -- a silent partial parse would defeat the point."""
    if len(data) < _HEADER.size:
        raise NavError(f"too short to hold a header ({len(data)} bytes)")

    magic, version, num_nodes, num_links, num_traversals, heuristic = _HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise NavError(f"bad magic {magic!r}, expected {MAGIC!r}")
    if version not in SUPPORTED_VERSIONS:
        raise UnsupportedVersion(version)

    nav = NavFile(version=version, heuristic=heuristic)
    trav_struct = traversal_struct(version)
    off = _HEADER.size

    expected = (_HEADER.size
                + num_nodes * (_NODE.size + _VEC3.size)
                + num_links * _LINK.size
                + num_traversals * trav_struct.size)
    if len(data) < expected + _COUNT.size:
        raise NavError(f"truncated: header promises {num_nodes} nodes / {num_links} links / "
                       f"{num_traversals} traversals, which needs at least "
                       f"{expected + _COUNT.size} bytes, file has {len(data)}")

    raw_nodes = []
    for _ in range(num_nodes):
        raw_nodes.append(_NODE.unpack_from(data, off))
        off += _NODE.size
    for flags, node_links, first_link, radius in raw_nodes:
        origin = _VEC3.unpack_from(data, off)
        off += _VEC3.size
        nav.nodes.append(Node(flags, node_links, first_link, radius, origin))

    for _ in range(num_links):
        target, link_type, flags, traversal = _LINK.unpack_from(data, off)
        off += _LINK.size
        nav.links.append(Link(target, link_type, flags, traversal))

    for _ in range(num_traversals):
        v = trav_struct.unpack_from(data, off)
        off += trav_struct.size
        ladder = v[9:12] if version >= VERSION_LADDER_PLANE else None
        nav.traversals.append(Traversal(v[0:3], v[3:6], v[6:9], ladder))

    num_edicts, = _COUNT.unpack_from(data, off)
    off += _COUNT.size
    need = off + num_edicts * _EDICT.size
    if len(data) < need:
        raise NavError(f"truncated: {num_edicts} nav edicts need {need} bytes, file has {len(data)}")
    for _ in range(num_edicts):
        link, model, *box = _EDICT.unpack_from(data, off)
        off += _EDICT.size
        nav.edicts.append(NavEdict(link, model, tuple(box[0:3]), tuple(box[3:6])))

    if off != len(data):
        raise NavError(f"{len(data) - off} trailing bytes after the last nav edict "
                       f"(parsed {off} of {len(data)})")
    return nav


def dumps(nav):
    """Serialise back to bytes. Node link ranges are written as stored, not
    recomputed, so that a file that violates the usual contiguity invariant
    still round-trips."""
    if nav.version not in SUPPORTED_VERSIONS:
        raise UnsupportedVersion(nav.version)

    out = bytearray()
    out += _HEADER.pack(MAGIC, nav.version, len(nav.nodes), len(nav.links),
                        len(nav.traversals), nav.heuristic)
    for n in nav.nodes:
        out += _NODE.pack(n.flags, n.num_links, n.first_link, n.radius)
    for n in nav.nodes:
        out += _VEC3.pack(*n.origin)
    for l in nav.links:
        out += _LINK.pack(l.target, l.type, l.flags, l.traversal)
    if nav.version >= VERSION_LADDER_PLANE:
        for t in nav.traversals:
            out += _TRAVERSAL.pack(*t.funnel, *t.start, *t.end,
                                   *(t.ladder_plane or (0.0, 0.0, 0.0)))
    else:
        for t in nav.traversals:
            out += _TRAVERSAL_NO_LADDER.pack(*t.funnel, *t.start, *t.end)
    out += _COUNT.pack(len(nav.edicts))
    for e in nav.edicts:
        out += _EDICT.pack(e.link, e.model, *e.mins, *e.maxs)
    return bytes(out)


def load(path):
    with open(path, "rb") as fp:
        return loads(fp.read())


def dump(nav, path):
    with open(path, "wb") as fp:
        fp.write(dumps(nav))

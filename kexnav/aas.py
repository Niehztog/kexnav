"""Reader for the AAS (Area Awareness System) files BSPC compiles from a BSP.

This is the input side of the converter. BSPC decomposes a map's empty space
into convex areas and computes *reachabilities* between them -- one record per
possible move, tagged with the travel type needed to make it (walk, jump,
ladder, elevator, teleport...). That graph is what becomes a NAV3 waypoint
graph; see the wiki's GENERATING page for the type mapping.

The format is documented in the Quake III source (``aasfile.h``): a header of
14 lump directory entries followed by the lumps, all little-endian fixed-size
records. Note that BSPC's own source tree carries *two* copies of that header
and the one to believe is ``deps/botlib/aasfile.h``, not ``bspc/aasfile.h`` --
the record layouts are identical but the bspc-local copy is older and is
missing the names for six content bits, two area flags, one face flag and the
packed mover model number, all of which BSPC does write. Three further things
neither header spells out, all confirmed against files on this machine:

**The version 5 lump directory is obfuscated.** ``AAS_DData``
(``aas_file.c:230``) XORs every byte of the header from offset 8 to
``sizeof(aas_header_t)`` with ``i * 119``, ``i`` counting from that offset.
Read the lumps without undoing it and you get garbage offsets.

**``facenum`` and ``edgenum`` are a per-travel-type payload**, not always
indices into their lumps. For an elevator they carry the mover's model number
and the plat's travel height; for a ladder, a *signed* face index; for a jump
pad, velocities. :class:`Reachability` names each reading. Validating them as
face and edge indices is simply wrong, and the elevator case is how the
converter learns which brush model a nav edict must point at.

**Version 3 -- Mr. Elusive's 1999 Q2 BSPC -- is plain and has no
``bspchecksum``**, so its header is 120 bytes rather than 124, and no XOR is
applied. Every record this module needs is byte-identical between the two
versions; the one struct that did grow is ``aas_cluster_t``, 12 bytes in v3
against 16 in v5, Q3 having inserted ``numreachabilityareas`` after
``numareas``; see :func:`cluster_struct`. Those files are read here for
cross-checking only; the converter's input is Q3 BSPC v5 output, because
the Q2 tool computed no reachabilities at all.

Same discipline as :mod:`kexnav.nav3`: the model mirrors the file. Lumps stay
parallel arrays of raw indices, ``traveltype`` keeps its team flags packed in,
and the enums are for *naming*, never for coercion.
"""

import struct
from dataclasses import dataclass, field
from enum import IntEnum, IntFlag
from typing import List, Optional, Tuple

#: ``AASID``, which is ``('S'<<24)+('A'<<16)+('A'<<8)+'E'`` -- so on disk,
#: little-endian, it reads as these four bytes.
IDENT = b"EAAS"

#: Q3 BSPC output. What the converter consumes.
VERSION = 5
#: Q3's own previous version. Same record layout as 5.
VERSION_OLD = 4
#: Mr. Elusive's Q2 BSPC. Plain header, no ``bspchecksum``.
VERSION_GLADIATOR = 3

SUPPORTED_VERSIONS = (3, 4, 5)

#: Lump directory entries, in order.
NUM_LUMPS = 14
(LUMP_BBOXES, LUMP_VERTEXES, LUMP_PLANES, LUMP_EDGES, LUMP_EDGEINDEX,
 LUMP_FACES, LUMP_FACEINDEX, LUMP_AREAS, LUMP_AREASETTINGS,
 LUMP_REACHABILITY, LUMP_NODES, LUMP_PORTALS, LUMP_PORTALINDEX,
 LUMP_CLUSTERS) = range(NUM_LUMPS)

LUMP_NAMES = ("bboxes", "vertexes", "planes", "edges", "edgeindex", "faces",
              "faceindex", "areas", "areasettings", "reachability", "nodes",
              "portals", "portalindex", "clusters")

#: ``AAS_DData`` XOR multiplier, from ``aas_file.c:236``.
DDATA_KEY = 119
#: Header bytes before the obfuscated region -- ident and version.
DDATA_OFFSET = 8

#: ``traveltype`` carries team flags in its top byte.
TRAVELTYPE_MASK = 0xFFFFFF
TRAVELFLAG_NOTTEAM1 = 1 << 24
TRAVELFLAG_NOTTEAM2 = 2 << 24

#: ``contents`` carries the mover's inline brush model number in its top byte.
AREACONTENTS_MODELNUMSHIFT = 24
AREACONTENTS_MAXMODELNUM = 0xFF

#: Mask ``Reachability.facenum`` with this to recover a mover model number --
#: ``be_ai_move.c`` does exactly this on the elevator and func_bob paths.
MOVER_MODEL_MASK = 0xFFFF


class TravelType(IntEnum):
    """``traveltype`` after masking with :data:`TRAVELTYPE_MASK`.

    Values 13 upward are Q3 moves that Q2 geometry cannot produce, kept for
    naming completeness. Values outside this enum are legal on disk.
    """

    INVALID = 1
    WALK = 2
    CROUCH = 3
    BARRIERJUMP = 4
    JUMP = 5
    LADDER = 6
    WALKOFFLEDGE = 7
    SWIM = 8
    WATERJUMP = 9
    TELEPORT = 10
    ELEVATOR = 11
    ROCKETJUMP = 12
    BFGJUMP = 13
    GRAPPLEHOOK = 14
    DOUBLEJUMP = 15
    RAMPJUMP = 16
    STRAFEJUMP = 17
    JUMPPAD = 18
    FUNCBOB = 19


class AreaFlags(IntFlag):
    """``aas_areasettings_t.areaflags``.

    ``WEAPONJUMP`` is not in either ``aasfile.h``; it is defined locally in
    ``be_aas_reach.c:63`` and set during the reachability pass, so it appears
    in files but not in the format header.
    """

    GROUNDED = 1
    LADDER = 2
    LIQUID = 4
    DISABLED = 8
    BRIDGE = 16
    WEAPONJUMP = 8192


class AreaContents(IntFlag):
    """``aas_areasettings_t.contents``."""

    WATER = 1
    LAVA = 2
    SLIME = 4
    CLUSTERPORTAL = 8
    TELEPORTAL = 16
    ROUTEPORTAL = 32
    TELEPORTER = 64
    JUMPPAD = 128
    DONOTENTER = 256
    VIEWPORTAL = 512
    MOVER = 1024
    NOTTEAM1 = 2048
    NOTTEAM2 = 4096


class FaceFlags(IntFlag):
    """``aas_face_t.faceflags``."""

    SOLID = 1
    LADDER = 2
    GROUND = 4
    GAP = 8
    LIQUID = 16
    LIQUIDSURFACE = 32
    BRIDGE = 64


class PresenceType(IntFlag):
    """``presencetype`` -- how a bot can occupy an area."""

    NONE = 1
    NORMAL = 2
    CROUCH = 4


_IDENT = struct.Struct("<4s")
_LUMP = struct.Struct("<ii")
_BBOX = struct.Struct("<ii6f")
_VERTEX = struct.Struct("<3f")
_PLANE = struct.Struct("<4fi")
_EDGE = struct.Struct("<2i")
_INDEX = struct.Struct("<i")
_FACE = struct.Struct("<6i")
_AREA = struct.Struct("<3i9f")
_AREASETTINGS = struct.Struct("<7i")
#: 42 bytes of fields in a 44-byte struct -- the trailing 2 are C padding
#: after ``unsigned short traveltime``, and BSPC writes them as zero.
_REACHABILITY = struct.Struct("<3i6fiH2x")
_NODE = struct.Struct("<3i")
_PORTAL = struct.Struct("<5i")
_CLUSTER = struct.Struct("<4i")
#: Version 3 has no ``numreachabilityareas``. Established from the shipped
#: Gladiator files rather than from a header: on q2dm1, q2ctf1 and q2ctf4 the
#: 12-byte reading is the one whose ``firstportal`` accumulates by
#: ``numportals`` to exactly ``portalindexsize``, and whose record 0 is the
#: expected zero dummy. At 16 bytes the lump is not even a whole number of
#: records.
_CLUSTER_GLADIATOR = struct.Struct("<3i")

Vec3 = Tuple[float, float, float]


class AasError(Exception):
    pass


class UnsupportedVersion(AasError):
    def __init__(self, version):
        super().__init__(f"unsupported AAS version {version}, "
                         f"expected one of {SUPPORTED_VERSIONS}")
        self.version = version


def cluster_struct(version):
    """The ``aas_cluster_t`` layout for a given AAS version."""
    return _CLUSTER if version > VERSION_GLADIATOR else _CLUSTER_GLADIATOR


def header_size(version):
    """Header length in bytes. Version 3 has no ``bspchecksum`` field."""
    return 4 + 4 + NUM_LUMPS * _LUMP.size + (4 if version > VERSION_GLADIATOR else 0)


def ddata(data):
    """``AAS_DData`` -- the involutive XOR the v4/v5 lump directory is stored
    under. Applying it a second time undoes it."""
    return bytes(b ^ ((i * DDATA_KEY) & 0xFF) for i, b in enumerate(data))


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


@dataclass
class BBox:
    presencetype: int
    flags: int
    mins: Vec3
    maxs: Vec3


@dataclass
class Plane:
    normal: Vec3
    dist: float
    type: int


@dataclass
class Face:
    planenum: int
    faceflags: int
    numedges: int
    firstedge: int
    frontarea: int
    backarea: int

    @property
    def flag_names(self):
        return _flag_names(FaceFlags, self.faceflags)


@dataclass
class Area:
    areanum: int
    numfaces: int
    firstface: int
    mins: Vec3
    maxs: Vec3
    center: Vec3


@dataclass
class AreaSettings:
    contents: int
    areaflags: int
    presencetype: int
    cluster: int
    clusterareanum: int
    numreachableareas: int
    firstreachablearea: int

    @property
    def grounded(self):
        return bool(self.areaflags & AreaFlags.GROUNDED)

    @property
    def ladder(self):
        return bool(self.areaflags & AreaFlags.LADDER)

    @property
    def liquid(self):
        return bool(self.areaflags & AreaFlags.LIQUID)

    @property
    def flag_names(self):
        return _flag_names(AreaFlags, self.areaflags)

    @property
    def mover(self):
        return bool(self.contents & AreaContents.MOVER)

    @property
    def mover_model(self):
        """Inline brush model number of the mover occupying this area, packed
        into the top byte of ``contents``. 0 when there is none."""
        return (self.contents >> AREACONTENTS_MODELNUMSHIFT) & AREACONTENTS_MAXMODELNUM

    @property
    def content_names(self):
        names = _flag_names(AreaContents, self.contents & ~(
            AREACONTENTS_MAXMODELNUM << AREACONTENTS_MODELNUMSHIFT))
        if self.mover_model:
            names.append(f"model{self.mover_model}")
        return names


@dataclass
class Reachability:
    """One possible move out of an area, into :attr:`areanum`.

    ``traveltype`` is kept raw -- masking is :attr:`travel_type`, the team bits
    are :attr:`not_team1` / :attr:`not_team2` -- because a file may carry flag
    combinations this module does not model.

    ``facenum`` and ``edgenum`` are likewise raw, because BSPC reuses them as
    a payload whose meaning depends on the travel type (``be_aas_reach.c``,
    and ``be_ai_move.c`` reading them back):

    ========================  ==============================  =====================
    travel type               ``facenum``                     ``edgenum``
    ========================  ==============================  =====================
    ``ELEVATOR``              mover model number              plat travel height
    ``FUNCBOB``               ``spawnflags<<16 | modelnum``   packed move endpoints
    ``JUMPPAD``               z velocity                      horizontal velocity
    ``LADDER``                *signed* face index             shared edge index
    ``TELEPORT``              0                               0
    everything else           face index towards the area     edge index
    ========================  ==============================  =====================
    """

    areanum: int
    facenum: int
    edgenum: int
    start: Vec3
    end: Vec3
    traveltype: int
    traveltime: int

    @property
    def travel_type(self):
        return self.traveltype & TRAVELTYPE_MASK

    @property
    def travel_name(self):
        return _name(TravelType, self.travel_type)

    @property
    def not_team1(self):
        return bool(self.traveltype & TRAVELFLAG_NOTTEAM1)

    @property
    def not_team2(self):
        return bool(self.traveltype & TRAVELFLAG_NOTTEAM2)

    @property
    def mover_model(self):
        """The mover's inline brush model number, for ``ELEVATOR`` and
        ``FUNCBOB``. Meaningless for any other travel type."""
        return self.facenum & MOVER_MODEL_MASK

    @property
    def funcbob_spawnflags(self):
        """``FUNCBOB`` only -- the mover's spawnflags, which encode its axis."""
        return self.facenum >> 16

    @property
    def elevator_height(self):
        """``ELEVATOR`` only -- how far the plat travels."""
        return self.edgenum

    @property
    def jumppad_velocity(self):
        """``JUMPPAD`` only -- (horizontal, vertical) launch speed."""
        return self.edgenum, self.facenum


@dataclass
class Node:
    """A BSP tree node over the areas. A negative child is ``-areanum``; a
    zero child is a solid leaf."""

    planenum: int
    children: Tuple[int, int]


@dataclass
class Portal:
    areanum: int
    frontcluster: int
    backcluster: int
    clusterareanum: Tuple[int, int]


@dataclass
class Cluster:
    numareas: int
    numportals: int
    firstportal: int
    #: Absent in version 3, which predates the field.
    numreachabilityareas: Optional[int] = None


@dataclass
class AasFile:
    version: int = VERSION
    bspchecksum: int = 0
    bboxes: List[BBox] = field(default_factory=list)
    vertexes: List[Vec3] = field(default_factory=list)
    planes: List[Plane] = field(default_factory=list)
    edges: List[Tuple[int, int]] = field(default_factory=list)
    edgeindex: List[int] = field(default_factory=list)
    faces: List[Face] = field(default_factory=list)
    faceindex: List[int] = field(default_factory=list)
    areas: List[Area] = field(default_factory=list)
    areasettings: List[AreaSettings] = field(default_factory=list)
    reachability: List[Reachability] = field(default_factory=list)
    nodes: List[Node] = field(default_factory=list)
    portals: List[Portal] = field(default_factory=list)
    portalindex: List[int] = field(default_factory=list)
    clusters: List[Cluster] = field(default_factory=list)

    # -- the two lumps the converter actually walks -------------------------

    def area_reachabilities(self, areanum):
        """The reachabilities leading *out of* area `areanum`, sliced out of
        the flat lump. Area 0 is a dummy and has none."""
        s = self.areasettings[areanum]
        return self.reachability[s.firstreachablearea:
                                 s.firstreachablearea + s.numreachableareas]

    def grounded_areas(self):
        """Area numbers a player can stand in. These become nav nodes."""
        return [i for i, s in enumerate(self.areasettings)
                if i and s.grounded]

    # -- geometry ----------------------------------------------------------

    def area_faces(self, areanum):
        """(face, front_facing) for each face bounding the area. A negative
        faceindex entry means the plane normal points *out* of the area."""
        area = self.areas[areanum]
        out = []
        for i in range(area.firstface, area.firstface + area.numfaces):
            fi = self.faceindex[i]
            out.append((self.faces[abs(fi)], fi > 0))
        return out

    def face_points(self, face):
        """The face's boundary vertexes, counter-clockwise."""
        pts = []
        for i in range(face.firstedge, face.firstedge + face.numedges):
            ei = self.edgeindex[i]
            edge = self.edges[abs(ei)]
            pts.append(self.vertexes[edge[0] if ei >= 0 else edge[1]])
        return pts

    def ground_face(self, areanum):
        """The area's ``FACE_GROUND`` face, or None. An area may not have a
        mixture of ground and gap faces, so there is at most one useful
        answer; the largest is returned if several are flagged."""
        best = None
        best_area = -1.0
        for face, _ in self.area_faces(areanum):
            if not face.faceflags & FaceFlags.GROUND:
                continue
            pts = self.face_points(face)
            a = _polygon_area_xy(pts)
            if a > best_area:
                best, best_area = face, a
        return best

    def point_area_num(self, point):
        """``AAS_PointAreaNum`` -- walk the tree to the area containing
        `point`, or 0 for solid or outside. Areas are convex and the tree is
        complete, so this is exact, not nearest-neighbour."""
        if not self.nodes:
            return 0
        nodenum = 1
        while nodenum > 0:
            node = self.nodes[nodenum]
            plane = self.planes[node.planenum]
            dist = (point[0] * plane.normal[0] + point[1] * plane.normal[1]
                    + point[2] * plane.normal[2] - plane.dist)
            nodenum = node.children[0 if dist > 0 else 1]
        return -nodenum

    def totals(self):
        """The counts ``bspc -aasinfo`` prints, for cross-checking a parse."""
        return {
            "numvertexes": len(self.vertexes),
            "numplanes": len(self.planes),
            "numedges": len(self.edges),
            "edgeindexsize": len(self.edgeindex),
            "numfaces": len(self.faces),
            "faceindexsize": len(self.faceindex),
            "numareas": len(self.areas),
            "numareasettings": len(self.areasettings),
            "reachabilitysize": len(self.reachability),
            "numnodes": len(self.nodes),
            "numportals": len(self.portals),
            "portalindexsize": len(self.portalindex),
            "numclusters": len(self.clusters),
        }


def _polygon_area_xy(points):
    """Twice the shoelace area of a polygon projected onto the xy plane. Only
    used to compare candidate faces, so the factor of two does not matter."""
    total = 0.0
    for i, p in enumerate(points):
        q = points[(i + 1) % len(points)]
        total += p[0] * q[1] - q[0] * p[1]
    return abs(total)


def _records(data, lump, rec, build):
    ofs, length = lump
    if length % rec.size:
        raise AasError(f"lump at {ofs} is {length} bytes, "
                       f"not a multiple of the {rec.size}-byte record")
    if ofs + length > len(data):
        raise AasError(f"lump at {ofs} runs {ofs + length - len(data)} bytes "
                       f"past the end of a {len(data)}-byte file")
    return [build(rec.unpack_from(data, ofs + i * rec.size))
            for i in range(length // rec.size)]


def loads(data):
    """Parse an AAS file."""
    if len(data) < header_size(VERSION_GLADIATOR):
        raise AasError(f"too short to hold a header ({len(data)} bytes)")
    ident, = _IDENT.unpack_from(data, 0)
    if ident != IDENT:
        raise AasError(f"bad ident {ident!r}, expected {IDENT!r}")
    version, = struct.unpack_from("<i", data, 4)
    if version not in SUPPORTED_VERSIONS:
        raise UnsupportedVersion(version)

    size = header_size(version)
    if len(data) < size:
        raise AasError(f"version {version} needs a {size}-byte header, "
                       f"file has {len(data)} bytes")
    head = data[DDATA_OFFSET:size]
    if version > VERSION_GLADIATOR:
        head = ddata(head)

    aas = AasFile(version=version)
    if version > VERSION_GLADIATOR:
        aas.bspchecksum, = struct.unpack_from("<i", head, 0)
        head = head[4:]
    lumps = [_LUMP.unpack_from(head, i * _LUMP.size) for i in range(NUM_LUMPS)]

    for i, (ofs, length) in enumerate(lumps):
        if length < 0 or ofs < 0:
            raise AasError(f"lump {LUMP_NAMES[i]} has offset {ofs} length {length} "
                           f"-- a v{version} header read the wrong way looks like this")

    aas.bboxes = _records(data, lumps[LUMP_BBOXES], _BBOX,
                          lambda v: BBox(v[0], v[1], v[2:5], v[5:8]))
    aas.vertexes = _records(data, lumps[LUMP_VERTEXES], _VERTEX, lambda v: v)
    aas.planes = _records(data, lumps[LUMP_PLANES], _PLANE,
                          lambda v: Plane(v[0:3], v[3], v[4]))
    aas.edges = _records(data, lumps[LUMP_EDGES], _EDGE, lambda v: v)
    aas.edgeindex = _records(data, lumps[LUMP_EDGEINDEX], _INDEX, lambda v: v[0])
    aas.faces = _records(data, lumps[LUMP_FACES], _FACE, lambda v: Face(*v))
    aas.faceindex = _records(data, lumps[LUMP_FACEINDEX], _INDEX, lambda v: v[0])
    aas.areas = _records(data, lumps[LUMP_AREAS], _AREA,
                         lambda v: Area(v[0], v[1], v[2], v[3:6], v[6:9], v[9:12]))
    aas.areasettings = _records(data, lumps[LUMP_AREASETTINGS], _AREASETTINGS,
                                lambda v: AreaSettings(*v))
    aas.reachability = _records(data, lumps[LUMP_REACHABILITY], _REACHABILITY,
                                lambda v: Reachability(v[0], v[1], v[2], v[3:6],
                                                       v[6:9], v[9], v[10]))
    aas.nodes = _records(data, lumps[LUMP_NODES], _NODE,
                         lambda v: Node(v[0], (v[1], v[2])))
    aas.portals = _records(data, lumps[LUMP_PORTALS], _PORTAL,
                           lambda v: Portal(v[0], v[1], v[2], (v[3], v[4])))
    aas.portalindex = _records(data, lumps[LUMP_PORTALINDEX], _INDEX, lambda v: v[0])
    aas.clusters = _records(
        data, lumps[LUMP_CLUSTERS], cluster_struct(version),
        (lambda v: Cluster(v[0], v[2], v[3], v[1])) if version > VERSION_GLADIATOR
        else (lambda v: Cluster(*v)))

    if len(aas.areasettings) != len(aas.areas):
        raise AasError(f"{len(aas.areas)} areas but {len(aas.areasettings)} "
                       f"area settings")
    return aas


def load(path):
    with open(path, "rb") as fp:
        return loads(fp.read())

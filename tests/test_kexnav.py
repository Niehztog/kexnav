#!/usr/bin/env python3
"""Unit tests for the parts the corpus gates cannot reach.

The real proof of this project is elsewhere: ``tests/roundtrip.py`` on 174
shipped nav files, ``tests/aascheck.py`` against BSPC's own reading of an
AAS, and ``kexnav.py check`` against Nightdive's hand-authored output. Those need the retail pak
and a BSPC binary, so they are not tests -- they are gates, and they are the
ones to believe.

What is left for a unit test is the handful of pure functions where a corpus
gate would pass either way: the involutive header XOR, the BSP repacker, the
funnel geometry, the CSR emission, and the error paths a well-formed file never
exercises. Everything here runs on synthesised data and needs nothing on disk.

    python3 -m unittest discover -s tests -v
"""

import math
import os
import shutil
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kexnav import aas, bsp, convert, env, nav3, validate
from kexnav.cli import check, generate


def a_nav(version=nav3.VERSION):
    """A small but complete nav file: three nodes, a walk and a jump."""
    nav = nav3.NavFile(version=version, heuristic=0.8)
    nav.nodes = [
        nav3.Node(0, 1, 0, 32, (0.0, 0.0, 0.53)),
        nav3.Node(0, 1, 1, 32, (128.0, 0.0, 0.53)),
        nav3.Node(int(nav3.NodeFlags.LADDER), 0, 2, 32, (256.0, 0.0, 64.53)),
    ]
    nav.links = [
        nav3.Link(1, int(nav3.LinkType.WALK), int(nav3.ALL_TEAMS), nav3.NO_TRAVERSAL),
        nav3.Link(2, int(nav3.LinkType.LONG_JUMP), int(nav3.ALL_TEAMS), 0),
    ]
    nav.traversals = [
        nav3.Traversal(funnel=(96.0, 0.0, 0.53), start=(128.0, 0.0, 0.53),
                       end=(256.0, 0.0, 64.53), ladder_plane=(0.0, 0.0, 0.0)),
    ]
    return nav


class TestNav3RoundTrip(unittest.TestCase):
    def test_round_trip_is_byte_exact(self):
        for version in nav3.SUPPORTED_VERSIONS:
            with self.subTest(version=version):
                nav = a_nav(version)
                data = nav3.dumps(nav)
                self.assertEqual(nav3.dumps(nav3.loads(data)), data)

    def test_byte_size_predicts_the_encoding(self):
        for version in nav3.SUPPORTED_VERSIONS:
            with self.subTest(version=version):
                nav = a_nav(version)
                self.assertEqual(nav.byte_size(), len(nav3.dumps(nav)))

    def test_pre_v4_traversal_drops_the_ladder_plane(self):
        # the only layout difference across versions 2..6
        self.assertEqual(len(nav3.dumps(a_nav(4))) - len(nav3.dumps(a_nav(3))), 12)

    def test_edicts_survive(self):
        nav = a_nav()
        nav.edicts = [nav3.NavEdict(link=0, model=3, mins=(-2.0, -2.0, -2.0),
                                    maxs=(66.0, 66.0, 66.0))]
        again = nav3.loads(nav3.dumps(nav))
        self.assertEqual(again.edicts, nav.edicts)


class TestNav3Errors(unittest.TestCase):
    """A silent partial parse would defeat the point of the round trip, so
    every one of these has to raise."""

    def test_bad_magic(self):
        data = bytearray(nav3.dumps(a_nav()))
        data[0:4] = b"NAV2"
        with self.assertRaisesRegex(nav3.NavError, "bad magic"):
            nav3.loads(bytes(data))

    def test_unsupported_version(self):
        data = bytearray(nav3.dumps(a_nav()))
        struct.pack_into("<I", data, 4, 7)
        with self.assertRaises(nav3.UnsupportedVersion):
            nav3.loads(bytes(data))

    def test_truncated(self):
        data = nav3.dumps(a_nav())
        with self.assertRaisesRegex(nav3.NavError, "truncated|too short"):
            nav3.loads(data[:len(data) - 8])

    def test_trailing_bytes(self):
        with self.assertRaisesRegex(nav3.NavError, "trailing"):
            nav3.loads(nav3.dumps(a_nav()) + b"\0\0\0\0")

    def test_header_alone_is_too_short(self):
        with self.assertRaisesRegex(nav3.NavError, "too short"):
            nav3.loads(b"NAV3")


class TestFunnelSentinels(unittest.TestCase):
    """Two spellings of unset, both seen in the corpus."""

    def test_1e30_is_unset(self):
        t = nav3.Traversal((nav3.UNSET_COORD,) * 3, (0, 0, 0), (1, 1, 1))
        self.assertFalse(t.has_funnel)

    def test_all_zero_is_unset(self):
        t = nav3.Traversal((0.0, 0.0, 0.0), (0, 0, 0), (1, 1, 1))
        self.assertFalse(t.has_funnel)

    def test_a_real_point_is_set(self):
        t = nav3.Traversal((0.0, 0.0, 0.53), (0, 0, 0), (1, 1, 1))
        self.assertTrue(t.has_funnel)


class TestValidate(unittest.TestCase):
    def test_a_good_file_is_clean(self):
        self.assertEqual(validate.check(a_nav()), [])

    def test_broken_csr_is_caught(self):
        nav = a_nav()
        nav.nodes[1].first_link = 0
        self.assertTrue(any("first_link" in p for p in validate.check(nav)))

    def test_out_of_range_target_is_caught(self):
        nav = a_nav()
        nav.links[0].target = 99
        self.assertTrue(any("target 99" in p for p in validate.check(nav)))

    def test_shared_traversal_is_caught(self):
        nav = a_nav()
        nav.links[0].traversal = 0        # both links now point at traversal 0
        problems = validate.check(nav)
        self.assertTrue(any("more than one link" in p for p in problems))

    def test_orphan_traversal_is_caught(self):
        nav = a_nav()
        nav.links[1].traversal = nav3.NO_TRAVERSAL
        self.assertTrue(any("no link" in p for p in validate.check(nav)))

    def test_ladder_plane_must_match_the_link_type(self):
        nav = a_nav()
        nav.traversals[0].ladder_plane = (1.0, 0.0, 0.0)   # but it is a jump
        self.assertTrue(any("ladder plane" in p for p in validate.check(nav)))

    def test_ladder_plane_must_be_a_unit_vector(self):
        nav = a_nav()
        nav.links[1].type = int(nav3.LinkType.LADDER)
        nav.traversals[0].ladder_plane = (3.0, 0.0, 0.0)
        self.assertTrue(any("unit vector" in p for p in validate.check(nav)))

    def test_walk_may_not_carry_a_traversal(self):
        nav = a_nav()
        nav.links[0].type = int(nav3.LinkType.WALK)
        nav.links[0].traversal = 0
        nav.links[1].traversal = nav3.NO_TRAVERSAL
        self.assertTrue(any("WALK carries a traversal" in p
                            for p in validate.check(nav)))

    def test_u16_limits_are_enforced(self):
        nav = a_nav()
        nav.nodes = nav.nodes * 30000          # 90000 nodes
        self.assertTrue(any("u16" in p for p in
                            validate.check_format_limits(nav)))


class TestAasHeader(unittest.TestCase):
    def test_ddata_is_involutive(self):
        blob = bytes(range(256)) * 2
        self.assertEqual(aas.ddata(aas.ddata(blob)), blob)

    def test_ddata_matches_the_c_loop(self):
        # aas_file.c:236 -- data[i] ^= (unsigned char) i * 119
        blob = bytes(64)
        self.assertEqual(aas.ddata(blob),
                         bytes((i * 119) & 0xFF for i in range(64)))

    def test_header_size_by_version(self):
        # v3 has no bspchecksum
        self.assertEqual(aas.header_size(3), 120)
        self.assertEqual(aas.header_size(5), 124)

    def test_cluster_record_grew_in_v4(self):
        self.assertEqual(aas.cluster_struct(3).size, 12)
        self.assertEqual(aas.cluster_struct(5).size, 16)

    def test_record_sizes(self):
        # every one of these is load-bearing: get one wrong and the lump
        # divides unevenly, which is how the v3 cluster size was found
        for record, size in ((aas._BBOX, 32), (aas._VERTEX, 12), (aas._PLANE, 20),
                             (aas._EDGE, 8), (aas._FACE, 24), (aas._AREA, 48),
                             (aas._AREASETTINGS, 28), (aas._REACHABILITY, 44),
                             (aas._NODE, 12), (aas._PORTAL, 20), (aas._CLUSTER, 16)):
            self.assertEqual(record.size, size)

    def test_bad_ident_raises(self):
        with self.assertRaisesRegex(aas.AasError, "bad ident"):
            aas.loads(b"XAAS" + struct.pack("<i", 5) + bytes(200))

    def test_unsupported_version_raises(self):
        with self.assertRaises(aas.UnsupportedVersion):
            aas.loads(aas.IDENT + struct.pack("<i", 9) + bytes(200))


class TestAasPayloadFields(unittest.TestCase):
    """``facenum``/``edgenum`` are a per-travel-type payload, not indices."""

    def test_mover_model_is_masked_out_of_facenum(self):
        r = aas.Reachability(1, (7 << 16) | 42, 0, (0, 0, 0), (0, 0, 0),
                             aas.TravelType.FUNCBOB, 0)
        self.assertEqual(r.mover_model, 42)
        self.assertEqual(r.funcbob_spawnflags, 7)

    def test_team_flags_come_off_the_travel_type(self):
        r = aas.Reachability(1, 0, 0, (0, 0, 0), (0, 0, 0),
                             aas.TravelType.WALK | aas.TRAVELFLAG_NOTTEAM1, 0)
        self.assertEqual(r.travel_type, aas.TravelType.WALK)
        self.assertTrue(r.not_team1)
        self.assertFalse(r.not_team2)

    def test_mover_model_comes_out_of_area_contents(self):
        s = aas.AreaSettings(int(aas.AreaContents.MOVER) | (5 << 24),
                             int(aas.AreaFlags.GROUNDED), 6, 1, 0, 0, 0)
        self.assertTrue(s.mover)
        self.assertEqual(s.mover_model, 5)
        # the packed model number must not leak into the flag names
        self.assertIn("MOVER", s.content_names)
        self.assertIn("model5", s.content_names)


class TestBspStripping(unittest.TestCase):
    def _bsp(self, lighting=8192, vis=4096):
        lumps = [(0, 0)] * bsp.NUM_LUMPS
        body = bytearray()
        def add(index, payload):
            lumps[index] = (0, len(payload))
            return payload
        ents = b'{ "classname" "worldspawn" }\n'
        models = struct.pack("<9f3i", -1, -1, -1, 1, 1, 1, 0, 0, 0, 0, 0, 1)
        pieces = {bsp.LUMP_ENTITIES: ents, bsp.LUMP_MODELS: models,
                  7: b"L" * lighting, 3: b"V" * vis}
        head = bytearray(struct.calcsize("<4si") + bsp.NUM_LUMPS * 8)
        struct.pack_into("<4si", head, 0, bsp.IDENT, bsp.VERSION)
        base = len(head)
        for index, payload in pieces.items():
            struct.pack_into("<ii", head, 8 + index * 8, base + len(body), len(payload))
            body += payload
        return bytes(head + body)

    def test_stripping_removes_the_big_unused_lumps(self):
        data = self._bsp()
        slim = bsp.strip_for_bspc(data)
        self.assertLess(len(slim), len(data))
        for lump in bsp.BSPC_UNUSED_LUMPS:
            _, length = struct.unpack_from("<ii", slim, 8 + lump * 8)
            self.assertEqual(length, 0, f"lump {lump} survived")

    def test_stripping_keeps_what_bspc_needs(self):
        slim = bsp.loads(bsp.strip_for_bspc(self._bsp()))
        self.assertEqual(len(slim.models), 1)
        self.assertEqual(slim.entities[0]["classname"], "worldspawn")

    def test_a_map_with_nothing_to_drop_is_returned_unchanged(self):
        data = self._bsp(lighting=0, vis=0)
        self.assertIs(bsp.strip_for_bspc(data), data)

    def test_every_kept_lump_stays_four_byte_aligned(self):
        slim = bsp.strip_for_bspc(self._bsp(lighting=8191, vis=4095))
        for i in range(bsp.NUM_LUMPS):
            ofs, length = struct.unpack_from("<ii", slim, 8 + i * 8)
            self.assertEqual(ofs % 4, 0, f"lump {i} at {ofs}")
            self.assertLessEqual(ofs + length, len(slim))


class TestFloodSeeds(unittest.TestCase):
    """Seeds are what let BSPC keep a disjoint volume it would otherwise fill
    as outside. Getting the leaf record wrong would silently produce none."""

    def _bsp_with_leaves(self, leaves):
        """leaves: (contents, mins, maxs) triples."""
        lumps = {}
        ents = b'{ "classname" "worldspawn" }\n\0'
        models = struct.pack("<9f3i", -1, -1, -1, 1, 1, 1, 0, 0, 0, 0, 0, 1)
        leaf_blob = b"".join(
            struct.pack("<ihh6h4H", c, 0, 0, *mins, *maxs, 0, 0, 0, 0)
            for c, mins, maxs in leaves)
        lumps[bsp.LUMP_ENTITIES] = ents
        lumps[bsp.LUMP_MODELS] = models
        lumps[bsp.LUMP_LEAFS] = leaf_blob
        head = bytearray(struct.calcsize("<4si") + bsp.NUM_LUMPS * 8)
        struct.pack_into("<4si", head, 0, bsp.IDENT, bsp.VERSION)
        body = bytearray()
        base = len(head)
        for index, payload in lumps.items():
            struct.pack_into("<ii", head, 8 + index * 8,
                             base + len(body), len(payload))
            body += payload
            while len(body) % 4:
                body += b"\0"
        return bytes(head + body)

    def test_only_empty_leaves_become_seeds(self):
        data = self._bsp_with_leaves([
            (0, (0, 0, 0), (64, 64, 64)),          # empty -> a seed
            (1, (100, 0, 0), (164, 64, 64)),       # solid -> not
        ])
        centres = bsp.empty_leaf_centres(data)
        self.assertEqual(centres, [(32.0, 32.0, 32.0)])

    def test_thin_leaves_are_skipped(self):
        data = self._bsp_with_leaves([(0, (0, 0, 0), (64, 64, 1))])
        self.assertEqual(bsp.empty_leaf_centres(data), [])

    def test_a_centre_at_the_world_origin_is_skipped(self):
        # FloodEntities ignores an entity whose origin is exactly (0,0,0)
        data = self._bsp_with_leaves([(0, (-32, -32, -32), (32, 32, 32))])
        self.assertEqual(bsp.empty_leaf_centres(data), [])

    def test_seeds_are_appended_as_entities_and_lumps_survive(self):
        data = self._bsp_with_leaves([(0, (0, 0, 0), (64, 64, 64)),
                                      (0, (128, 0, 0), (192, 64, 64))])
        seeded, count = bsp.add_flood_seeds(data)
        self.assertEqual(count, 2)
        parsed = bsp.loads(seeded)
        self.assertEqual(len(parsed.entities), 3)   # worldspawn plus two seeds
        self.assertEqual(parsed.entities[0]["classname"], "worldspawn")
        # a classname no reachability pass matches on
        self.assertEqual({e["classname"] for e in parsed.entities[1:]}, {"light"})
        self.assertEqual(bsp.parse_vec3(parsed.entities[1]["origin"]),
                         (32.0, 32.0, 32.0))
        self.assertEqual(len(parsed.models), 1)

    def test_seeding_also_strips_the_unused_lumps(self):
        data = self._bsp_with_leaves([(0, (0, 0, 0), (64, 64, 64))])
        seeded, _ = bsp.add_flood_seeds(data)
        for lump in bsp.BSPC_UNUSED_LUMPS:
            _, length = struct.unpack_from("<ii", seeded, 8 + lump * 8)
            self.assertEqual(length, 0)

    def test_seeds_are_capped_against_the_maps_own_entities(self):
        # exceeding BSPC's MAX_MAP_ENTITIES is fatal, so a map already at the
        # limit must come back with no seeds at all
        leaves = [(0, (i * 128, 0, 0), (i * 128 + 64, 64, 64))
                  for i in range(1, 40)]
        data = self._bsp_with_leaves(leaves)
        # rebuild with a crowded entity lump
        crowded = bytearray(data)
        ents = b"".join(b'{ "classname" "light" "origin" "%d 0 0" }\n' % i
                        for i in range(1, bsp.BSPC_MAX_ENTITIES)) + b"\0"
        head = bytearray(struct.calcsize("<4si") + bsp.NUM_LUMPS * 8)
        struct.pack_into("<4si", head, 0, bsp.IDENT, bsp.VERSION)
        body = bytearray()
        base = len(head)
        for index, payload in ((bsp.LUMP_ENTITIES, bytes(ents)),
                               (bsp.LUMP_LEAFS, b"".join(
                                   struct.pack("<ihh6h4H", c, 0, 0, *mn, *mx,
                                               0, 0, 0, 0)
                                   for c, mn, mx in leaves))):
            struct.pack_into("<ii", head, 8 + index * 8,
                             base + len(body), len(payload))
            body += payload
            while len(body) % 4:
                body += b"\0"
        _, count = bsp.add_flood_seeds(bytes(head + body))
        self.assertEqual(count, 0)

    def test_seed_count_is_capped(self):
        leaves = [(0, (i * 128, 0, 0), (i * 128 + 64, 64, 64))
                  for i in range(1, 60)]
        self.assertEqual(len(bsp.empty_leaf_centres(self._bsp_with_leaves(leaves),
                                                    limit=10)), 10)


class TestBspEntities(unittest.TestCase):
    def test_repeated_keys_are_joined_not_dropped(self):
        # Quake II entities legally carry several `target` keys
        ents = bsp.parse_entities(
            b'{ "classname" "trigger_multiple" "target" "a" "target" "b" }')
        self.assertEqual(ents[0]["target"], "a b")

    def test_nested_and_blank_blocks(self):
        self.assertEqual(bsp.parse_entities(b'{}\n{ "a" "b" }\n{}'),
                         [{"a": "b"}])

    def test_unbalanced_braces_raise(self):
        with self.assertRaises(bsp.BspError):
            bsp.parse_entities(b'{ "a" "b" }}')
        with self.assertRaises(bsp.BspError):
            bsp.parse_entities(b'{ "a" "b"')

    def test_model_index(self):
        b = bsp.BspFile(models=[], entities=[])
        self.assertEqual(b.model_index({"model": "*7"}), 7)
        self.assertIsNone(b.model_index({"model": "maps/x.bsp"}))
        self.assertIsNone(b.model_index({}))

    def test_parse_vec3(self):
        self.assertEqual(bsp.parse_vec3("0 128 -64"), (0.0, 128.0, -64.0))
        self.assertEqual(bsp.parse_vec3("nonsense"), (0.0, 0.0, 0.0))
        self.assertEqual(bsp.parse_vec3(""), (0.0, 0.0, 0.0))


def a_bsp(lumps):
    """A BSP carrying just the named lumps: {index: bytes}."""
    head = bytearray(struct.calcsize("<4si") + bsp.NUM_LUMPS * 8)
    struct.pack_into("<4si", head, 0, bsp.IDENT, bsp.VERSION)
    body = bytearray()
    base = len(head)
    for index, payload in lumps.items():
        struct.pack_into("<ii", head, 8 + index * 8, base + len(body), len(payload))
        body += payload
        while len(body) % 4:
            body += b"\0"
    return bytes(head + body)


def a_model(mins, maxs, headnode=1):
    return struct.pack("<9f3i", *mins, *maxs, 0, 0, 0, headnode, 0, 1)


def a_faced_bsp(mins, maxs, face_heights, flipped=()):
    """A BSP whose model 1 has an upward face at each of `face_heights`.

    `flipped` names heights that face *downward*. Quake II stores that as a
    plane pointing up with the face marked ``side``, which negates it -- the
    compiler keeps one plane per orientation and flips per face.
    """
    planes = b""
    faces = b""
    for i, h in enumerate(face_heights):
        planes += struct.pack("<4fi", 0.0, 0.0, 1.0, h, 0)
        faces += struct.pack("<Hhihh4Bi", i, 1 if h in flipped else 0,
                             0, 0, 0, 0, 0, 0, 0, 0)
    models = (a_model((-1, -1, -1), (1, 1, 1), headnode=0)
              + struct.pack("<9f3i", *mins, *maxs, 0, 0, 0, 1, 0,
                            len(face_heights)))
    return a_bsp({bsp.LUMP_ENTITIES: b'{ "classname" "worldspawn" }\n\0',
                  bsp.LUMP_PLANES: planes,
                  bsp.LUMP_FACES: faces,
                  bsp.LUMP_MODELS: models})


class TestRideSurface(unittest.TestCase):
    """A Quake II train is often a hollow car, so the surface a rider stands on
    is a face inside the model rather than its bounding box top. Getting this
    wrong put generated TRAIN links 150 units above Nightdive's."""

    def test_a_slab_rides_on_its_top(self):
        data = a_faced_bsp((0, 0, 0), (64, 64, 16), [16.0])
        self.assertAlmostEqual(bsp.model_ride_surface(data, 1), 16.0)

    def test_a_car_rides_on_its_interior_floor(self):
        # roof at 176, interior floor at 20 -- q2dm2's train in miniature
        data = a_faced_bsp((0, 0, 0), (64, 64, 176), [176.0, 20.0, 60.0])
        self.assertAlmostEqual(bsp.model_ride_surface(data, 1), 20.0)

    def test_a_downward_face_is_ignored(self):
        # the same plane heights, but the low one faces down: a ceiling
        data = a_faced_bsp((0, 0, 0), (64, 64, 176), [176.0, 20.0],
                           flipped=(20.0,))
        self.assertAlmostEqual(bsp.model_ride_surface(data, 1), 176.0)

    def test_a_face_outside_the_box_is_ignored(self):
        data = a_faced_bsp((0, 0, 0), (64, 64, 16), [16.0, -500.0])
        self.assertAlmostEqual(bsp.model_ride_surface(data, 1), 16.0)

    def test_no_upward_face_falls_back_to_the_box_top(self):
        # every face points down, so there is nothing to stand on and the
        # bounding box top is the only answer left
        data = a_faced_bsp((0, 0, 0), (64, 64, 96), [20.0, 60.0],
                           flipped=(20.0, 60.0))
        self.assertAlmostEqual(bsp.model_ride_surface(data, 1), 96.0)


def a_train_bsp(corners, spawnflags=0, mins=(0, 0, 0), maxs=(64, 64, 16)):
    """A BSP with one func_train on a path through `corners`."""
    ents = b'{ "classname" "worldspawn" }\n'
    ents += (b'{ "classname" "func_train" "model" "*1" "target" "c0" '
             b'"spawnflags" "%d" }\n' % spawnflags)
    for i, c in enumerate(corners):
        nxt = "c%d" % ((i + 1) % len(corners))
        ents += (b'{ "classname" "path_corner" "targetname" "c%d" '
                 b'"origin" "%g %g %g" "target" "%s" }\n'
                 % (i, c[0], c[1], c[2], nxt.encode()))
    models = a_model((-1, -1, -1), (1, 1, 1), headnode=0) + a_model(mins, maxs, headnode=1)
    return a_bsp({bsp.LUMP_ENTITIES: ents + b"\0", bsp.LUMP_MODELS: models})


class TestTrainTranslation(unittest.TestCase):
    """A func_train becomes a func_bobbing so BSPC can compute the ride. The
    arithmetic has to be exact -- BSPC oscillates the *centre* of the model's
    bounds by `height` along the spawnflag's axis, so a mistake here invents
    movement that is not in the map."""

    def test_a_two_corner_train_gives_one_segment(self):
        # mins aligns to the corner, so the box travels from y=0 to y=128
        b = bsp.loads(a_train_bsp([(0, 0, 0), (0, 128, 0)]))
        segs = bsp.train_travel_segments(b)
        self.assertEqual(len(segs), 1)
        model, offset, height, axis = segs[0]
        self.assertEqual(model, 1)
        self.assertEqual(axis, 1)                  # y
        self.assertAlmostEqual(height, 64.0)       # half the travel
        # the box centre is (32,32,8); the travel midpoint puts it at y=96
        self.assertAlmostEqual(offset[1], 64.0)

    def test_spawnflag_axis_mapping(self):
        for corners, axis, flag in ((((0,0,0),(128,0,0)), 0, 1),
                                    (((0,0,0),(0,128,0)), 1, 2),
                                    (((0,0,0),(0,0,128)), 2, 0)):
            with self.subTest(axis=axis):
                b = bsp.loads(a_train_bsp(list(corners)))
                segs = bsp.train_travel_segments(b)
                self.assertEqual(segs[0][3], axis)
                self.assertEqual(bsp._BOB_AXIS_SPAWNFLAG[segs[0][3]], flag)

    def test_diagonal_segments_are_dropped(self):
        b = bsp.loads(a_train_bsp([(0, 0, 0), (128, 128, 0)]))
        self.assertEqual(bsp.train_travel_segments(b), [])

    def test_a_zero_length_segment_is_dropped(self):
        # BSPC substitutes height 32 for a zero one, which would invent travel
        b = bsp.loads(a_train_bsp([(0, 0, 0), (0, 0, 0)]))
        self.assertEqual(bsp.train_travel_segments(b), [])

    def test_a_loop_closes(self):
        # three corners in a square-ish loop: two axis-aligned legs plus the
        # closing one
        b = bsp.loads(a_train_bsp([(0, 0, 0), (0, 128, 0), (0, 256, 0)]))
        segs = bsp.train_travel_segments(b)
        # 0->128, 128->256 and the closing 256->0, all along y
        self.assertEqual(len(segs), 3)
        self.assertEqual({round(s[2]) for s in segs}, {64, 128})

    def test_use_origin_puts_the_entity_origin_at_the_corner(self):
        plain = bsp.train_travel_segments(bsp.loads(a_train_bsp([(0,0,0),(0,128,0)])))
        used = bsp.train_travel_segments(
            bsp.loads(a_train_bsp([(0,0,0),(0,128,0)], spawnflags=bsp.TRAIN_USE_ORIGIN)))
        # same travel either way, but offset by where mins sits
        self.assertAlmostEqual(plain[0][2], used[0][2])
        self.assertAlmostEqual(plain[0][1][1] - used[0][1][1], 0.0)

    def test_fix_offset_shifts_the_midpoint_by_one(self):
        plain = bsp.train_travel_segments(bsp.loads(a_train_bsp([(0,0,0),(0,128,0)])))
        fixed = bsp.train_travel_segments(
            bsp.loads(a_train_bsp([(0,0,0),(0,128,0)], spawnflags=bsp.TRAIN_FIX_OFFSET)))
        self.assertAlmostEqual(plain[0][2], fixed[0][2])          # height unchanged
        self.assertAlmostEqual(plain[0][1][1] - fixed[0][1][1], 1.0)

    def test_bobbing_entities_carry_no_origin_key(self):
        # an origin key would seed FloodEntities from a geometric offset and
        # make BSPC leak -- the whole reason for the duplicate models
        data, count = bsp.add_train_bobbing(a_train_bsp([(0,0,0),(0,128,0)]))
        self.assertEqual(count, 1)
        bobs = bsp.loads(data).by_classname("func_bobbing")
        self.assertEqual(len(bobs), 1)
        self.assertNotIn("origin", bobs[0])
        self.assertEqual(bobs[0]["height"], "64.00")
        self.assertEqual(bobs[0]["spawnflags"], "2")

    def test_the_duplicate_model_carries_the_shifted_bounds(self):
        raw = a_train_bsp([(0,0,0),(0,128,0)])
        data, _ = bsp.add_train_bobbing(raw)
        before, after = bsp.loads(raw), bsp.loads(data)
        self.assertEqual(len(after.models), len(before.models) + 1)
        new = after.models[-1]
        real = before.models[1]
        # same geometry, bounds moved to the travel midpoint
        self.assertEqual(new.headnode, real.headnode)
        self.assertEqual(new.numfaces, real.numfaces)
        self.assertAlmostEqual(new.mins[1] - real.mins[1], 64.0)
        self.assertAlmostEqual(new.maxs[1] - real.maxs[1], 64.0)
        # maxs.z is the ride surface, which BSPC puts the rider 24 above
        self.assertAlmostEqual(new.maxs[2],
                               bsp.model_ride_surface(raw, 1))
        # and the bobbing entity references it
        self.assertEqual(after.by_classname("func_bobbing")[0]["model"],
                         "*%d" % (len(after.models) - 1))

    def test_a_map_with_no_translatable_train_is_just_stripped(self):
        raw = a_train_bsp([(0, 0, 0), (128, 128, 0)])       # diagonal only
        data, count = bsp.add_train_bobbing(raw)
        self.assertEqual(count, 0)
        self.assertEqual(bsp.loads(data).by_classname("func_bobbing"), [])

    def test_canonical_model_maps_a_duplicate_back(self):
        data, _ = bsp.add_train_bobbing(a_train_bsp([(0,0,0),(0,128,0)]))
        b = bsp.loads(data)
        self.assertEqual(bsp.canonical_model(b, len(b.models) - 1), 1)
        self.assertEqual(bsp.canonical_model(b, 1), 1)

    def test_canonical_model_tolerates_a_bad_index(self):
        b = bsp.loads(a_train_bsp([(0,0,0),(0,128,0)]))
        self.assertEqual(bsp.canonical_model(b, 999), 999)


def a_mover_bsp(entities, mins=(0, 0, 0), maxs=(64, 64, 128),
                face_heights=None):
    """A BSP with one brush entity on model 1. `entities` is a list of key
    dicts; ``model`` is filled in."""
    text = b'{ "classname" "worldspawn" }\n'
    for ent in entities:
        pairs = "".join('"%s" "%s" ' % (k, v) for k, v in ent.items())
        text += b'{ "model" "*1" ' + pairs.encode() + b'}\n'
    heights = [maxs[2]] if face_heights is None else face_heights
    planes = b"".join(struct.pack("<4fi", 0.0, 0.0, 1.0, h, 0) for h in heights)
    faces = b"".join(struct.pack("<Hhihh4Bi", i, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                     for i in range(len(heights)))
    models = (a_model((-1, -1, -1), (1, 1, 1), headnode=0)
              + struct.pack("<9f3i", *mins, *maxs, 0, 0, 0, 1, 0, len(heights)))
    return a_bsp({bsp.LUMP_ENTITIES: text + b"\0",
                  bsp.LUMP_PLANES: planes, bsp.LUMP_FACES: faces,
                  bsp.LUMP_MODELS: models})


class TestPlatLifts(unittest.TestCase):
    """BSPC's elevator pass matches the exact classname "func_plat", so a
    func_plat2 or a vertical func_door -- which are the same mover -- gets no
    ride reachability. These are described to it as func_plat instead."""

    def lifts(self, entities, **kw):
        return bsp.plat_lift_entities(bsp.loads(a_mover_bsp(entities, **kw)))

    def test_plat2_lip_defaults_to_zero_not_eight(self):
        # SP_func_plat2 has the lip default commented out
        lifts = self.lifts([{"classname": "func_plat2"}])
        self.assertEqual(len(lifts), 1)
        self.assertAlmostEqual(lifts[0][2], 128.0)      # the full box height

    def test_plat2_subtracts_lip_from_an_explicit_height(self):
        # func_plat does not; func_plat2 does
        lifts = self.lifts([{"classname": "func_plat2", "height": "100",
                             "lip": "8"}])
        self.assertAlmostEqual(lifts[0][2], 92.0)

    def test_plat2_brush_position_is_already_the_top(self):
        lifts = self.lifts([{"classname": "func_plat2"}])
        self.assertAlmostEqual(lifts[0][1], 0.0)        # no rise

    def test_a_door_opening_up_has_its_top_position_above_the_brush(self):
        lifts = self.lifts([{"classname": "func_door", "angle": "-1"}])
        self.assertAlmostEqual(lifts[0][2], 120.0)      # 128 less the lip of 8
        self.assertAlmostEqual(lifts[0][1], 120.0)      # and rises by that much

    def test_a_door_opening_down_is_already_at_its_top(self):
        lifts = self.lifts([{"classname": "func_door", "angle": "-2"}])
        self.assertAlmostEqual(lifts[0][2], 120.0)
        self.assertAlmostEqual(lifts[0][1], 0.0)

    def test_a_horizontal_door_is_not_a_lift(self):
        for angle in ("0", "90", "180", "270", None):
            with self.subTest(angle=angle):
                ent = {"classname": "func_door"}
                if angle is not None:
                    ent["angle"] = angle
                self.assertEqual(self.lifts([ent]), [])

    def test_a_door_lip_wider_than_the_box_is_dropped(self):
        # travel would be zero or negative: no movement to describe
        self.assertEqual(self.lifts([{"classname": "func_door", "angle": "-1",
                                      "lip": "128"}]), [])

    def test_other_classnames_are_ignored(self):
        for classname in ("func_plat", "func_button", "func_wall",
                          "func_door_rotating"):
            with self.subTest(classname=classname):
                self.assertEqual(self.lifts([{"classname": classname,
                                              "angle": "-1"}]), [])

    def test_the_synthetic_entity_is_a_func_plat_with_no_origin(self):
        raw = a_mover_bsp([{"classname": "func_plat2", "lip": "8"}])
        data, count = bsp.add_plat_lifts(raw)
        self.assertEqual(count, 1)
        plats = bsp.loads(data).by_classname("func_plat")
        self.assertEqual(len(plats), 1)
        # an origin key would seed FloodEntities; Q2 model bounds are absolute
        self.assertNotIn("origin", plats[0])
        self.assertEqual(plats[0]["height"], "120.00")

    def test_a_plat2_needs_no_duplicate_model(self):
        # its brush bounds already describe the top position and its box top
        # is the ride surface
        raw = a_mover_bsp([{"classname": "func_plat2"}])
        data, _ = bsp.add_plat_lifts(raw)
        self.assertEqual(len(bsp.loads(data).models),
                         len(bsp.loads(raw).models))
        self.assertEqual(bsp.loads(data).by_classname("func_plat")[0]["model"],
                         "*1")

    def test_a_door_opening_up_gets_a_raised_duplicate(self):
        raw = a_mover_bsp([{"classname": "func_door", "angle": "-1"}])
        data, _ = bsp.add_plat_lifts(raw)
        after = bsp.loads(data)
        self.assertEqual(len(after.models), len(bsp.loads(raw).models) + 1)
        new = after.models[-1]
        self.assertAlmostEqual(new.maxs[2], 128.0 + 120.0)   # top of its travel
        self.assertEqual(after.by_classname("func_plat")[0]["model"],
                         "*%d" % (len(after.models) - 1))

    def test_a_car_shaped_door_rides_on_its_interior_floor(self):
        raw = a_mover_bsp([{"classname": "func_plat2"}],
                          face_heights=[128.0, 20.0])
        data, _ = bsp.add_plat_lifts(raw)
        after = bsp.loads(data)
        # a duplicate was needed because maxs.z is not the ride surface
        self.assertAlmostEqual(after.models[-1].maxs[2], 20.0)

    def test_a_map_with_no_candidate_is_just_stripped(self):
        raw = a_mover_bsp([{"classname": "func_plat"}])
        data, count = bsp.add_plat_lifts(raw)
        self.assertEqual(count, 0)
        self.assertEqual(bsp.loads(data).by_classname("func_plat")[0]["model"],
                         "*1")


class TestDoorMovers(unittest.TestCase):
    """BSPC assigns CONTENTS_MOVER to `func_door` alone, so it bakes every
    other door into the AAS as a permanent wall. A rotating or secret door
    swings out of its own volume and is a doorway, so it is presented as a
    func_door."""

    def a_door_bsp(self, classnames):
        text = b'{ "classname" "worldspawn" }\n'
        for i, name in enumerate(classnames):
            text += (b'{\n"origin" "0 0 0"\n"classname" "%s"\n"model" "*%d"\n}\n'
                     % (name.encode(), i + 1))
        models = a_model((-1, -1, -1), (1, 1, 1), headnode=0)
        for i in range(len(classnames)):
            models += a_model((0, 0, 0), (64, 64, 64), headnode=i + 1)
        return a_bsp({bsp.LUMP_ENTITIES: text + b"\0",
                      bsp.LUMP_MODELS: models})

    def test_rotating_and_secret_doors_are_renamed(self):
        raw = self.a_door_bsp(["func_door_rotating", "func_door_secret"])
        data, count = bsp.mark_doors_as_movers(raw)
        self.assertEqual(count, 2)
        after = bsp.loads(data)
        self.assertEqual(len(after.by_classname("func_door")), 2)
        self.assertEqual(after.by_classname("func_door_rotating"), [])
        self.assertEqual(after.by_classname("func_door_secret"), [])

    def test_other_solid_brush_entities_are_left_alone(self):
        # these really are walls until something happens to them
        names = ["func_wall", "func_explosive", "func_button", "func_plat",
                 "func_train", "func_rotating"]
        raw = self.a_door_bsp(names)
        data, count = bsp.mark_doors_as_movers(raw)
        self.assertEqual(count, 0)
        after = bsp.loads(data)
        for name in names:
            with self.subTest(classname=name):
                self.assertEqual(len(after.by_classname(name)), 1)

    def test_an_untouched_map_still_comes_back_stripped(self):
        raw = self.a_door_bsp(["func_wall"])
        data, count = bsp.mark_doors_as_movers(raw)
        self.assertEqual(count, 0)
        for lump in bsp.BSPC_UNUSED_LUMPS:
            _, length = struct.unpack_from("<ii", data, 8 + lump * 8)
            self.assertEqual(length, 0)

    def test_the_models_lump_survives(self):
        raw = self.a_door_bsp(["func_door_rotating"])
        data, _ = bsp.mark_doors_as_movers(raw)
        self.assertEqual(len(bsp.loads(data).models),
                         len(bsp.loads(raw).models))

    def test_it_composes_after_the_mover_rewrites(self):
        # order matters: renaming first would let plat_lift_entities read a
        # rotating door's `angle` as a move direction, which it is not
        text = (b'{ "classname" "worldspawn" }\n'
                b'{\n"classname" "func_door_rotating"\n"angle" "-1"\n'
                b'"model" "*1"\n}\n'
                b'{\n"classname" "func_plat2"\n"model" "*2"\n}\n')
        models = (a_model((-1, -1, -1), (1, 1, 1), headnode=0)
                  + a_model((0, 0, 0), (64, 64, 64), headnode=1)
                  + a_model((0, 0, 0), (64, 64, 128), headnode=2))
        raw = a_bsp({bsp.LUMP_ENTITIES: text + b"\0", bsp.LUMP_MODELS: models})
        movers, count = bsp.add_movers(raw, lifts=True)
        self.assertEqual(count, 1)              # the plat2 only, not the door
        doors, renamed = bsp.mark_doors_as_movers(movers)
        self.assertEqual(renamed, 1)
        after = bsp.loads(doors)
        self.assertEqual(len(after.by_classname("func_plat")), 1)
        self.assertEqual(len(after.by_classname("func_door")), 1)


class TestConvertGeometry(unittest.TestCase):
    def test_funnel_is_a_run_up_point_behind_start(self):
        start, end = (0.0, 0.0, 100.0), (128.0, 0.0, 100.0)
        funnel = convert.funnel_for(nav3.LinkType.LONG_JUMP, start, end)
        # 32 units back along the reverse of start->end, at start's height
        self.assertAlmostEqual(funnel[0], -32.0, places=4)
        self.assertAlmostEqual(funnel[1], 0.0, places=4)
        self.assertAlmostEqual(funnel[2], 100.0, places=4)

    def test_funnel_distance_is_per_link_type(self):
        start, end = (0.0, 0.0, 0.0), (100.0, 0.0, 0.0)
        for link_type, back in ((nav3.LinkType.BARRIER_JUMP, 8.0),
                                (nav3.LinkType.LONG_JUMP, 32.0),
                                (nav3.LinkType.ROCKET_JUMP, 48.0),
                                (nav3.LinkType.TRAIN, 16.0)):
            with self.subTest(link_type=link_type):
                f = convert.funnel_for(link_type, start, end)
                self.assertAlmostEqual(math.dist(f, start), back, places=4)

    def test_types_with_no_run_up_get_the_unset_sentinel(self):
        for link_type in (nav3.LinkType.LADDER, nav3.LinkType.WALK_OFF_LEDGE,
                          nav3.LinkType.ELEVATOR, nav3.LinkType.WALK):
            with self.subTest(link_type=link_type):
                f = convert.funnel_for(link_type, (0, 0, 0), (1, 0, 0))
                self.assertTrue(all(c >= nav3.UNSET_COORD for c in f))

    def test_a_purely_vertical_move_has_no_run_up_axis(self):
        f = convert.funnel_for(nav3.LinkType.LONG_JUMP,
                               (0.0, 0.0, 0.0), (0.0, 0.0, 128.0))
        self.assertTrue(all(c >= nav3.UNSET_COORD for c in f))

    def test_polygon_centroid_is_area_weighted(self):
        # a square with extra vertexes bunched along one edge: a plain vertex
        # average would be dragged toward them, an area-weighted one is not
        square = [(0, 0, 0), (10, 0, 0), (20, 0, 0), (30, 0, 0),
                  (30, 30, 0), (0, 30, 0)]
        cx, cy, _ = convert._polygon_centroid(square)
        self.assertAlmostEqual(cx, 15.0, places=4)
        self.assertAlmostEqual(cy, 15.0, places=4)

    def test_point_in_polygon(self):
        square = [(0, 0, 0), (64, 0, 0), (64, 64, 0), (0, 64, 0)]
        self.assertTrue(convert._point_in_polygon_xy((32, 32), square))
        self.assertFalse(convert._point_in_polygon_xy((96, 32), square))
        self.assertFalse(convert._point_in_polygon_xy((-1, 32), square))


class TestConvertTables(unittest.TestCase):
    """The tables are the interface between the two formats; a typo in one
    would produce a plausible file with the wrong moves in it."""

    def test_every_mapped_link_type_is_a_real_one(self):
        for travel, link in convert.TRAVEL_TO_LINK.items():
            with self.subTest(travel=travel):
                self.assertIsInstance(aas.TravelType(travel), aas.TravelType)
                self.assertIsInstance(nav3.LinkType(link), nav3.LinkType)

    def test_q3_only_moves_are_not_mapped(self):
        for travel in (aas.TravelType.BFGJUMP, aas.TravelType.GRAPPLEHOOK,
                       aas.TravelType.DOUBLEJUMP, aas.TravelType.RAMPJUMP,
                       aas.TravelType.STRAFEJUMP, aas.TravelType.INVALID):
            self.assertNotIn(travel, convert.TRAVEL_TO_LINK)

    def test_traversal_and_funnel_tables_agree(self):
        # a link type with a run-up distance must be one that carries a
        # traversal to put the funnel in
        for link_type in convert.FUNNEL_RUNUP:
            self.assertIn(link_type, convert.NEEDS_TRAVERSAL)

    def test_edict_types_carry_a_traversal_too(self):
        for link_type in convert.NEEDS_EDICT:
            self.assertIn(link_type, convert.NEEDS_TRAVERSAL)

    def test_teleport_needs_neither(self):
        self.assertNotIn(nav3.LinkType.TELEPORT, convert.NEEDS_TRAVERSAL)
        self.assertNotIn(nav3.LinkType.TELEPORT, convert.FUNNEL_RUNUP)

    def test_clearance_radius_does_not_exceed_the_separation(self):
        self.assertLessEqual(convert.CLEARANCE_MERGE_RADIUS,
                             convert.MIN_NODE_SEPARATION)

    def test_gaps_are_documented_pairs(self):
        self.assertTrue(convert.GAPS)
        for field, note in convert.GAPS:
            self.assertTrue(field and note)


class TestProximity(unittest.TestCase):
    def test_finds_a_node_inside_the_radius_and_not_outside(self):
        origins = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]
        p = convert._Proximity(64.0)
        for i, o in enumerate(origins):
            p.add(i, o)
        self.assertEqual(p.find(origins, (10.0, 0.0, 0.0)), 0)
        self.assertEqual(p.find(origins, (95.0, 0.0, 0.0)), 1)
        self.assertIsNone(p.find(origins, (500.0, 0.0, 0.0)))

    def test_returns_the_nearest_when_several_qualify(self):
        origins = [(0.0, 0.0, 0.0), (20.0, 0.0, 0.0)]
        p = convert._Proximity(64.0)
        for i, o in enumerate(origins):
            p.add(i, o)
        self.assertEqual(p.find(origins, (18.0, 0.0, 0.0)), 1)

    def test_the_allowed_set_gates_the_answer(self):
        origins = [(0.0, 0.0, 0.0)]
        node_areas = [{7}]
        p = convert._Proximity(64.0)
        p.add(0, origins[0])
        self.assertEqual(p.find(origins, (8.0, 0, 0), node_areas, {7}), 0)
        self.assertIsNone(p.find(origins, (8.0, 0, 0), node_areas, {9}))


class TestPolygonArea(unittest.TestCase):
    def test_shoelace_is_orientation_independent(self):
        square = [(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0)]
        self.assertAlmostEqual(aas._polygon_area_xy(square), 200.0)
        self.assertAlmostEqual(aas._polygon_area_xy(square[::-1]), 200.0)

    def test_a_degenerate_polygon_has_no_area(self):
        self.assertAlmostEqual(
            aas._polygon_area_xy([(0, 0, 0), (10, 0, 0), (20, 0, 0)]), 0.0)


class SolidBelow:
    """The smallest thing :func:`kexnav.convert.push_trajectory` needs: an
    object answering ``point_area_num``. Everything inside `bounds` is area 1
    and everything else is solid."""

    def __init__(self, bounds):
        self.bounds = bounds

    def point_area_num(self, point):
        lo, hi = self.bounds
        return 1 if all(lo[i] <= point[i] <= hi[i] for i in range(3)) else 0


class TestPushVelocity(unittest.TestCase):
    """Quake II's trigger_push has no target entity for BSPC to aim at, so the
    launch velocity is read off the entity the way G_SetMovedir does."""

    def test_angle_minus_one_is_straight_up(self):
        v = convert.push_velocity({"angle": "-1", "speed": "100"})
        self.assertEqual(tuple(round(c, 6) for c in v), (0.0, 0.0, 1000.0))

    def test_angle_minus_two_is_straight_down(self):
        v = convert.push_velocity({"angle": "-2", "speed": "100"})
        self.assertEqual(tuple(round(c, 6) for c in v), (0.0, 0.0, -1000.0))

    def test_speed_is_scaled_by_ten_and_defaults_to_a_thousand(self):
        # SP_trigger_push: speed defaults to 1000, trigger_push_touch scales
        # it by 10
        v = convert.push_velocity({"angle": "-1"})
        self.assertAlmostEqual(v[2], 10000.0)

    def test_a_yaw_pushes_horizontally(self):
        v = convert.push_velocity({"angle": "90", "speed": "10"})
        self.assertAlmostEqual(v[0], 0.0, places=6)
        self.assertAlmostEqual(v[1], 100.0, places=6)
        self.assertAlmostEqual(v[2], 0.0, places=6)

    def test_an_angles_pitch_of_minus_ninety_is_up(self):
        v = convert.push_velocity({"angles": "-90 0 0", "speed": "10"})
        self.assertAlmostEqual(v[2], 100.0, places=4)
        self.assertAlmostEqual(math.hypot(v[0], v[1]), 0.0, places=4)

    def test_a_speed_of_zero_still_gets_the_default(self):
        self.assertIsNotNone(convert.push_velocity({"angle": "-1",
                                                    "speed": "0"}))


class TestPushGeometry(unittest.TestCase):
    def test_a_vertical_pad_is_entered_in_the_middle(self):
        # its slight horizontal lean says nothing about which side you walk on
        entry = convert.push_entry_point((0, 0, 0), (64, 64, 16),
                                         (40.0, 0.0, 780.0))
        self.assertEqual(entry, (32.0, 32.0, 0.0))

    def test_a_wind_tunnel_is_entered_at_its_upwind_end(self):
        entry = convert.push_entry_point((0, 0, 0), (512, 64, 64),
                                         (-1000.0, 0.0, 0.0))
        self.assertEqual(entry, (512.0, 32.0, 0.0))
        entry = convert.push_entry_point((0, 0, 0), (512, 64, 64),
                                         (1000.0, 0.0, 0.0))
        self.assertEqual(entry, (0.0, 32.0, 0.0))

    def test_only_the_dominant_horizontal_axis_moves_to_a_face(self):
        # a push five degrees off the x axis is still entered along x, not at
        # the corner
        entry = convert.push_entry_point((0, 0, 0), (512, 64, 64),
                                         (996.0, 87.0, 0.0))
        self.assertEqual(entry, (0.0, 32.0, 0.0))

    def test_a_downward_push_is_entered_at_the_ceiling(self):
        entry = convert.push_entry_point((0, 0, 0), (64, 64, 128),
                                         (0.0, 0.0, -1000.0))
        self.assertEqual(entry[2], 128.0)

    def test_the_origin_box_is_the_trigger_reflected_by_the_player_box(self):
        lo, hi = convert._origin_box((0, 0, 0), (64, 64, 64))
        # an origin touches the trigger from PLAYER_MAXS before it and until
        # PLAYER_MINS past it
        self.assertEqual(lo, (-16.0, -16.0, -32.0))
        self.assertEqual(hi, (80.0, 80.0, 88.0))


class TestHorizontalVelocityForJump(unittest.TestCase):
    """AAS_HorizontalVelocityForJump, reproduced -- it is what decides which
    ledges a bot can steer to off a vertical pad."""

    def test_a_target_at_the_launch_height_takes_the_whole_flight(self):
        # 800 up under gravity 800: one second to the apex, one back down
        speed = convert.horizontal_velocity_for_jump(
            800.0, (0.0, 0.0, 0.0), (200.0, 0.0, 0.0))
        self.assertAlmostEqual(speed, 100.0, places=4)

    def test_a_target_above_the_apex_is_unreachable(self):
        # 800 up reaches 400; asking for 500 must fail rather than clamp
        self.assertIsNone(convert.horizontal_velocity_for_jump(
            800.0, (0.0, 0.0, 0.0), (10.0, 0.0, 500.0)))

    def test_a_target_needing_more_than_phys_maxvelocity_is_rejected(self):
        self.assertIsNone(convert.horizontal_velocity_for_jump(
            800.0, (0.0, 0.0, 0.0), (10000.0, 0.0, 0.0)))

    def test_the_apex_itself_needs_no_horizontal_speed(self):
        self.assertAlmostEqual(convert.horizontal_velocity_for_jump(
            800.0, (0.0, 0.0, 0.0), (0.0, 0.0, 400.0)), 0.0, places=4)


class TestPushTrajectory(unittest.TestCase):
    def test_the_trigger_re_applies_its_velocity_while_you_are_inside(self):
        # a tall shaft carries a player at a constant speed rather than
        # launching them ballistically, so they leave the top rather than
        # falling back
        world = SolidBelow(((-64, -64, 0), (64, 64, 512)))
        box = ((-64, -64, 0), (64, 64, 400))
        end, _ = convert.push_trajectory(world, (0.0, 0.0, 8.0),
                                         (0.0, 0.0, 200.0), box)
        self.assertGreater(end[2], 400.0)

    def test_the_ride_and_the_flight_after_it_end_in_different_places(self):
        # the whole reason a shaft yields two landings: with 200 up in a
        # 400-tall volume the ride ends at its top and the flight carries on
        world = SolidBelow(((-64, -64, 0), (64, 64, 4096)))
        box = ((-64, -64, 0), (64, 64, 400))
        end, ridden = convert.push_trajectory(world, (0.0, 0.0, 8.0),
                                              (0.0, 0.0, 200.0), box)
        self.assertGreater(ridden[2], 380.0)
        self.assertGreater(end[2], ridden[2])

    def test_a_thin_pad_rides_nowhere(self):
        world = SolidBelow(((-64, -64, 0), (64, 64, 4096)))
        box = ((-64, -64, 0), (64, 64, 8))
        end, ridden = convert.push_trajectory(world, (0.0, 0.0, 8.0),
                                              (0.0, 0.0, 800.0), box)
        self.assertLess(ridden[2], 40.0)
        self.assertGreater(end[2], ridden[2])

    def test_gravity_acts_once_you_are_out_of_the_volume(self):
        world = SolidBelow(((-64, -64, 0), (64, 64, 4096)))
        box = ((-64, -64, 0), (64, 64, 8))
        end, _ = convert.push_trajectory(world, (0.0, 0.0, 8.0),
                                         (0.0, 0.0, 800.0), box)
        # apex is v^2/2g = 400 above the launch, then it comes back down
        self.assertLess(end[2], 500.0)

    def test_it_stops_at_solid(self):
        world = SolidBelow(((-64, -64, 0), (64, 64, 100)))
        end, _ = convert.push_trajectory(world, (0.0, 0.0, 8.0),
                                         (0.0, 0.0, 8000.0),
                                         ((-64, -64, 0), (64, 64, 4096)))
        self.assertLessEqual(end[2], 100.0)

    def test_an_absurd_speed_terminates(self):
        # an arena set carries pads with speed 250000; the path cap is what
        # stops that integrating for a million steps
        world = SolidBelow(((-1e9, -1e9, -1e9), (1e9, 1e9, 1e9)))
        end, _ = convert.push_trajectory(world, (0.0, 0.0, 0.0),
                                         (0.0, 0.0, 2.5e6),
                                         ((-1, -1, -1), (1, 1, 1)))
        self.assertLessEqual(end[2], convert.PUSH_MAX_PATH + 1.0)


def an_aas(settings, reachabilities):
    """An AasFile carrying only what usable_areas reads: one areasettings row
    per area, and a flat reachability lump they index into."""
    a = aas.AasFile()
    a.areas = [aas.Area(i, 0, 0, (0, 0, 0), (0, 0, 0), (0, 0, 0))
               for i in range(len(settings))]
    a.areasettings = settings
    a.reachability = reachabilities
    return a


def area_settings(flags=0, first=0, count=0):
    return aas.AreaSettings(contents=0, areaflags=flags,
                            presencetype=int(aas.PresenceType.NORMAL),
                            cluster=0, clusterareanum=0,
                            numreachableareas=count, firstreachablearea=first)


def a_reach(areanum, travel=aas.TravelType.WALK):
    return aas.Reachability(areanum, 0, 0, (0, 0, 0), (0, 0, 0), int(travel), 1)


class TestUsableAreas(unittest.TestCase):
    """A node can only sit in an area usable_areas returns, so anything it
    leaves out silently drops every reachability that ends there."""

    def test_grounded_liquid_and_ladder_areas_are_usable(self):
        a = an_aas([area_settings(),                       # 0, the dummy
                    area_settings(aas.AreaFlags.GROUNDED),
                    area_settings(aas.AreaFlags.LIQUID),
                    area_settings(aas.AreaFlags.LADDER),
                    area_settings()], [])
        self.assertEqual(convert.usable_areas(a), [1, 2, 3])

    def test_a_reachability_endpoint_is_usable_whatever_its_flags(self):
        # the standing spot on a raised plat is routinely a mover area with no
        # ground face of its own, and BSPC's own simulation says a player gets
        # there
        a = an_aas([area_settings(),
                    area_settings(aas.AreaFlags.GROUNDED, first=0, count=1),
                    area_settings()],
                   [a_reach(2, aas.TravelType.ELEVATOR)])
        self.assertEqual(convert.usable_areas(a), [1, 2])

    def test_area_zero_is_never_usable(self):
        a = an_aas([area_settings(aas.AreaFlags.GROUNDED, first=0, count=1),
                    area_settings(aas.AreaFlags.GROUNDED)],
                   [a_reach(0)])
        self.assertNotIn(0, convert.usable_areas(a))


class TestNofill(unittest.TestCase):
    """A `_nofill` entity is one the map's own compiler was told to ignore when
    deciding what is inside the map. BSPC floods from it anyway and leaks."""

    def _bsp(self, entities):
        text = b"".join(
            b"{\n" + b"".join(b'"%s" "%s"\n' % (k.encode(), v.encode())
                              for k, v in e.items()) + b"}\n"
            for e in entities) + b"\0"
        return a_bsp({bsp.LUMP_ENTITIES: text})

    def test_a_nofill_entity_is_dropped(self):
        data = self._bsp([{"classname": "worldspawn"},
                          {"classname": "path_corner", "origin": "1 2 3",
                           "_nofill": "1"},
                          {"classname": "light", "origin": "4 5 6"}])
        kept = bsp.loads(bsp.strip_for_bspc(data)).entities
        self.assertEqual([e["classname"] for e in kept],
                         ["worldspawn", "light"])

    def test_nofill_zero_is_kept(self):
        data = self._bsp([{"classname": "light", "_nofill": "0"}])
        self.assertEqual(len(bsp.loads(bsp.strip_for_bspc(data)).entities), 1)

    def test_the_other_entities_survive_byte_for_byte(self):
        entities = [{"classname": "worldspawn", "message": "a map"},
                    {"classname": "info_player_start", "origin": "0 0 24"},
                    {"classname": "path_corner", "origin": "1 2 3",
                     "_nofill": "1"},
                    {"classname": "func_door", "model": "*1", "angle": "-1"}]
        kept = bsp.loads(bsp.strip_for_bspc(self._bsp(entities))).entities
        self.assertEqual(kept, [e for e in entities if "_nofill" not in e])

    def test_a_map_without_one_is_handed_over_untouched(self):
        data = self._bsp([{"classname": "worldspawn"}])
        self.assertIs(bsp.strip_for_bspc(data, drop=()), data)

    def test_drop_nofill_leaves_text_with_none_alone(self):
        text = b'{\n"classname" "light"\n}\n'
        self.assertIs(bsp.drop_nofill(text), text)


class TestScratchName(unittest.TestCase):
    """A map name keeps its directory everywhere -- the cache, the output file
    and the deploy path -- because six basenames in the retail pak belong to
    two different maps and 33 of Nightdive's nav files live in a
    subdirectory."""

    def test_a_directory_is_flattened_only_for_a_scratch_prefix(self):
        self.assertEqual(generate._scratch_name("q64/command"), "q64_command")
        self.assertEqual(generate._scratch_name("q2dm1"), "q2dm1")


class TestWaterLifts(unittest.TestCase):
    """SP_func_water builds a door in every respect plat_lift_entities cares
    about, except that its lip defaults to 0 rather than 8."""

    def lifts(self, entities, **kw):
        return bsp.plat_lift_entities(bsp.loads(a_mover_bsp(entities, **kw)))

    def test_a_vertical_func_water_is_a_lift(self):
        lifts = self.lifts([{"classname": "func_water", "angle": "-1"}])
        self.assertEqual(len(lifts), 1)
        self.assertEqual(lifts[0][3], "func_water")

    def test_its_lip_defaults_to_zero_where_a_door_gets_eight(self):
        water = self.lifts([{"classname": "func_water", "angle": "-1"}])
        door = self.lifts([{"classname": "func_door", "angle": "-1"}])
        self.assertAlmostEqual(water[0][2], 128.0)
        self.assertAlmostEqual(door[0][2], 120.0)

    def test_a_horizontal_func_water_is_not_a_lift(self):
        self.assertEqual(self.lifts([{"classname": "func_water",
                                      "angle": "90"}]), [])


class TestOracleMismatch(unittest.TestCase):
    """One of Nightdive's 174 nav files was built for a different revision of
    the map it is named after, and counting it as a coverage failure blames the
    generator for something it cannot fix."""

    def _bsp(self, mins, maxs):
        models = a_model(mins, maxs, headnode=0)
        return bsp.loads(a_bsp({bsp.LUMP_MODELS: models,
                                bsp.LUMP_ENTITIES: b'{ "classname" "worldspawn" }\0'}))

    def _nav(self, origins):
        nav = nav3.NavFile()
        for o in origins:
            nav.nodes.append(nav3.Node(flags=0, num_links=0, first_link=0,
                                       radius=32, origin=o))
        return nav

    def test_nodes_inside_the_world_are_no_mismatch(self):
        b = self._bsp((-100, -100, -100), (100, 100, 100))
        self.assertEqual(
            check.oracle_mismatch(b, self._nav([(0, 0, 0), (90, 90, 90)])), 0)

    def test_a_node_beyond_the_tolerance_counts(self):
        b = self._bsp((-100, -100, -100), (100, 100, 100))
        self.assertEqual(
            check.oracle_mismatch(b, self._nav([(0, 0, 0), (1000, 0, 0)])), 1)

    def test_the_tolerance_forgives_a_node_just_outside(self):
        b = self._bsp((-100, -100, -100), (100, 100, 100))
        self.assertEqual(
            check.oracle_mismatch(b, self._nav([(132, 0, 0)])), 0)
        self.assertEqual(
            check.oracle_mismatch(b, self._nav([(200, 0, 0)])), 1)

    def test_an_empty_nav_is_not_a_mismatch(self):
        b = self._bsp((-1, -1, -1), (1, 1, 1))
        self.assertEqual(check.oracle_mismatch(b, self._nav([])), 0)


class TestConnectivity(unittest.TestCase):
    """The nav-level measure. An AAS-level score can rank a variant higher for
    having more areas while the graph it produces is less connected."""

    def graph(self, edges, count):
        nav = nav3.NavFile()
        by_source = {}
        for src, dst in edges:
            by_source.setdefault(src, []).append(dst)
        for i in range(count):
            targets = by_source.get(i, [])
            nav.nodes.append(nav3.Node(flags=0, num_links=len(targets),
                                       first_link=len(nav.links), radius=32,
                                       origin=(float(i), 0.0, 0.0)))
            for dst in targets:
                nav.links.append(nav3.Link(target=dst, type=0, flags=3,
                                           traversal=nav3.NO_TRAVERSAL))
        return nav

    def test_two_islands(self):
        nav = self.graph([(0, 1), (1, 0), (2, 3), (3, 2)], 4)
        self.assertEqual(check.connectivity(nav), (2, 2))

    def test_a_one_way_link_is_one_island_but_not_mutually_reachable(self):
        nav = self.graph([(0, 1)], 2)
        self.assertEqual(check.connectivity(nav), (1, 1))

    def test_a_cycle_is_wholly_mutually_reachable(self):
        nav = self.graph([(0, 1), (1, 2), (2, 0)], 3)
        self.assertEqual(check.connectivity(nav), (1, 3))

    def test_an_empty_graph(self):
        self.assertEqual(check.connectivity(nav3.NavFile()), (0, 0))


class TestLocalCandidates(unittest.TestCase):
    """``kexnav.local`` is how a machine states its own paths without the
    checkout carrying them. It is optional by design, so every malformed case
    has to degrade to "no override" rather than raise."""

    def parse(self, text, what="pak"):
        path = os.path.join(self.tmp, "kexnav.local")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return env.local_candidates(what, path)

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_a_key_may_repeat_and_order_is_kept(self):
        self.assertEqual(self.parse("pak /a/0.pak\npak /b/0.pak\n"),
                         ["/a/0.pak", "/b/0.pak"])

    def test_other_keys_are_ignored(self):
        self.assertEqual(self.parse("bspc /b/bspc\npak /a/0.pak\n"),
                         ["/a/0.pak"])
        self.assertEqual(self.parse("bspc /b/bspc\npak /a/0.pak\n", "bspc"),
                         ["/b/bspc"])

    def test_tilde_is_expanded(self):
        self.assertEqual(self.parse("pak ~/q2/0.pak\n"),
                         [os.path.expanduser("~/q2/0.pak")])

    def test_blanks_comments_and_trailing_comments(self):
        self.assertEqual(
            self.parse("\n# a whole line\n  pak  /a/0.pak   # why\n\n"),
            ["/a/0.pak"])

    def test_a_hash_inside_a_path_is_not_a_comment(self):
        self.assertEqual(self.parse("pak /a/q2#1/0.pak\n"), ["/a/q2#1/0.pak"])

    def test_a_key_with_no_path_is_skipped(self):
        self.assertEqual(self.parse("pak\npak /a/0.pak\n"), ["/a/0.pak"])

    def test_a_missing_file_is_not_an_error(self):
        self.assertEqual(env.local_candidates("pak", "/nonexistent/kexnav.local"),
                         [])


if __name__ == "__main__":
    unittest.main()

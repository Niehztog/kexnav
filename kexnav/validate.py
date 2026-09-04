"""Structural checks a nav file must pass, whoever wrote it.

Every one of these is a property all 174 shipped files hold, established by
``python3 -m tests.roundtrip`` across 66424 nodes, 224836 links, 12511 traversals and 1120
nav edicts. They live here rather than in the round-trip script so that a
*generated* file is held to the identical standard -- if the corpus satisfies
an invariant and a generator does not, that is a bug in the generator, and it
should be one check reporting it rather than two implementations drifting.

``kexnav.nav3`` deliberately does not enforce any of this: it mirrors the file
so that a parse/serialise cycle is byte-exact even for a file that breaks a
rule. Enforcement is this module's job.
"""

from . import nav3


def check_link_ranges(nav):
    """``firstLink`` is a running sum of the preceding ``numLinks`` -- a CSR
    layout -- so links are stored grouped by node, in node order, and the
    field is derived rather than authored."""
    problems = []
    running = 0
    for i, node in enumerate(nav.nodes):
        if node.first_link != running:
            problems.append(f"node {i}: first_link {node.first_link}, "
                            f"expected {running}")
            running = node.first_link
        running += node.num_links
    if running != len(nav.links):
        problems.append(f"node link ranges cover {running} links, "
                        f"file has {len(nav.links)}")
    return problems


def check_indices(nav):
    """Every index that points into another array is in range."""
    problems = []
    for i, link in enumerate(nav.links):
        if link.target >= len(nav.nodes):
            problems.append(f"link {i}: target {link.target} out of range "
                            f"({len(nav.nodes)} nodes)")
        if link.has_traversal and link.traversal >= len(nav.traversals):
            problems.append(f"link {i}: traversal {link.traversal} out of range "
                            f"({len(nav.traversals)} traversals)")
    for i, e in enumerate(nav.edicts):
        if e.link >= len(nav.links):
            problems.append(f"nav edict {i}: link {e.link} out of range "
                            f"({len(nav.links)} links)")
    return problems


def check_traversal_ownership(nav):
    """Every traversal is referenced by exactly one link. Never shared,
    never orphaned -- no exceptions in the corpus."""
    problems = []
    owners = [0] * len(nav.traversals)
    for link in nav.links:
        if link.has_traversal and link.traversal < len(owners):
            owners[link.traversal] += 1
    shared = sum(1 for n in owners if n > 1)
    orphans = sum(1 for n in owners if n == 0)
    if shared:
        problems.append(f"{shared} traversal(s) referenced by more than one link")
    if orphans:
        problems.append(f"{orphans} traversal(s) referenced by no link")
    return problems


def check_ladder_planes(nav):
    """The traversal's fourth vec3 is non-zero for exactly the traversals a
    ``LADDER`` link points at, and zero for every other one. Also a unit
    vector: length 1.0 in all 264 corpus ladder traversals, so it is the
    ladder's surface normal rather than a plane equation."""
    problems = []
    ladders = {link.traversal for link in nav.links
               if link.has_traversal and link.type == nav3.LinkType.LADDER}
    for i, t in enumerate(nav.traversals):
        if t.ladder_plane is None:
            continue                       # version 2 or 3, no such field
        non_zero = any(t.ladder_plane)
        is_ladder = i in ladders
        if non_zero != is_ladder:
            problems.append(
                f"traversal {i}: ladder plane "
                f"{'set' if non_zero else 'unset'} but "
                f"{'a' if is_ladder else 'no'} LADDER link points at it")
        elif non_zero:
            length = sum(c * c for c in t.ladder_plane) ** 0.5
            if abs(length - 1.0) > 1e-3:
                problems.append(f"traversal {i}: ladder plane length {length:.4f}, "
                                f"expected a unit vector")
    return problems


def check_type_traversal_rule(nav):
    """Whether a link type carries a traversal is fixed per type, not "anything
    but WALK". In 174 files ``WALK``, ``TELEPORT``, ``PUSHER``, ``CROUCH`` and
    ``PIVOT_AND_JUMP`` never carry one; the jumps, ``LADDER``, ``ELEVATOR`` and
    ``TRAIN`` always do. ``WALK_OFF_LEDGE`` is genuinely mixed and is not
    checked.

    The ``LADDER`` exception is real: the 8 ladder links with no traversal are
    all in version 2 files, which predate the ladder plane.
    """
    never = {nav3.LinkType.WALK, nav3.LinkType.TELEPORT, nav3.LinkType.PUSHER,
             nav3.LinkType.CROUCH, nav3.LinkType.PIVOT_AND_JUMP}
    always = {nav3.LinkType.LONG_JUMP, nav3.LinkType.MANUAL_LONG_JUMP,
              nav3.LinkType.BARRIER_JUMP, nav3.LinkType.MANUAL_BARRIER_JUMP,
              nav3.LinkType.ROCKET_JUMP, nav3.LinkType.ELEVATOR,
              nav3.LinkType.TRAIN}
    if nav.version >= nav3.VERSION_LADDER_PLANE:
        always = always | {nav3.LinkType.LADDER}

    problems = []
    for i, link in enumerate(nav.links):
        if link.type in never and link.has_traversal:
            problems.append(f"link {i}: {link.type_name} carries a traversal, "
                            f"which no shipped file does")
        elif link.type in always and not link.has_traversal:
            problems.append(f"link {i}: {link.type_name} has no traversal, "
                            f"which every shipped one has")
    return problems


def check_format_limits(nav):
    """The node record's ``numLinks``, ``firstLink`` and ``radius``, and every
    ``link.target``, are u16. A hand-authored file never came close -- the
    largest is base64.nav at 1167 nodes and 4138 links -- but a generated
    graph is finer, and silently truncating would write a corrupt file."""
    problems = []
    if len(nav.nodes) > 0xFFFF:
        problems.append(f"{len(nav.nodes)} nodes: a link target is u16, "
                        f"maximum {0xFFFF}")
    if len(nav.links) > 0xFFFF:
        problems.append(f"{len(nav.links)} links: first_link is u16, "
                        f"maximum {0xFFFF}")
    for i, n in enumerate(nav.nodes):
        if n.radius > 0xFFFF or n.num_links > 0xFFFF:
            problems.append(f"node {i}: radius or link count over the u16 field")
    return problems


#: Every check, in the order a report should run them.
CHECKS = (
    ("link ranges", check_link_ranges),
    ("indices", check_indices),
    ("traversal ownership", check_traversal_ownership),
    ("ladder planes", check_ladder_planes),
    ("type/traversal rule", check_type_traversal_rule),
    ("format limits", check_format_limits),
)


def check(nav, skip=()):
    """Run every check. Returns a list of human-readable violations.

    `skip` names checks to leave out -- ``python3 -m tests.roundtrip`` skips the
    type/traversal rule, because it is a corpus *observation* being reported
    there rather than a rule the corpus is being tested against.
    """
    problems = []
    for name, fn in CHECKS:
        if name in skip:
            continue
        problems.extend(fn(nav))
    return problems

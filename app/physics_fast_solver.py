
import math

WALL_CATEGORIES = {'wall_art', 'mirror', 'curtain', 'window', 'door', 'wall_planter', 'wall_vase'}
CEILING_CATEGORIES = {'ceiling_lamp', 'hanging_planter'}
FLOOR_UNDER_CATEGORIES = {'rug'}
SUPPORTER_CATEGORIES = {'coffee_table', 'dining_table', 'desk', 'nightstand', 'cabinet', 'shelf', 'tv_stand', 'counter'}
NEEDS_SUPPORT_CATEGORIES = {'tv', 'vase', 'book', 'decor', 'lamp', 'plant', 'hanging_planter', 'wall_vase'}


CATEGORY_ALIASES = {
    'flower_vase': 'vase', 'table_vase': 'vase', 'decor_vase': 'vase', 'vase_decor': 'vase',
    'hanging planter': 'hanging_planter', 'hanging plant': 'hanging_planter',
    'wall planter': 'wall_planter', 'wall plant': 'wall_planter',
    'ceiling light': 'ceiling_lamp', 'ceiling lamp': 'ceiling_lamp', 'pendant light': 'ceiling_lamp',
    'pendant lamp': 'ceiling_lamp', 'pendant': 'ceiling_lamp', 'chandelier': 'ceiling_lamp',
    'hanging light': 'ceiling_lamp', 'hanging lamp': 'ceiling_lamp',
    'table lamp': 'lamp', 'floor lamp': 'lamp', 'desk lamp': 'lamp'
}


def canonical_category(value):
    s = str(value or 'unknown').strip().lower().replace('-', ' ').replace('_', ' ')
    return CATEGORY_ALIASES.get(s, CATEGORY_ALIASES.get(s.replace(' ', '_'), s.replace(' ', '_')))


def fp(i):
    f = i.get('footprint') or {}
    return float(f.get('widthM', i.get('widthM', .5))), float(f.get('depthM', i.get('depthM', .5))), float(f.get('heightM', i.get('heightM', .5)))


def clamp(v, a, b):
    return max(a, min(b, v))


def norm_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def rotated_dims(i):
    w, d, _ = fp(i)
    r = float(i.get('rotationY', 0.0))
    c = abs(math.cos(r))
    s = abs(math.sin(r))
    return w * c + d * s, w * s + d * c


def snap_inside(i, room):
    rw = float(room.get('widthM', 4))
    rl = float(room.get('lengthM', 5))
    w, d = rotated_dims(i)
    i['x'] = clamp(float(i.get('x', rw / 2)), w / 2, max(w / 2, rw - w / 2))
    i['z'] = clamp(float(i.get('z', rl / 2)), d / 2, max(d / 2, rl - d / 2))
    return i


def aabb(i):
    w, d = rotated_dims(i)
    x = float(i.get('x', 0))
    z = float(i.get('z', 0))
    return x - w / 2, z - d / 2, x + w / 2, z + d / 2


def overlap(a, b):
    ax1, az1, ax2, az2 = aabb(a)
    bx1, bz1, bx2, bz2 = aabb(b)
    dx = min(ax2, bx2) - max(ax1, bx1)
    dz = min(az2, bz2) - max(az1, bz1)
    return max(0, dx) * max(0, dz)


def snap_wall(i, room, wall=None):
    rw = float(room.get('widthM', 4))
    rl = float(room.get('lengthM', 5))
    w, d, _ = fp(i)
    wall = wall or i.get('wallAnchor') or 'front'
    if wall == 'left':
        i['x'] = w / 2
        i['rotationY'] = math.pi / 2
    elif wall == 'right':
        i['x'] = rw - w / 2
        i['rotationY'] = -math.pi / 2
    elif wall == 'back':
        i['z'] = rl - d / 2
        i['rotationY'] = math.pi
    else:
        i['z'] = d / 2
        i['rotationY'] = 0
        wall = 'front'
    i['wallAnchor'] = wall
    return snap_inside(i, room)


def find_support(item, items):
    iw, id_, _ = fp(item)
    cand = []
    for s in items:
        if s.get('productId') == item.get('productId'):
            continue
        if str(s.get('category', '')).lower() not in SUPPORTER_CATEGORIES and not s.get('isSupporter', False):
            continue
        sw, sd, _ = fp(s)
        if sw >= iw and sd >= id_:
            cand.append((((float(item.get('x', 0)) - float(s.get('x', 0))) ** 2 + (float(item.get('z', 0)) - float(s.get('z', 0))) ** 2), s))
    return sorted(cand, key=lambda x: x[0])[0][1] if cand else None


def resolve_collisions(items, room):
    floor = [x for x in items if x.get('layer', 'floor') == 'floor']
    for _ in range(80):
        moved = False
        for a_i in range(len(floor)):
            for b_i in range(a_i + 1, len(floor)):
                a, b = floor[a_i], floor[b_i]
                ov = overlap(a, b)
                if ov <= 1e-3:
                    continue
                dx = float(b.get('x', 0)) - float(a.get('x', 0))
                dz = float(b.get('z', 0)) - float(a.get('z', 0))
                n = (dx * dx + dz * dz) ** .5 or 1
                step = min(.2, ov ** .5 + .02)
                b['x'] = float(b.get('x', 0)) + dx / n * step
                b['z'] = float(b.get('z', 0)) + dz / n * step
                snap_inside(b, room)
                moved = True
        if not moved:
            break
    return items


def score(items, room):
    pen = {'collision': 0, 'unsupported': 0, 'outside': 0, 'wall': 0}
    rw = float(room.get('widthM', 4))
    rl = float(room.get('lengthM', 5))
    for it in items:
        cat = canonical_category(it.get('category', ''))
        w, d = rotated_dims(it)
        x, z = float(it.get('x', 0)), float(it.get('z', 0))
        if x - w / 2 < 0 or z - d / 2 < 0 or x + w / 2 > rw or z + d / 2 > rl:
            pen['outside'] += 1
        if cat in NEEDS_SUPPORT_CATEGORIES and it.get('layer') != 'wall' and not it.get('supportParentId'):
            pen['unsupported'] += 1
        if cat in WALL_CATEGORIES and not it.get('wallAnchor'):
            pen['wall'] += 1
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i].get('layer') == 'floor' and items[j].get('layer') == 'floor' and overlap(items[i], items[j]) > 1e-3:
                pen['collision'] += 1
    return max(0, 1 - .1 * pen['collision'] - .12 * pen['unsupported'] - .08 * pen['outside'] - .05 * pen['wall']), pen


def _clearance_penalty_for_room(room_type, item):
    cat = canonical_category(item.get('category', ''))
    if room_type == 'bedroom':
        return 0.18 if cat in {'bed', 'wardrobe', 'bench'} else 0.08
    if room_type in {'living_room', 'livingroom'}:
        return 0.18 if cat in {'sofa', 'tv', 'coffee_table'} else 0.08
    if room_type == 'kitchen':
        return 0.2 if cat in {'counter', 'sink', 'stove', 'fridge'} else 0.1
    if room_type == 'bathroom':
        return 0.2 if cat in {'toilet', 'sink', 'bathtub', 'shower'} else 0.1
    return 0.12


def fast_solve_layout(room, raw_items):
    room_type = str(room.get('type', room.get('room_type', 'unknown'))).lower()
    if room_type == 'livingroom':
        room_type = 'living_room'
    items = [dict(x) for x in raw_items]
    rejected = []
    min_clearance = {'bedroom': 0.5, 'living_room': 0.45, 'kitchen': 0.55, 'bathroom': 0.5}.get(room_type, 0.45)
    for it in items:
        cat = canonical_category(it.get('category', 'unknown'))
        it.setdefault('footprint', {'widthM': float(it.get('widthM', .5)), 'depthM': float(it.get('depthM', .5)), 'heightM': float(it.get('heightM', .5))})
        it['rotationY'] = norm_angle(float(it.get('rotationY', 0.0)))
        if cat in WALL_CATEGORIES:
            it['layer'] = 'wall'
        elif cat in CEILING_CATEGORIES:
            it['layer'] = 'ceiling'
        elif cat in FLOOR_UNDER_CATEGORIES:
            it['layer'] = 'floor_under'
        elif cat in NEEDS_SUPPORT_CATEGORIES:
            it.setdefault('layer', 'top_surface')
        else:
            it.setdefault('layer', 'floor')
        snap_inside(it, room)
    for it in items:
        cat = str(it.get('category', '')).lower()
        if it.get('layer') == 'wall':
            snap_wall(it, room)
            it['y'] = max(float(it.get('y', 1.4)), 1.2)
        elif it.get('layer') == 'ceiling':
            it['x'] = float(room.get('widthM', 4)) / 2
            it['z'] = float(room.get('lengthM', 5)) / 2
            it['y'] = float(room.get('heightM', 2.8)) - fp(it)[2] / 2 - 0.12
            it['rotationY'] = 0.0
        elif cat in {'tv_stand', 'wall_vase', 'wall_planter'}:
            it['layer'] = 'wall' if cat in {'wall_vase', 'wall_planter'} else 'floor'
            if cat in {'wall_vase', 'wall_planter'}:
                snap_wall(it, room)
            else:
                snap_inside(it, room)
    for it in items:
        cat = str(it.get('category', '')).lower()
        if cat in FLOOR_UNDER_CATEGORIES:
            it['x'] = float(room.get('widthM', 4)) / 2
            it['z'] = float(room.get('lengthM', 5)) / 2
            it['y'] = .01
            snap_inside(it, room)
        if cat in CEILING_CATEGORIES:
            it['x'] = float(room.get('widthM', 4)) / 2
            it['z'] = float(room.get('lengthM', 5)) / 2
            it['y'] = float(room.get('heightM', 2.8)) - fp(it)[2] / 2 - 0.12
        if cat in NEEDS_SUPPORT_CATEGORIES and it.get('layer') != 'wall':
            sup = find_support(it, items)
            if sup is None:
                if cat in {'vase', 'plant'}:
                    it['layer'] = 'floor'
                    it['x'] = float(room.get('widthM', 4)) / 2
                    it['z'] = float(room.get('lengthM', 5)) / 2
                    it['y'] = .0
                else:
                    it['rejectReason'] = 'needs_support_but_no_valid_supporter'
                    rejected.append(it)
            else:
                it['x'] = sup.get('x', 0)
                it['z'] = sup.get('z', 0)
                it['y'] = float(sup.get('y', 0)) + fp(sup)[2]
                it['supportParentId'] = sup.get('productId')
    # clearance pass based on room type
    for it in items:
        if it.get('layer') != 'floor':
            continue
        w, d = rotated_dims(it)
        x, z = float(it.get('x', 0)), float(it.get('z', 0))
        margin = _clearance_penalty_for_room(room_type, it)
        x = clamp(x, w / 2 + margin, float(room.get('widthM', 4)) - w / 2 - margin)
        z = clamp(z, d / 2 + margin, float(room.get('lengthM', 5)) - d / 2 - margin)
        it['x'], it['z'] = x, z
    items = [x for x in items if not x.get('rejectReason')]
    resolve_collisions(items, room)
    sc, pen = score(items, room)
    return items, rejected, {'fastScore': sc, 'penalties': pen, 'itemCount': len(items), 'rejectedCount': len(rejected), 'collisionCount': pen['collision'], 'outsideCount': pen['outside'], 'roomType': room_type, 'minClearance': min_clearance}

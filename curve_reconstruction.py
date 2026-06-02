import numpy as np
from scipy.spatial import Delaunay


def delaunay_neighbors(pts):
    tri = Delaunay(pts)
    nb = [set() for _ in range(len(pts))]
    for s in tri.simplices:
        for a, b in ((s[0], s[1]), (s[1], s[2]), (s[0], s[2])):
            nb[a].add(b)
            nb[b].add(a)
    return [list(s) for s in nb]


def nn_crust_edges(pts, nb):
    edges = set()
    for s in range(len(pts)):
        cand = nb[s]
        q = min(cand, key=lambda j: np.dot(pts[j] - pts[s], pts[j] - pts[s]))
        edges.add((min(s, q), max(s, q)))
        e0 = pts[q] - pts[s]
        best, bd = None, np.inf
        for j in cand:
            v = pts[j] - pts[s]
            if np.dot(v, e0) <= 0:
                d = np.dot(v, v)
                if d < bd:
                    bd, best = d, j
        if best is not None:
            edges.add((min(s, best), max(s, best)))
    return edges


def edges_to_cycle(edges, N):
    adj = [[] for _ in range(N)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    if any(len(a) != 2 for a in adj):
        return None
    order = [0, adj[0][0]]
    while len(order) < N:
        prev, cur = order[-2], order[-1]
        nxt = adj[cur][0] if adj[cur][0] != prev else adj[cur][1]
        if nxt == order[0]:
            break
        order.append(nxt)
    return np.array(order) if len(order) == N else None


def seg_cross(p1, p2, p3, p4):
    """True if segment p1-p2 and segment p3-p4 properly intersect (not at endpoints)."""
    def cross2d(a, b):
        return a[0] * b[1] - a[1] * b[0]

    d1 = p2 - p1
    d2 = p4 - p3
    denom = cross2d(d1, d2)
    if abs(denom) < 1e-12:
        return False
    t = cross2d(p3 - p1, d2) / denom
    u = cross2d(p3 - p1, d1) / denom
    return 1e-9 < t < 1 - 1e-9 and 1e-9 < u < 1 - 1e-9


def has_crossing(order, pts):
    """Return True if the closed polygon defined by order contains any self-intersection."""
    n = len(order)
    verts = pts[order]
    for i in range(n):
        p1, p2 = verts[i], verts[(i + 1) % n]
        for j in range(i + 2, n):
            if j == n - 1 and i == 0:
                continue  # wrap-around adjacent pair
            p3, p4 = verts[j], verts[(j + 1) % n]
            if seg_cross(p1, p2, p3, p4):
                return True
    return False


def smooth_traversal(pts, nb, start, ang_w=4.0, max_turn_deg=100.0):
    """
    Greedy fallback traversal. At each step choose the unvisited Delaunay neighbor
    that minimises  dist + ang_w * angle * dist, disambiguating near-tangent regions
    (e.g. sharp trailing edges) via directional continuity.
    """
    N = len(pts)
    max_turn = np.deg2rad(max_turn_deg)
    visited = np.zeros(N, dtype=bool)

    order = [start]
    visited[start] = True

    # Bootstrap: pick nearest Delaunay neighbor as the first step
    first = min(nb[start], key=lambda j: np.sum((pts[j] - pts[start]) ** 2))
    order.append(first)
    visited[first] = True

    for _ in range(N - 2):
        prev, cur = order[-2], order[-1]
        direction = pts[cur] - pts[prev]
        d_len = np.linalg.norm(direction)
        if d_len > 0:
            direction = direction / d_len

        best, best_cost = None, np.inf
        for j in nb[cur]:
            if visited[j]:
                continue
            v = pts[j] - pts[cur]
            dist = np.linalg.norm(v)
            if dist < 1e-12:
                continue
            cos_a = np.clip(np.dot(direction, v / dist), -1.0, 1.0)
            angle = np.arccos(cos_a)
            if angle > max_turn:
                continue
            cost = dist * (1.0 + ang_w * angle)
            if cost < best_cost:
                best_cost, best = cost, j

        if best is None:
            # Angle constraint too tight — relax and take the nearest unvisited neighbor
            unvisited = [j for j in nb[cur] if not visited[j]]
            if not unvisited:
                break
            best = min(unvisited, key=lambda j: np.sum((pts[j] - pts[cur]) ** 2))

        order.append(best)
        visited[best] = True

    return np.array(order)


def uncross(order, pts):
    """2-opt: repeatedly reverse sub-sequences until no crossing segments remain."""
    order = list(order)
    N = len(order)
    improved = True
    while improved:
        improved = False
        for i in range(N - 1):
            p1 = pts[order[i]]
            p2 = pts[order[(i + 1) % N]]
            for j in range(i + 2, N):
                if j == N - 1 and i == 0:
                    continue
                p3 = pts[order[j]]
                p4 = pts[order[(j + 1) % N]]
                if seg_cross(p1, p2, p3, p4):
                    order[i + 1: j + 1] = order[i + 1: j + 1][::-1]
                    improved = True
                    break
            if improved:
                break
    return np.array(order)


def make_clockwise(order, pts):
    """Reverse order if shoelace area is positive (counter-clockwise in standard coords)."""
    x, y = pts[order, 0], pts[order, 1]
    signed_area2 = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return order[::-1] if signed_area2 > 0 else order


def order_curve_cw(points, ang_w=4.0, max_turn_deg=100.0):
    """
    Order scattered 2-D points that form a smooth, non-self-intersecting closed curve
    into a clockwise sequence anchored at the rightmost point (argmax x).

    Parameters
    ----------
    points : array-like, shape (N, 2)
    ang_w : float
        Angle-penalty weight used by the smooth_traversal fallback.
    max_turn_deg : float
        Maximum allowed turning angle (degrees) in smooth_traversal.

    Returns
    -------
    order : ndarray of int, shape (N,)
        Indices into `points`, clockwise from argmax(x).
    """
    pts = np.asarray(points, float)
    N = len(pts)
    start = int(np.argmax(pts[:, 0]))   # anchor on original coords

    # Normalise to [0,1]^2 so that aspect-ratio distortion doesn't bias
    # distance-based decisions (critical near sharp trailing edges where the
    # two surfaces are close only in the compressed axis).
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    scale = hi - lo
    scale[scale == 0] = 1.0
    npts = (pts - lo) / scale

    nb = delaunay_neighbors(npts)

    order = edges_to_cycle(nn_crust_edges(npts, nb), N)          # primary path
    if order is None or has_crossing(order, npts):               # validate
        order = smooth_traversal(npts, nb, start, ang_w, max_turn_deg)  # fallback
        order = uncross(order, npts)

    order = make_clockwise(order, pts)                           # orient CW (original coords)
    k = int(np.where(order == start)[0][0])
    return np.roll(order, -k)                                    # anchor to start

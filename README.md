# ReconstructCurve

Reconstruct an ordered, **clockwise** point sequence from unorganized 2-D scatter points that form a closed curve.

The starting point is defined as `argmax(x)` (the rightmost point). The algorithm handles non-convex curves, non-uniform sampling, and regions where the curve runs close to itself (e.g. airfoil trailing edges).

## Problem

Given a point cloud $\{(x_i, y_i)\}$ known to lie on a smooth, simply-connected, non-self-intersecting closed curve (typical case: an airfoil cross-section), recover the along-curve ordering.

The core difficulty is in regions with a small *local feature size* (LFS), such as a sharp trailing edge where upper and lower surfaces are nearly coincident. In those regions, the Euclidean nearest neighbor is not the true curve neighbor, so any purely distance-based method short-circuits across the gap. The solution is to exploit two priors simultaneously: **smoothness** (tangential continuity) to disambiguate, and **no self-intersection** as a hard correctness criterion.

## Pipeline

```
Unordered points
       │
       ▼
delaunay_neighbors     — one Delaunay triangulation; candidate neighbors for
       │                 every point (density-adaptive, prevents long jumps)
       ▼
nn_crust_edges         — primary path: NN-Crust
       │                 edge 1 → nearest neighbor q
       │                 edge 2 → nearest point in the half-plane opposite q
       ▼
edges_to_cycle         — walk the edge set into an ordered cycle
       │                 degree ≠ 2 or disconnected → returns None (primary fails)
       ▼
has_crossing           — segment-intersection check
       │                 self-intersection found → trigger fallback
       ├── pass ────────────────────────────────────────────┐
       │                                                    │
       ▼ fail                                               │
smooth_traversal       — greedy traversal weighted by       │
       │                 turning-angle cost                 │
uncross                — 2-opt to remove residual crossings |
       │                                                    │
       └────────────────────────────────────────────────────┘
       ▼
make_clockwise         — shoelace orientation (clockwise)
       ▼
rotate_to_start        — roll sequence to anchor at argmax(x)
       ▼
Clockwise index array
```

Coordinates are normalized to $[0, 1]^2$ internally before any distance computation, eliminating aspect-ratio bias that would otherwise cause cross-gap errors at sharp trailing edges. The returned values are indices into the original point array.

## API

| Function | Description |
|---|---|
| `delaunay_neighbors(pts)` | Delaunay triangulation; returns per-point neighbor lists |
| `nn_crust_edges(pts, nb)` | NN-Crust primary path; returns an undirected edge set |
| `edges_to_cycle(edges, N)` | Walk edge set into an ordered cycle; returns `None` on failure |
| `seg_cross(p1, p2, p3, p4)` | Proper segment–segment intersection test |
| `has_crossing(order, pts)` | Check whether a cycle contains any self-intersection |
| `smooth_traversal(pts, nb, start, ang_w, max_turn_deg)` | Angle-cost greedy traversal (fallback path) |
| `uncross(order, pts)` | 2-opt reversal to eliminate self-intersections |
| `make_clockwise(order, pts)` | Shoelace-based clockwise orientation |
| `order_curve_cw(points, ang_w, max_turn_deg)` | **Main entry point** — returns a clockwise index array |

## Usage

```python
import numpy as np
from curve_reconstruction import order_curve_cw

pts = np.loadtxt('my_points.csv', delimiter=',')  # shape (N, 2)
order = order_curve_cw(pts)                       # ndarray of int, shape (N,)
ordered_pts = pts[order]                          # coordinates in clockwise order
```

`order_curve_cw` parameters:

| Parameter | Default | Description |
|---|---|---|
| `points` | — | Input scatter points, shape `(N, 2)` |
| `ang_w` | `4.0` | Turning-angle penalty weight in the fallback traversal |
| `max_turn_deg` | `100.0` | Maximum allowed turning angle (degrees) in the fallback traversal |

> **Coordinate system**: `make_clockwise` uses standard mathematical orientation (y up). For image coordinates (y down), negate the shoelace sign condition.

## Running the tests

```bash
python test_reconstruction.py
```

Reads all CSV files from `data/` (two columns, one header row: x/c and z coordinate), reconstructs each curve, and saves a side-by-side scatter / reconstructed-curve figure to `output/`.

## Dependencies

- Python 3.x
- NumPy
- SciPy

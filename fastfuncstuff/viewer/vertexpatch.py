"""A patch of cortex around one vertex, flattened for drawing time courses in place.

The vertex graph shows each vertex's own data where that vertex sits, so the
reason one vertex and not its neighbour lit up can be read off the picture. A
mesh has no grid to borrow, so the patch is laid out on the page:

* **which vertices**: ``rings`` mesh edges out from the centre, so a step adds
  the next ring whatever the mesh's density;
* **how far out**: the shortest path along midthickness edges, i.e. mm along
  the cortex -- a gyral crown and the sulcal wall beside it stay apart even
  where the folded surface brings them close in space;
* **which way**: the direction in the shown surface's tangent plane at the
  centre, turned so up is the camera's up. The page then reads like the 3-D
  view in front of you.

That is the exponential map's polar form, with path length for geodesic
distance. Edge paths overestimate a straight geodesic by up to ~10 %, the same
for every vertex, which a picture that is about who neighbours whom can carry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["VertexPatch", "hex_layout", "vertex_patch"]


@dataclass
class VertexPatch:
    hemi: str
    centre: int
    #: Mesh vertex ids, centre first.
    vertices: np.ndarray
    #: Page positions, mm along the cortex, +y up. Row ``i`` is ``vertices[i]``.
    xy: np.ndarray
    #: Mesh edges inside the patch, as row pairs into ``vertices``.
    edges: np.ndarray
    #: The patch's outline: mesh edges on exactly one patch face, vertex id pairs.
    outline: np.ndarray
    #: Mesh rings out from the centre per row (0 for the centre).
    ring: np.ndarray


def _rings(faces: np.ndarray, centre: int, rings: int, n: int) -> np.ndarray:
    """Ring number per vertex (-1 beyond ``rings``), by breadth-first steps."""
    import scipy.sparse as sp

    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    adj = sp.csr_matrix((np.ones(len(e), bool), (e[:, 0], e[:, 1])), shape=(n, n))
    adj = adj + adj.T
    ring = np.full(n, -1, np.int64)
    ring[centre] = 0
    front = np.zeros(n, bool)
    front[centre] = True
    for k in range(1, rings + 1):
        nxt = (adj @ front) & (ring < 0)
        if not nxt.any():
            break
        ring[nxt] = k
        front = nxt
    return ring


def vertex_patch(
    hemi: str,
    faces: np.ndarray,
    mid: np.ndarray,
    shown: np.ndarray,
    centre: int,
    rings: int,
    up: np.ndarray,
    toward: np.ndarray | None = None,
) -> VertexPatch:
    """The flattened patch ``rings`` edges around ``centre``.

    ``mid`` gives distances (the midthickness, or whatever lies between white
    and pial); ``shown`` gives directions (the surface as drawn, in the same
    frame as the camera's ``up``). ``toward`` points at the camera: the
    normal is turned to face it, or a mesh wound the other way (or seen from
    inside) would come out mirrored.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import dijkstra

    faces = np.asarray(faces, np.int64)
    n = len(mid)
    ring_all = _rings(faces, centre, rings, n)
    inside = ring_all >= 0
    ids = np.flatnonzero(inside)
    ids = np.r_[centre, ids[ids != centre]]
    row = np.full(n, -1, np.int64)
    row[ids] = np.arange(len(ids))

    patch_faces = faces[inside[faces].all(1)]
    e = np.concatenate([patch_faces[:, [0, 1]], patch_faces[:, [1, 2]], patch_faces[:, [2, 0]]])
    key = np.sort(e, axis=1)
    uniq, count = np.unique(key, axis=0, return_counts=True)
    outline = uniq[count == 1]
    edges = row[uniq]

    # Path length along midthickness edges, from the centre.
    a, b = edges[:, 0], edges[:, 1]
    length = np.linalg.norm(mid[ids[a]] - mid[ids[b]], axis=1)
    graph = sp.csr_matrix((length, (a, b)), shape=(len(ids), len(ids)))
    dist = dijkstra(graph, directed=False, indices=0)
    dist = np.where(np.isfinite(dist), dist, 0.0)

    # Directions in the shown surface's tangent plane at the centre.
    centre_faces = faces[(faces == centre).any(1)]
    p = np.asarray(shown, np.float64)
    fn = np.cross(
        p[centre_faces[:, 1]] - p[centre_faces[:, 0]], p[centre_faces[:, 2]] - p[centre_faces[:, 0]]
    )
    normal = fn.sum(0)
    normal /= np.linalg.norm(normal) or 1.0
    if toward is not None and float(np.dot(normal, toward)) < 0:
        normal = -normal
    y = np.asarray(up, np.float64) - normal * float(np.dot(up, normal))
    if np.linalg.norm(y) < 1e-6:  # looking straight down the normal's side
        y = np.cross(normal, [1.0, 0.0, 0.0])
        if np.linalg.norm(y) < 1e-6:
            y = np.cross(normal, [0.0, 1.0, 0.0])
    y /= np.linalg.norm(y)
    x = np.cross(y, normal)
    d = p[ids] - p[centre]
    angle = np.arctan2(d @ y, d @ x)
    xy = np.c_[dist * np.cos(angle), dist * np.sin(angle)]
    return VertexPatch(hemi, int(centre), ids, xy, edges, outline, ring_all[ids])


def hex_layout(xy: np.ndarray) -> np.ndarray:
    """``xy`` moved onto a hexagonal lattice, one vertex per site, none dropped.

    The lattice spacing is the patch's typical neighbour distance. Vertices are
    placed from the centre outward, each on the free site nearest its own
    position, so the middle stays exact and the rings keep their order; where a
    dense stretch wants more sites than the lattice has there, the overflow
    lands on the next-nearest free one rather than on top of a neighbour.
    """
    from scipy.spatial import KDTree

    xy = np.asarray(xy, np.float64)
    if len(xy) < 2:
        return xy.copy()
    d, _ = KDTree(xy).query(xy, k=2)
    s = float(np.median(d[:, 1])) or 1.0
    reach = float(np.abs(xy).max()) + 2 * s * np.sqrt(len(xy))
    n = int(np.ceil(reach / s)) + 1
    i, j = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1), indexing="ij")
    sites = np.c_[(i + 0.5 * j).ravel() * s, (j * np.sqrt(3) / 2).ravel() * s]
    sites = sites[np.linalg.norm(sites, axis=1) <= reach]
    tree = KDTree(sites)
    taken = np.zeros(len(sites), bool)
    out = np.empty_like(xy)
    for row in np.argsort(np.linalg.norm(xy, axis=1), kind="stable"):
        k = 8
        while True:
            _, near = tree.query(xy[row], k=min(k, len(sites)))
            free = [int(c) for c in np.atleast_1d(near) if not taken[c]]
            if free:
                taken[free[0]] = True
                out[row] = sites[free[0]]
                break
            k *= 4
    return out

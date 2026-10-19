"""
Fast cluster-extent null builder for [[ffs_perm]] — Numba-JIT'd
sorted-voxel union-find.

Idea
----
For one permutation, one NN connectivity, and one sidedness, naïvely we'd
threshold the t-map at every ``pthr`` and run a separate CCL pass per
threshold (the cc3d path).  Instead:

1. Apply sidedness (raw t for 1sided; ``|t|`` for 2sided).
2. Keep only voxels above the lowest pthr's tcrit.
3. Sort those voxels by stat **descending**.
4. Walk them: each voxel joins a disjoint-set forest, unioning with any
   already-active neighbours under the chosen NN connectivity.
5. Whenever we're about to add a voxel whose stat falls below the next
   tcrit, snapshot the current ``max(cluster size)`` for that pthr.

That processes all 9 default pthrs in **one pass per (NN, sided)**.  Mass
is intentionally skipped in v1 but the kernel layout (parent / size
arrays per root) makes adding a ``mass[root]`` accumulator a small
follow-up.

Performance budget (≈381 k mask voxels):
* sort: ~5 ms
* DSU walk with neighbour lookup: ~10–15 ms in Numba
* per (NN, sided) call: ~20 ms
* per perm × 6 combos: ~120 ms
* 1000 perms / 8 workers: ≈ 15 s
* 10 000 perms / 8 workers: ≈ 150 s
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange, types
from numba.typed import List as TypedList  # noqa: F401  (kept for future mass extension)

# ---------------------------------------------------------------------------
# Neighbour offset tables
# ---------------------------------------------------------------------------


def _offsets_for_nn(nn: int) -> np.ndarray:
    """``[K, 3]`` int8 array of (dx, dy, dz) offsets, K = 6/18/26."""
    if nn == 1:
        # 6 face neighbours
        return np.array(
            [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)],
            dtype=np.int8,
        )
    if nn == 2:
        # 18 = faces + edges
        ofs = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    if abs(dx) + abs(dy) + abs(dz) <= 2:
                        ofs.append((dx, dy, dz))
        return np.asarray(ofs, dtype=np.int8)
    if nn == 3:
        # 26 = full Moore neighbourhood
        ofs = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    ofs.append((dx, dy, dz))
        return np.asarray(ofs, dtype=np.int8)
    raise ValueError(f"NN must be 1, 2, or 3; got {nn}")


# ---------------------------------------------------------------------------
# Numba kernel
# ---------------------------------------------------------------------------


@njit(cache=True, boundscheck=False, fastmath=True)
def _walk_dsu_extent(
    sorted_voxel_idx: np.ndarray,  # int64[n], flat indices, stat-descending
    sorted_voxel_stat: np.ndarray,  # float32[n], effective stat (signed/abs), descending
    parent: np.ndarray,  # int64[V_total], scratch, init = -1
    size: np.ndarray,  # int64[V_total], scratch
    offsets: np.ndarray,  # int8[K, 3] neighbour offsets
    nx: int,
    ny: int,
    nz: int,
    tcrits_desc: np.ndarray,  # float64[npthr], sorted *descending*
) -> np.ndarray:
    """Single-pass DSU; returns ``max_extent[npthr]`` int64."""
    n = sorted_voxel_idx.shape[0]
    npthr = tcrits_desc.shape[0]
    n_off = offsets.shape[0]
    # numpy C-order: for array of shape (nx, ny, nz), flat index
    # v = i*(ny*nz) + j*nz + k, with k fastest.
    yz = ny * nz

    max_extent = np.zeros(npthr, dtype=np.int64)
    cur_max = np.int64(0)
    ip = 0  # next pthr index to snapshot

    for kk in range(n):
        s = sorted_voxel_stat[kk]
        # Before adding this voxel, snapshot for every pthr whose tcrit
        # is now above s (i.e. this voxel is below their threshold).
        while ip < npthr and s <= tcrits_desc[ip]:
            max_extent[ip] = cur_max
            ip += 1
        if ip >= npthr:
            break  # all snapshots collected; remaining voxels irrelevant

        v = sorted_voxel_idx[kk]
        # Decompose v → (i, j, k) in C order; k fastest.
        i = v // yz
        rem = v - i * yz
        j = rem // nz
        k = rem - j * nz

        parent[v] = v
        size[v] = 1
        if cur_max < 1:
            cur_max = 1

        # Iterate neighbours
        for o in range(n_off):
            di = offsets[o, 0]
            dj = offsets[o, 1]
            dk = offsets[o, 2]
            i2 = i + di
            j2 = j + dj
            k2 = k + dk
            if i2 < 0 or i2 >= nx or j2 < 0 or j2 >= ny or k2 < 0 or k2 >= nz:
                continue
            nbr = i2 * yz + j2 * nz + k2
            if parent[nbr] < 0:
                continue
            # find(v) — v is its own root right now, but it may have been
            # unioned by an earlier neighbour in this same loop iteration.
            ra = v
            while parent[ra] != ra:
                parent[ra] = parent[parent[ra]]
                ra = parent[ra]
            # find(nbr)
            rb = nbr
            while parent[rb] != rb:
                parent[rb] = parent[parent[rb]]
                rb = parent[rb]
            if ra == rb:
                continue
            # union by size
            if size[ra] < size[rb]:
                parent[ra] = rb
                size[rb] += size[ra]
                if size[rb] > cur_max:
                    cur_max = size[rb]
            else:
                parent[rb] = ra
                size[ra] += size[rb]
                if size[ra] > cur_max:
                    cur_max = size[ra]

    # Tail: any remaining pthrs adopt the final cur_max
    while ip < npthr:
        max_extent[ip] = cur_max
        ip += 1
    return max_extent


# ---------------------------------------------------------------------------
# Top-level helper: one perm, all (NN, sided)
# ---------------------------------------------------------------------------


def cluster_extent_one_perm(
    stat3d: np.ndarray,
    mask_flat_idx: np.ndarray,
    shape_xyz: tuple[int, int, int],
    nns: tuple[int, ...],
    sideds: tuple[str, ...],
    tcrits_by_sided: dict[str, np.ndarray],
    offsets_by_nn: dict[int, np.ndarray],
    parent_scratch: np.ndarray,
    size_scratch: np.ndarray,
) -> dict[tuple[str, int], np.ndarray]:
    """One perm; returns ``{(sided, nn): max_extent_per_pthr[npthr]}``.

    Parameters
    ----------
    stat3d : float32 3-D array (already with mask scattered in).
    mask_flat_idx : int64[V_mask] — flat indices of in-mask voxels.
    tcrits_by_sided : dict mapping ``"1-sided"``/``"2-sided"`` to
        ``tcrits_desc`` (descending).
    offsets_by_nn : dict mapping NN → offset table.
    parent_scratch, size_scratch : preallocated int64[V_total] buffers,
        reset to -1 / 0 inside this function.
    """
    nx, ny, nz = shape_xyz
    stat_flat = stat3d.reshape(-1)
    stat_at_mask = stat_flat[mask_flat_idx]
    out: dict[tuple[str, int], np.ndarray] = {}

    for sided in sideds:
        if sided == "1-sided":
            eff_passes = [stat_at_mask]
        elif sided == "2-sided":
            eff_passes = [np.abs(stat_at_mask)]
        elif sided == "bi-sided":
            # Two independent passes: positive tail and negative tail.
            # Tail = max over the two; mass would be the sum within each
            # tail's connected components.
            eff_passes = [stat_at_mask, -stat_at_mask]
        else:
            raise ValueError(f"unknown sidedness: {sided}")

        tcrits_desc = tcrits_by_sided[sided]
        lowest = tcrits_desc[-1]

        # Threshold and sort once per tail, not once per (tail, NN): the three
        # connectivities differ only in which neighbours they union, and they
        # walk the exact same suprathreshold voxels in the exact same order.
        prepared = []
        for eff in eff_passes:
            above = eff > lowest
            stat_above = eff[above]
            order = np.argsort(-stat_above, kind="stable")
            prepared.append(
                (
                    np.ascontiguousarray(mask_flat_idx[above][order]),
                    np.ascontiguousarray(stat_above[order].astype(np.float32)),
                )
            )

        for nn in nns:
            per_pass_extents = []
            for idx_sorted, stat_sorted in prepared:
                per_pass_extents.append(
                    _walk_dsu_extent(
                        idx_sorted,
                        stat_sorted,
                        parent_scratch,
                        size_scratch,
                        offsets_by_nn[nn],
                        nx,
                        ny,
                        nz,
                        tcrits_desc,
                    )
                )
                # Clear only what the walk could have touched.  A full
                # parent.fill(-1) is a whole-volume memset, and at a p=0.05
                # null under 5% of the volume is ever suprathreshold -- it
                # was costing more than some of the DSU walks it reset.
                parent_scratch[idx_sorted] = -1
            # Bi-sided: elementwise max over the +/- passes.  1-/2-sided
            # have a single pass so this is a no-op.
            stacked = np.stack(per_pass_extents, axis=0)
            out[(sided, nn)] = stacked.max(axis=0)

    return out


# ---------------------------------------------------------------------------
# Fused batch kernel: every perm, every (NN, sidedness), threaded
# ---------------------------------------------------------------------------
#
# The per-perm path above runs 4 tail walks x 3 NN = 12 sorted DSU walks per perm
# from Python, in a process pool. This kernel computes the same tables with less:
#   * no sort. A snapshot at threshold c is the max cluster over voxels with stat
#     > c, which does not depend on the order they were added, so voxels are
#     bucketed by threshold bin in O(n).
#   * one walk per tail for all three connectivities: each neighbour is looked up
#     once and unioned into the NN1 / NN2 / NN3 forests it belongs to (NN1 offsets
#     are a subset of NN2's, which are a subset of NN3's).
#   * the 1-sided walk and the positive tail of bi-sided / 2-sided sort the same
#     values (raw t): one walk over the merged thresholds snapshots them all.
#   * 2-sided (|t|) clusters equal the larger of the two tails' at every
#     threshold unless a positive and a negative suprathreshold voxel touch; that
#     is checked per perm, and only then is the |t| walk run.
#   * a padded lookup grid turns each neighbour into one indexed load (no bounds
#     tests), and the forests are indexed by in-mask voxel in int32, so they stay
#     in cache.
# Threads over perms replace the process pool (no pickling of the t matrix).

_FOREST_NN = (1, 2, 3)


@njit(cache=True, boundscheck=False, inline="always")
def _union(parent, size, a, b, cur):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    while parent[b] != b:
        parent[b] = parent[parent[b]]
        b = parent[b]
    if a == b:
        return cur
    if size[a] < size[b]:
        a, b = b, a
    parent[b] = a
    size[a] += size[b]
    if size[a] > cur:
        return size[a]
    return cur


@njit(cache=True, boundscheck=False)
def _walk_fused(row, mode, tcrits_desc, lutp, pflat, deltas, w0, w1, w2,
                p0, s0, p1, s1, p2, s2, active, out):  # fmt: skip
    """Max cluster extent per threshold for one stat map.

    ``row`` float32[Vm], read as ``row`` (``mode`` 0), ``-row`` (1) or
    ``|row|`` (2) -- in place of a negated or absolute copy per perm; ``tcrits_desc`` descending; ``lutp`` padded-grid
    flat -> in-mask index (-1 outside the mask or on the pad); ``pflat`` each
    in-mask voxel's padded flat index; ``deltas`` the 26 neighbour offsets in
    padded flat units, NN1 first (6), then the NN2 edges (12), then corners (8).
    ``out`` int64[3, npthr] (rows NN1/NN2/NN3).
    """
    npthr = tcrits_desc.shape[0]
    lowest = tcrits_desc[npthr - 1]
    vm = row.shape[0]
    counts = np.zeros(npthr + 1, dtype=np.int64)
    binof = np.empty(vm, dtype=np.int16)
    for m in range(vm):
        s = row[m]
        if mode == 1:
            s = -s
        elif mode == 2:
            s = abs(s)
        if s > lowest:
            lo = 0
            hi = npthr - 1
            while lo < hi:
                mid = (lo + hi) >> 1
                if s > tcrits_desc[mid]:
                    hi = mid
                else:
                    lo = mid + 1
            binof[m] = lo
            counts[lo + 1] += 1
        else:
            binof[m] = -1
    for b in range(npthr):
        counts[b + 1] += counts[b]
    n_above = counts[npthr]
    idx = np.empty(n_above, dtype=np.int32)
    fill = counts[:npthr].copy()
    for m in range(vm):
        b = binof[m]
        if b >= 0:
            idx[fill[b]] = m
            fill[b] += 1

    c0 = np.int64(0)
    c1 = np.int64(0)
    c2 = np.int64(0)
    for b in range(npthr):
        for kk in range(counts[b], counts[b + 1]):
            m = idx[kk]
            active[m] = 1
            p0[m] = m
            s0[m] = 1
            p1[m] = m
            s1[m] = 1
            p2[m] = m
            s2[m] = 1
            if c0 < 1:
                c0 = 1
                c1 = 1
                c2 = 1
            base = pflat[m]
            for o in range(6):
                m2 = lutp[base + deltas[o]]
                if m2 < 0 or active[m2] == 0:
                    continue
                if w0:
                    c0 = _union(p0, s0, m, m2, c0)
                if w1:
                    c1 = _union(p1, s1, m, m2, c1)
                if w2:
                    c2 = _union(p2, s2, m, m2, c2)
            if w1 or w2:
                for o in range(6, 18):
                    m2 = lutp[base + deltas[o]]
                    if m2 < 0 or active[m2] == 0:
                        continue
                    if w1:
                        c1 = _union(p1, s1, m, m2, c1)
                    if w2:
                        c2 = _union(p2, s2, m, m2, c2)
            if w2:
                for o in range(18, 26):
                    m2 = lutp[base + deltas[o]]
                    if m2 < 0 or active[m2] == 0:
                        continue
                    c2 = _union(p2, s2, m, m2, c2)
        out[0, b] = c0
        out[1, b] = c1
        out[2, b] = c2
    for kk in range(n_above):
        active[idx[kk]] = 0


@njit(cache=True, boundscheck=False)
def _tails_touch(row, c, lutp, pflat, deltas):
    """Does any voxel with t < -c have an NN3 neighbour with t > c?"""
    for m in range(row.shape[0]):
        if row[m] < -c:
            base = pflat[m]
            for o in range(26):
                m2 = lutp[base + deltas[o]]
                if m2 >= 0 and row[m2] > c:
                    return True
    return False


@njit(parallel=True, cache=True, boundscheck=False)
def _null_extents_batch(t_pv, lutp, pflat, deltas, w0, w1, w2,
                        tc_plus, tc_two, do_minus, do_abs, n_blocks):  # fmt: skip
    n_perm, vm = t_pv.shape
    n2 = tc_two.shape[0]
    ext_plus = np.zeros((n_perm, 3, tc_plus.shape[0]), dtype=np.int64)
    ext_minus = np.zeros((n_perm, 3, n2), dtype=np.int64)
    # -1 marks "tails do not touch: take the max of the two tails".
    ext_abs = np.full((n_perm, 3, n2), -1, dtype=np.int64)
    lowest2 = tc_two[n2 - 1] if n2 > 0 else 0.0
    for blk in prange(n_blocks):  # ty: ignore[not-iterable]
        q0 = blk * n_perm // n_blocks
        q1 = (blk + 1) * n_perm // n_blocks
        p0 = np.empty(vm, dtype=np.int32)
        s0 = np.empty(vm, dtype=np.int32)
        p1 = np.empty(vm, dtype=np.int32)
        s1 = np.empty(vm, dtype=np.int32)
        p2 = np.empty(vm, dtype=np.int32)
        s2 = np.empty(vm, dtype=np.int32)
        active = np.zeros(vm, dtype=np.uint8)
        for q in range(q0, q1):
            row = t_pv[q]
            if tc_plus.shape[0] > 0:
                _walk_fused(row, 0, tc_plus, lutp, pflat, deltas, w0, w1, w2,
                            p0, s0, p1, s1, p2, s2, active, ext_plus[q])  # fmt: skip
            if do_minus:
                _walk_fused(row, 1, tc_two, lutp, pflat, deltas, w0, w1, w2,
                            p0, s0, p1, s1, p2, s2, active, ext_minus[q])  # fmt: skip
            if do_abs and _tails_touch(row, lowest2, lutp, pflat, deltas):
                _walk_fused(row, 2, tc_two, lutp, pflat, deltas, w0, w1, w2,
                            p0, s0, p1, s1, p2, s2, active, ext_abs[q])  # fmt: skip
    return ext_plus, ext_minus, ext_abs


def cluster_extent_null_batch(
    t_pv: np.ndarray,
    mask: np.ndarray,
    nns: tuple[int, ...],
    sideds: tuple[str, ...],
    tcrits_by_sided: dict[str, np.ndarray],
    n_threads: int | None = None,
) -> dict[tuple[str, int], np.ndarray]:
    """Max cluster extent for every perm, ``{(sided, nn): int64[P, npthr]}``.

    ``tcrits_by_sided`` maps each sidedness to its critical values in the
    caller's pthr order (the result columns follow it); 2-sided and bi-sided
    must share them. The same numbers as :func:`cluster_extent_one_perm`, from
    one threaded Numba pass over all perms.
    """
    import numba

    nx, ny, nz = mask.shape
    flat = np.flatnonzero(mask.ravel())
    i, j, k = np.unravel_index(flat, mask.shape)
    py, pz = ny + 2, nz + 2
    pflat = ((i + 1) * py + (j + 1)) * pz + (k + 1)
    lutp = np.full((nx + 2) * py * pz, -1, dtype=np.int32)
    lutp[pflat] = np.arange(flat.size, dtype=np.int32)
    offsets = _offsets_for_nn(3).astype(np.int64)
    level = np.abs(offsets).sum(axis=1)
    offsets = offsets[np.argsort(level, kind="stable")]
    deltas = np.ascontiguousarray((offsets[:, 0] * py + offsets[:, 1]) * pz + offsets[:, 2])
    w0, w1, w2 = (nn in nns for nn in _FOREST_NN)

    want_two = "2-sided" in sideds or "bi-sided" in sideds
    if (
        "2-sided" in sideds
        and "bi-sided" in sideds
        and not np.array_equal(
            np.asarray(tcrits_by_sided["2-sided"]), np.asarray(tcrits_by_sided["bi-sided"])
        )
    ):
        raise ValueError("2-sided and bi-sided must share critical values")
    two_vals = np.asarray(
        tcrits_by_sided["2-sided" if "2-sided" in sideds else "bi-sided"] if want_two else [],
        dtype=np.float64,
    )
    two_order = np.argsort(-two_vals, kind="stable")
    tc_two = np.ascontiguousarray(two_vals[two_order])
    # The raw-t walk carries the 1-sided thresholds and the positive tail of the
    # two-tailed ones; (source, original column) per merged column.
    plus_src = []
    if "1-sided" in sideds:
        plus_src += [(1, c, float(v)) for c, v in enumerate(tcrits_by_sided["1-sided"])]
    if want_two:
        plus_src += [(2, c, float(v)) for c, v in enumerate(two_vals)]
    plus_src.sort(key=lambda r: -r[2])
    tc_plus = np.array([r[2] for r in plus_src], dtype=np.float64)

    # get_num_threads() is the pool size (NUMBA_NUM_THREADS) unless something set
    # it lower, and set_num_threads refuses more than the pool.
    prev = numba.get_num_threads()
    n_threads = max(1, min(int(n_threads or prev), prev))
    n_perm = t_pv.shape[0]
    numba.set_num_threads(n_threads)
    try:
        ext_plus, ext_minus, ext_abs = _null_extents_batch(
            np.ascontiguousarray(t_pv, dtype=np.float32), lutp, pflat, deltas,
            w0, w1, w2, tc_plus, tc_two, want_two, "2-sided" in sideds,
            max(1, min(n_perm, 4 * n_threads)),
        )  # fmt: skip
    finally:
        numba.set_num_threads(prev)

    out: dict[tuple[str, int], np.ndarray] = {}
    for nn in nns:
        f = _FOREST_NN.index(nn)
        pos1 = np.zeros((n_perm, len(plus_src)), dtype=np.int64)
        pos2 = np.zeros((n_perm, two_vals.size), dtype=np.int64)
        for col, (src, c, _v) in enumerate(plus_src):
            (pos1 if src == 1 else pos2)[:, c] = ext_plus[:, f, col]
        neg2 = np.zeros_like(pos2)
        neg2[:, two_order] = ext_minus[:, f, :]
        tails = np.maximum(pos2, neg2)
        for sided in sideds:
            if sided == "1-sided":
                out[(sided, nn)] = pos1[:, : len(tcrits_by_sided["1-sided"])].copy()
            elif sided == "bi-sided":
                out[(sided, nn)] = tails
            else:
                ab = np.zeros_like(pos2)
                ab[:, two_order] = ext_abs[:, f, :]
                out[(sided, nn)] = np.where(ab >= 0, ab, tails)
    return out


# ---------------------------------------------------------------------------
# Numba warm-up (compile cache hit on import)
# ---------------------------------------------------------------------------


def precompile() -> None:
    """Trigger Numba JIT once so the first real call isn't slow."""
    parent = np.full(8, -1, dtype=np.int64)
    size = np.zeros(8, dtype=np.int64)
    offsets = _offsets_for_nn(1)
    sorted_idx = np.array([0], dtype=np.int64)
    sorted_stat = np.array([5.0], dtype=np.float32)
    tcrits = np.array([4.0, 3.0], dtype=np.float64)
    _walk_dsu_extent(sorted_idx, sorted_stat, parent, size, offsets, 2, 2, 2, tcrits)


# Avoid `types` lint warning by referencing once (Numba may use it later).
_ = types

"""The flags that name a cortical ribbon, and how ROI depth sampling cuts it.

Kept free of torch and the mesh code so a tool's parser (and ``-help``) stays fast;
:func:`surface.volume_depth.resolve_ribbon_surfaces` reads the ribbon flags and
:mod:`processing.roi_depth` the depth ones.
"""

from __future__ import annotations


def add_ribbon_args(group, *, hemi: bool = True) -> None:
    """Register ``-fs_subj -white -pial -white_mesh -pial_mesh -cortex_label`` (+ ``-hemi``)."""
    group.add_argument(
        "-fs_subj",
        default=None,
        help="FreeSurfer subject directory (or give -white_mesh and -pial_mesh)",
    )
    if hemi:
        group.add_argument("-hemi", default="both", choices=("both", "lh", "rh"))
    group.add_argument("-white", default="white", help="White surface name in surf/ (white)")
    group.add_argument("-pial", default="pial", help="Pial surface name in surf/ (pial)")
    group.add_argument(
        "-white_mesh",
        nargs="+",
        default=None,
        metavar="FILE",
        help="White surface file(s), one per hemisphere, instead of surf/?h.<-white>. "
        "Hemisphere read from the name (lh./rh.); else the order is lh, rh. One path "
        "without lh/rh that does not exist (surf/smoothwm) means lh./rh. of it",
    )
    group.add_argument(
        "-pial_mesh",
        nargs="+",
        default=None,
        metavar="FILE",
        help="Pial surface file(s) paired with -white_mesh by hemisphere",
    )
    group.add_argument(
        "-cortex_label",
        nargs="+",
        default=None,
        metavar="FILE",
        help="?h.cortex.label file(s) for -white_mesh/-pial_mesh (default with "
        "-fs_subj: its label/?h.cortex.label; otherwise the medial wall is kept)",
    )


def add_depth_args(group) -> None:
    """Ribbon flags plus ``-depths -equidistant/-equivolume -depth_dxyz`` for -atlas/-rois.

    Giving surfaces is what switches depth sampling on: the ROI means are then read
    along the cortical ribbon, one per ROI per depth, instead of over ROI voxels.
    """
    add_ribbon_args(group, hemi=False)
    group.add_argument(
        "-depths",
        type=int,
        default=1,
        metavar="N",
        help="With surfaces: N cortical depth bins per ROI, 1 = deepest (next to white "
        "matter) .. N = by pial.  1 averages the whole ribbon inside each ROI",
    )
    how = group.add_mutually_exclusive_group()
    how.add_argument(
        "-equidistant",
        dest="depth_method",
        action="store_const",
        const="equidistant",
        help="Depth bins of equal thickness (default)",
    )
    how.add_argument(
        "-equivolume",
        dest="depth_method",
        action="store_const",
        const="equivolume",
        help="Depth bins of equal volume within each cortical column (Bok): thinner "
        "where the cortex bends away from them",
    )
    group.set_defaults(depth_method="equidistant")
    group.add_argument(
        "-depth_dxyz",
        type=float,
        default=None,
        metavar="MM",
        help="Ribbon sampling grid, mm: the data grid upsampled to this, each ribbon "
        "point read from the data by trilinear interpolation.  Default: a third of "
        "the data's finest voxel, at most 0.5 mm",
    )


def depth_requested(args) -> bool:
    return bool(getattr(args, "fs_subj", None) or getattr(args, "white_mesh", None))

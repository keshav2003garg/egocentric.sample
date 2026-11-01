"""Both hands in 3D (surface and skeleton) in the room, and their keypoints over the video."""
import numpy as np
import pandas as pd
import rerun as rr
import rerun.blueprint as rrb
import cv2
from scipy import sparse
from scipy.ndimage import binary_dilation
from scipy.spatial.transform import Rotation

from ego_rerun.clip import FPS, HAND_COLOURS, HIDDEN, Clip, note, plot, times

# Standard 21-keypoint connections: thumb and four fingers from the wrist, plus the knuckle line
KP21_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),
              (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]
SURFACE_VOXEL_M = 0.006
TIPS = [4, 9, 14, 19, 24]  # fingertips in the 25 joints: the frame where they're furthest apart sets the web surface
ON_SCENE_MIN = 0.4  # a 3D hand with less of its surface than this on the hand or objects in the picture isn't drawn
ON_HAND_MIN = 0.35  # ...nor one clearly in view (outline >= BIG_HAND_PX) with less than this on its own outline
BIG_HAND_PX = 3000  # half-res px
OFF_SCENE_PAD = 2  # ...nor the frames this close to it
SMOOTH_HALF = 4  # hands.parquet is smoothed over 9 frames, so a bad frame pulls up to 4 neighbours off with it

FILES = ("hands.parquet", "hands_meta.json")


def bone_of_vertices(verts: np.ndarray, joints: np.ndarray, parent: list[int]) -> np.ndarray:
    """Each vertex's nearest bone (index of its child joint), on one frame."""
    best, bone = np.full(len(verts), np.inf), np.zeros(len(verts), int)
    for j in range(1, len(parent)):
        a, b = joints[parent[j]], joints[j]
        u = np.clip(((verts - a) @ (b - a)) / max((b - a) @ (b - a), 1e-12), 0, 1)
        d = np.linalg.norm(verts - (a + u[:, None] * (b - a)), axis=1)
        closer = d < best
        best[closer], bone[closer] = d[closer], j
    return bone


def simplify(rest: np.ndarray, bone: np.ndarray, faces: np.ndarray, voxel: float) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Fewer vertices with the same topology on every frame. Vertices are grouped by voxel on a frame with the fingers
    apart, and only with vertices on the same bone, so touching fingers never merge; each group is the mean of its
    vertices on every frame (an averaging matrix), and collapsed or repeated triangles are dropped."""
    key = np.c_[np.floor(rest / voxel).astype(np.int64), bone]
    _, group = np.unique(key, axis=0, return_inverse=True)
    group = group.reshape(-1)
    counts = np.bincount(group)
    mean = sparse.csr_matrix((1.0 / counts[group], (group, np.arange(len(group)))), shape=(len(counts), len(group)))
    f = group[faces]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    _, unique_rows = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
    return mean, f[np.sort(unique_rows)].astype(np.uint32)


def on_scene(clip: Clip, verts_world: np.ndarray, hand: str) -> tuple[np.ndarray, np.ndarray]:
    """Per frame, the share of one hand's surface (every 4th vertex) that lands (a) on the hand or any tracked object in
    the picture and (b) on that hand's own outline, where the hand is clearly in view (masks grown ~4 px); 1 where it
    can't be told. Low (a) means the 3D hand is drawn off the real one; low (b) on a big, visible hand means it trails."""
    lens = clip.json("camera_path_report.json").get("lens_used")
    if lens is None or not clip.has("masks.npz"):
        return np.ones(len(verts_world)), np.ones(len(verts_world))
    K = np.array([[lens["fx"], 0, lens["cx"]], [0, lens["fy"], lens["cy"]], [0, 0, 1.0]])
    D = np.array(lens["k"]).reshape(4, 1)
    path = pd.read_parquet(clip.dir / "camera_path.parquet")
    R = Rotation.from_quat(path[["qx", "qy", "qz", "qw"]].to_numpy()).as_matrix()
    P = path[["x", "y", "z"]].to_numpy()
    masks = np.load(clip.dir / "masks.npz")
    labels, own_id = masks["labels"], list(masks["names"]).index(f"{hand}_hand") + 1
    share, own = np.ones(len(verts_world)), np.ones(len(verts_world))
    for k in range(len(verts_world)):
        V = verts_world[k, ::4].astype(np.float64)
        if not np.isfinite(V).all():
            continue
        Vc = (V - P[k]) @ R[k]
        px = cv2.fisheye.projectPoints(Vc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2) / 2
        x, y = np.round(px).astype(int).T
        ok = (x >= 0) & (x < labels.shape[2]) & (y >= 0) & (y < labels.shape[1])
        if ok.sum() > 50:
            grow = np.ones((9, 9), np.uint8)
            share[k] = cv2.dilate((labels[k] != 0).astype(np.uint8), grow)[y[ok], x[ok]].mean()
            hand_px = labels[k] == own_id
            if hand_px.sum() >= BIG_HAND_PX:
                own[k] = cv2.dilate(hand_px.astype(np.uint8), grow)[y[ok], x[ok]].mean()
    return share, own


def run_starts(flags: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Frames where a run of True in flags begins."""
    return frames[flags & ~np.r_[False, flags[:-1]]]


def at(frame: int):
    rr.set_time("frame", sequence=int(frame))
    rr.set_time("time", duration=frame / FPS)


def log(clip: Clip) -> dict | None:
    if not clip.has(*FILES):
        return None
    hands = pd.read_parquet(clip.dir / "hands.parquet")
    meta = clip.json("hands_meta.json")
    parent = meta["skeleton"]["soma25_parent"]
    bones25 = [(j, parent[j]) for j in range(1, len(parent))]
    mesh = np.load(clip.dir / "hands_mesh.npz") if clip.has("hands_mesh.npz") else None

    seen = {}
    for hi, hand in enumerate(("left", "right")):
        everything = hands[hands["hand"] == hand].sort_values("frame")
        # Shown only where the hand was detected and placed against measured depth. Elsewhere (filled-in gaps, or a
        # hand half out of the picture where the fisheye corners have no depth) it lands far off, e.g. under the table
        bad = ~everything["present"].to_numpy() | everything["qc_depth_resid_mm"].isna().to_numpy()
        present = ~binary_dilation(bad, iterations=SMOOTH_HALF) if bad.any() else ~bad
        # also not where the 3D hand lands clearly off the hand and objects in the picture (a pose or place gone wrong)
        if mesh is not None:
            scene, own = on_scene(clip, mesh["vertices_world"][:, hi], hand)
            off = (scene < ON_SCENE_MIN) | (own < ON_HAND_MIN)
            present &= ~binary_dilation(off, iterations=OFF_SCENE_PAD) if off.any() else True
        # spans the clip notes mark as a known bad fit (e.g. a grip the hand model can't read)
        t_s = everything["frame"].to_numpy() / FPS
        for a, b in clip.notes.get("hide_hand", {}).get(hand, []):
            present &= ~((t_s >= a) & (t_s <= b))
        t = everything[present]
        frames = t["frame"].to_numpy()
        n = len(frames)
        seen[hand] = n
        colour = HAND_COLOURS[hand]
        drawn = [f"world/hands/{hand}/skeleton", f"video/hands/{hand}/bones", f"video/hands/{hand}/keypoints"]

        # 3D surface: triangles logged as each run of detected frames starts, vertex positions per frame
        if mesh is not None:
            verts = mesh["vertices_world"][frames, hi].astype(np.float32)
            faces = mesh[f"faces_{hand}"]
            if clip.web:
                joints = np.stack(t["joints25_world"].to_numpy()).reshape(n, -1, 3)
                rest = int(np.argmax(np.linalg.norm(joints[:, TIPS, None] - joints[:, None, TIPS], axis=-1).sum((1, 2))))
                mean, faces = simplify(verts[rest], bone_of_vertices(verts[rest], joints[rest], parent), faces,
                                       SURFACE_VOXEL_M)
                verts = (mean @ verts.transpose(1, 0, 2).reshape(verts.shape[1], -1)).reshape(-1, n, 3).transpose(1, 0, 2)
                verts = verts.astype(np.float32)
            rr.log(f"world/hands/{hand}/surface", rr.Mesh3D.from_fields(albedo_factor=[*colour, 255]), static=True)
            for start in run_starts(present, everything["frame"].to_numpy()):
                at(start)
                rr.log(f"world/hands/{hand}/surface", rr.Mesh3D.from_fields(triangle_indices=faces))
            rr.reset_time()
            # Triangles must be in the file before the positions, or a viewer still loading it sees points without triangles
            rr.get_global_data_recording().flush()
            drawn.append(f"world/hands/{hand}/surface")
            rr.send_columns(f"world/hands/{hand}/surface", indexes=times(frames),
                            columns=rr.Mesh3D.columns(vertex_positions=verts.reshape(-1, 3)).partition([verts.shape[1]] * n))

        # 3D skeleton, 25 joints
        joints = np.stack(t["joints25_world"].to_numpy()).reshape(n, -1, 3)
        strips = joints[:, np.array(bones25)].reshape(-1, 2, 3)
        rr.log(f"world/hands/{hand}/skeleton", rr.LineStrips3D.from_fields(
            colors=[colour], radii=rr.Radius.ui_points(1.5)), static=True)
        rr.send_columns(f"world/hands/{hand}/skeleton", indexes=times(frames),
                        columns=rr.LineStrips3D.columns(strips=list(strips)).partition([len(bones25)] * n))

        # 2D keypoints on the video; grey = hidden behind an object or uncertain
        px = np.stack(t["kp21_px"].to_numpy()).reshape(n, 21, 2).astype(np.float32)
        visible = np.stack(t["kp21_visible"].to_numpy()).reshape(n, 21)
        point_colours = np.where(visible[..., None], colour, HIDDEN).reshape(-1, 3).astype(np.uint8)
        rr.log(f"video/hands/{hand}/bones", rr.LineStrips2D.from_fields(colors=[colour], radii=2.0, draw_order=20.0),
               static=True)
        rr.send_columns(f"video/hands/{hand}/bones", indexes=times(frames),
                        columns=rr.LineStrips2D.columns(strips=list(px[:, np.array(KP21_BONES)].reshape(-1, 2, 2)))
                        .partition([len(KP21_BONES)] * n))
        rr.log(f"video/hands/{hand}/keypoints", rr.Points2D.from_fields(radii=4.5, draw_order=21.0), static=True)
        rr.send_columns(f"video/hands/{hand}/keypoints", indexes=times(frames),
                        columns=rr.Points2D.columns(positions=px.reshape(-1, 2), colors=point_colours).partition([21] * n))

        for start in run_starts(~present, everything["frame"].to_numpy()):
            at(start)
            for entity in drawn:
                rr.log(entity, rr.Clear(recursive=False))
        rr.reset_time()

        wrist = np.stack(t["wrist_pos_world"].to_numpy())
        # Speed only between neighbouring frames, not across a gap where the hand wasn't seen
        step = np.diff(frames)
        speed = np.r_[np.nan, np.where(step == 1, np.linalg.norm(np.diff(wrist, axis=0), axis=1) * FPS * 100, np.nan)]
        label = f"{hand} hand"
        plot(f"plots/hands/grip_cm/{hand}", frames, t["aperture_thumb_index_m"] * 100, colour, label)
        plot(f"plots/hands/wrist_height_cm/{hand}", frames, wrist[:, 2] * 100, colour, label)
        plot(f"plots/hands/wrist_speed_cm_s/{hand}", frames, speed, colour, label)
        plot(f"plots/hands/check_depth_mm/{hand}", frames, t["qc_depth_resid_mm"], colour, label)

    q = meta["qc"]
    rows = [
        ("Distance to measured depth (median)", "depth_residual_after_median_mm", " mm"),
        ("Outline overlap with the hand mask", "mask_iou_median", ""),
        ("Joints visible", "visible_joint_pct", "%"),
        ("Finger bone length variation", "middle_finger_bone_length_variation_pct", "%"),
        ("Wrist jitter after smoothing", "wrist_jitter_smoothed_m_s2", " m/s²"),
        ("Left/right swaps", "left_right_swaps", ""),
    ]
    table = "\n".join(f"| {name} | {q['left'][k]}{unit} | {q['right'][k]}{unit} |" for name, k, unit in rows)
    total = len(hands) // 2
    coverage = (f"on all {total} frames" if seen["left"] == seen["right"] == total else
                f"on {seen['left']} (left) and {seen['right']} (right) of {total} frames. On the others the hand "
                "is out of the picture, too far into its edge to be placed against measured depth, or its 3D fit lands off "
                "the hand in the picture, and isn't drawn")
    hidden_note = (f"\n\n**Not drawn:** {clip.notes['hide_hand_note']}" if clip.notes.get("hide_hand_note") else "")
    note("notes/hands", f"""
### Hands
NVIDIA EgoHand-v1 on the fisheye video, with hand boxes from the SAM 3.1 masks. Each hand's distance comes from the hand model and is moved onto the measured depth where the two disagree by more than 2.5 cm (and at the image edge); it is then put in the room with the camera path and smoothed over 9 frames (0.18 s). Seen from the side, fingertips pressed flat on a surface can sit 2–4 cm into it, because the finger tilt comes from the hand model, not from depth.

25 joints per hand (the standard 21 keypoints are also saved), in metres, {coverage}.

| | Left | Right |
|---|---|---|
{table}

**On the video:** coloured dots are visible joints; grey dots are joints hidden behind an object or the other hand.{hidden_note}
**Grip:** thumb-tip to index-tip distance.
""")
    return {"Hands": "25 joints and surface per hand, in metres"}


def tab(clip: Clip):
    return rrb.Vertical(
        rrb.TimeSeriesView(origin="plots/hands/grip_cm", name="Grip: thumb–index opening (cm)"),
        rrb.TimeSeriesView(origin="plots/hands/wrist_speed_cm_s", name="Wrist speed (cm/s)"),
        rrb.TimeSeriesView(origin="plots/hands/wrist_height_cm", name=f"Wrist height above the {clip.surface} (cm)"),
        rrb.TimeSeriesView(origin="plots/hands/check_depth_mm", name="Check: distance to measured depth (mm)"),
        rrb.TextDocumentView(origin="notes/hands", name="About"),
        row_shares=[2, 2, 2, 2, 4], name="Hands")

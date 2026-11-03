"""Video, the head camera's path through the room, and what the camera's own sensors recorded."""
import numpy as np
import pandas as pd
import rerun as rr
import rerun.blueprint as rrb

from ego_rerun.clip import Clip, note, plot, times

FILES = ("camera_info.json", "camera_frames.parquet", "camera_path.parquet", "camera_path_report.json")
PATH_COLOUR = [110, 110, 110]  # grey: the hands are green and blue


def log(clip: Clip) -> dict | None:
    if not clip.has(*FILES):
        return None
    info = clip.json("camera_info.json")
    report = clip.json("camera_path_report.json")
    frames_df = pd.read_parquet(clip.dir / "camera_frames.parquet")
    path = pd.read_parquet(clip.dir / "camera_path.parquet")
    lens = clip.lens

    # Video, stored once; each frame points at its timestamp in the file
    video = rr.AssetVideo(path=clip.video)
    rr.log("video", video, static=True)
    ts = video.read_frame_timestamps_nanos()
    rr.send_columns("video", indexes=times(np.arange(len(ts))), columns=rr.VideoFrameReference.columns_nanos(ts))

    # Camera path in the room
    frames = path["frame"].to_numpy()
    pos = path[["x", "y", "z"]].to_numpy()
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world/camera_path", rr.LineStrips3D([pos], colors=[PATH_COLOUR], radii=rr.Radius.ui_points(1.5)), static=True)
    rr.send_columns("world/camera", indexes=times(frames), columns=rr.Transform3D.columns(
        translation=pos, quaternion=path[["qx", "qy", "qz", "qw"]].to_numpy()))
    # The frustum is drawn as a pinhole; the real lens is fisheye
    rr.log("world/camera", rr.Pinhole(focal_length=lens["fx"], width=lens["width"], height=lens["height"],
                                      camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=0.05), static=True)
    rr.log("world/camera", rr.VideoFrameReference.from_fields(video_reference="video"), static=True)
    rr.send_columns("world/camera", indexes=times(frames), columns=rr.VideoFrameReference.columns_nanos(ts[: len(frames)]))

    for axis, i, colour in (("x", 0, [230, 80, 80]), ("y", 1, [80, 180, 80]), ("z (height)", 2, [80, 120, 230])):
        plot(f"plots/camera/position_cm/{axis.split()[0]}", frames, pos[:, i] * 100, colour, axis)
    plot("plots/camera/check_rotation_deg/vs_sensor", frames, path["rotation_vs_sensor_deg"], [230, 160, 40],
         "rotation vs camera sensor (°)")
    plot("plots/camera/check_reprojection_px/error", frames, path["error_px"], [170, 90, 200], "reprojection error (px)")

    # Camera sensor: head turn rate from 1 kHz orientation (10 ms window), accelerometer, exposure
    imu = pd.read_parquet(clip.dir / "camera_orientation_1khz.parquet") if clip.has("camera_orientation_1khz.parquet") else None
    if imu is not None:
        from scipy.spatial.transform import Rotation

        rot = Rotation.from_quat(imu[["qx", "qy", "qz", "qw"]].to_numpy())
        t = imu["time_s"].to_numpy()
        w = 10
        rate = np.degrees((rot[w:] * rot[:-w].inv()).magnitude()) / (t[w:] - t[:-w])
        mid = np.arange(w // 2, w // 2 + len(rate))
        rr.log("plots/sensor/turn_rate", rr.SeriesLines(colors=[[230, 120, 60]], names=["turn rate (°/s)"]), static=True)
        rr.send_columns("plots/sensor/turn_rate", columns=rr.Scalars.columns(scalars=rate), indexes=[
            rr.TimeColumn("frame", sequence=imu["frame"].to_numpy()[mid]), rr.TimeColumn("time", duration=t[mid])])
    sf = frames_df["frame"].to_numpy()
    for axis, colour in (("x", [230, 80, 80]), ("y", [80, 180, 80]), ("z", [80, 120, 230])):
        plot(f"plots/sensor/accelerometer/{axis}", sf, frames_df[f"acc_{axis}"], colour, axis)
    plot("plots/sensor/iso/iso", sf, frames_df["iso"], [220, 180, 60], "ISO")

    rot = report["rotation_vs_camera_sensor_deg"]
    placed = (f"{report['frames_solved'].split(' of ')[0]} keyframes (every {report['keyframe_step']}th frame), all "
              "placed; the frames between follow the camera's motion sensor" if "keyframe_step" in report
              else report["frames_solved"])
    fitted = report.get("levelled_from_work_surface", {})
    level = (f"levelled from the {clip.surface if fitted.get('surface_mask') else 'work surface'} fitted in the depth"
             if fitted else "levelled to gravity from the accelerometer")
    sensor_note = ("\n\nThe GoPro's own orientation track drifts and isn't tied to gravity, so the comparison with it "
                   "mostly measures that track, not the path." if "sensor_comparison_note" in report else "")
    if "path_caveat" in clip.notes:
        sensor_note += f"\n\n**Note:** {clip.notes['path_caveat']}"
    lo, hi = frames_df["iso"].min(), frames_df["iso"].max()
    iso = f"{lo:.0f}" if lo == hi else f"{lo:.0f}–{hi:.0f}"
    note("notes/camera", f"""
### Camera
**{info['camera']}**, head-mounted, {lens['width']}×{lens['height']} at {info['fps']:g} fps, stabilisation {info['stabilisation']}.

| | |
|---|---|
| Lens | OpenCV fisheye, f = {lens['fx']:.1f} px, centre ({lens['cx']:.0f}, {lens['cy']:.0f}) |
| Rolling shutter | {info['rolling_shutter_readout_ms']:.2f} ms readout |
| Shutter / ISO | 1/{round(1 / frames_df['exposure_s'].median())} s, ISO {iso} |
| Motion sensor | orientation at {info['imu_rate_hz']} Hz, accelerometer per frame |

### Camera path
Structure from motion (COLMAP, SIFT) with the camera's own lens held fixed; scaled to metres from metric depth; {level}.

| | |
|---|---|
| Frames placed | {placed} |
| Reprojection error | {report['average_error_px']} px average |
| Rotation vs the camera's sensor | median {rot['median']}°, 95th percentile {rot['p95']}°, max {rot['max']}° |
| Head height above the {clip.surface} | {report['head_height_above_table_m']} m |

**World frame:** metres, z up, origin on the {clip.surface}'s plane straight below the head, x = mean looking direction.
**Camera frame:** x right, y down, z forward (OpenCV).{sensor_note}
""")
    agree = "" if sensor_note else f", {rot['median']}° from the camera's sensor"
    return {"Camera path": f"Structure from motion, metres{agree}"}


def tab(clip: Clip):
    views = [
        rrb.TimeSeriesView(origin="plots/camera/position_cm", name="Head position (cm)"),
        rrb.TimeSeriesView(origin="plots/camera/check_rotation_deg", name="Check: rotation vs camera sensor (°)"),
        rrb.TimeSeriesView(origin="plots/sensor/turn_rate", name="Head turn rate (°/s)"),
        rrb.TimeSeriesView(origin="plots/sensor/accelerometer", name="Accelerometer (g)"),
        rrb.TextDocumentView(origin="notes/camera", name="About"),
    ]
    return rrb.Vertical(*views, row_shares=[2, 2, 2, 2, 3], name="Camera")

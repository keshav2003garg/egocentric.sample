"""The sample.zygot.ai site: an S3 bucket behind CloudFront, password-protected by a CloudFront Function. The page
lists every episode in episodes.json and opens one at a time; publishing clips adds or replaces their entries.

`preview` lays the same site out locally and serves it, so it can be checked before publishing.

`publish` signs in with the keys in .env.production.local and uploads into the bucket the SST stack made
(`pnpm deploy:prod` writes its name and the CloudFront ID to .sst/outputs.json). Uploads:
- episodes/<clip>.rrd (gzip), with its uncompressed size for the loading bar
- episodes.json, the list of episodes (never cached)
- rerun-<version>/re_viewer.js and re_viewer_bg.wasm (gzip), copied from the installed Rerun so the viewer
  always matches the SDK that wrote the episode files; a new folder per version, so stale caches can't mix them
- index.html (never cached), filled in from site/index.html
then clears the CloudFront cache.
"""
import gzip
import http.server
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from functools import partial
from pathlib import Path

import rerun as rr

ROOT = Path(__file__).resolve().parents[2]
SITE = "https://sample.zygot.ai"
VIEWER = f"rerun-{rr.__version__}"
PREVIEW_PORT = 4200


def site_env() -> dict[str, str]:
    """The keys in .env.production.local (KEY=VALUE lines)."""
    path = ROOT / ".env.production.local"
    if not path.exists():
        raise SystemExit(f"{path.name} is missing: it holds the sample site's AWS keys")
    pairs = (line.split("=", 1) for line in path.read_text().splitlines() if "=" in line and not line.startswith("#"))
    return {k.strip(): v.strip() for k, v in pairs}


def stack() -> dict[str, str]:
    """The SST stack's outputs: bucket, distribution, url."""
    path = ROOT / ".sst" / "outputs.json"
    if not path.exists():
        raise SystemExit("no .sst/outputs.json: deploy the stack first (pnpm deploy:prod)")
    return json.loads(path.read_text())


def aws(*args: str) -> str:
    # only the site's keys: no shell profile or session
    env = {k: v for k, v in os.environ.items() if k not in ("AWS_PROFILE", "AWS_SESSION_TOKEN")}
    env.update({k: v for k, v in site_env().items() if k.startswith("AWS_")})
    return subprocess.run(["aws", *args], check=True, capture_output=True, text=True, env=env).stdout


def viewer_files(dest: Path) -> None:
    """Copies the web viewer out of the installed Rerun by serving it briefly on a free port."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    rerun = Path(sys.executable).with_name("rerun")
    server = subprocess.Popen([str(rerun), "--serve-web", "--web-viewer-port", str(port), "--port", str(port + 1)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for name in ("re_viewer.js", "re_viewer_bg.wasm", "favicon.ico", "apple-touch-icon.png"):
            for _ in range(50):
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/{name}") as r:
                        (dest / name).write_bytes(r.read())
                    break
                except OSError:
                    time.sleep(0.2)
            else:
                raise SystemExit(f"could not copy {name} from the Rerun viewer")
    finally:
        server.terminate()


def entry(clip_name: str, title: str) -> dict:
    return {"id": clip_name, "title": title, "file": f"episodes/{clip_name}.rrd", "version": time.strftime("%Y%m%d%H%M%S")}


def merged(episodes: list, new: list) -> list:
    ids = {e["id"] for e in new}
    return sorted([e for e in episodes if e["id"] not in ids] + new, key=lambda e: e["id"])


def page() -> str:
    return (ROOT / "site/index.html").read_text().replace("{{VIEWER}}", VIEWER)


def preview(clips: list[tuple[str, Path, str]]) -> None:
    """Lays the site out in out/site (episode files linked, not copied) and serves it until stopped."""
    site = ROOT / "out/site"
    shutil.rmtree(site, ignore_errors=True)
    (site / VIEWER).mkdir(parents=True)
    (site / "episodes").mkdir()
    viewer_files(site / VIEWER)
    for icon in ("favicon.ico", "apple-touch-icon.png"):
        (site / VIEWER / icon).rename(site / icon)
    for name, rrd, _ in clips:
        (site / "episodes" / f"{name}.rrd").symlink_to(rrd)
    (site / "episodes.json").write_text(json.dumps(merged([], [entry(n, t) for n, _, t in clips]), indent=2))
    (site / "index.html").write_text(page())

    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(site))
    print(f"open http://localhost:{PREVIEW_PORT}  ({len(clips)} episode{'s' if len(clips) != 1 else ''}; Ctrl-C to stop)")
    http.server.ThreadingHTTPServer(("127.0.0.1", PREVIEW_PORT), handler).serve_forever()


def upload(bucket: str, path: Path, key: str, content_type: str, cache: str, gzipped: bool = False,
           size: int | None = None):
    args = ["s3", "cp", str(path), f"s3://{bucket}/{key}", "--content-type", content_type, "--cache-control", cache]
    if gzipped:
        args += ["--content-encoding", "gzip"]
    if size is not None:
        args += ["--metadata", f"uncompressed-size={size}"]
    aws(*args)
    print(f"  {key}")


def publish(clips: list[tuple[str, Path, str]]) -> None:
    expected = site_env()["AWS_ACCOUNT_ID"]
    account = json.loads(aws("sts", "get-caller-identity"))["Account"]
    if account != expected:
        raise SystemExit(f"AWS keys are for account {account}; the site lives in {expected}")
    out = stack()
    bucket = out["bucket"]

    try:
        episodes = json.loads(aws("s3", "cp", f"s3://{bucket}/episodes.json", "-"))
    except subprocess.CalledProcessError:
        episodes = []
    new = [entry(n, t) for n, _, t in clips]
    episodes = merged(episodes, new)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        viewer_files(tmp)
        raw = (tmp / "re_viewer_bg.wasm").read_bytes()
        (tmp / "re_viewer_bg.wasm.gz").write_bytes(gzip.compress(raw, 9))
        (tmp / "index.html").write_text(page())
        (tmp / "episodes.json").write_text(json.dumps(episodes, indent=2))

        print(f"uploading to s3://{bucket}")
        upload(bucket, tmp / "re_viewer.js", f"{VIEWER}/re_viewer.js", "application/javascript", "public,max-age=31536000,immutable")
        upload(bucket, tmp / "re_viewer_bg.wasm.gz", f"{VIEWER}/re_viewer_bg.wasm", "application/wasm",
               "public,max-age=31536000,immutable", gzipped=True, size=len(raw))
        upload(bucket, tmp / "favicon.ico", "favicon.ico", "image/x-icon", "public,max-age=86400")
        upload(bucket, tmp / "apple-touch-icon.png", "apple-touch-icon.png", "image/png", "public,max-age=86400")
        for (_, rrd, _), e in zip(clips, new):
            with rrd.open("rb") as src, gzip.open(tmp / "episode.rrd.gz", "wb", compresslevel=6) as dst:
                shutil.copyfileobj(src, dst)
            upload(bucket, tmp / "episode.rrd.gz", e["file"], "application/octet-stream", "private,max-age=3600", gzipped=True,
                   size=rrd.stat().st_size)
        upload(bucket, tmp / "episodes.json", "episodes.json", "application/json", "no-cache")
        upload(bucket, tmp / "index.html", "index.html", "text/html;charset=utf-8", "no-cache")

    aws("cloudfront", "create-invalidation", "--distribution-id", out["distribution"],
        "--paths", "/", "/index.html", "/episodes.json", *[f"/{e['file']}" for e in new])
    print(f"published {', '.join(t for _, _, t in clips)} (viewer {rr.__version__}); "
          f"{len(episodes)} episode{'s' if len(episodes) != 1 else ''} on {SITE}")

"""ego-rerun build|serve|web|preview|publish [CLIP ...|all]"""
import argparse
import os
import sys
from pathlib import Path

from ego_rerun.build import build
from ego_rerun.clip import NOTES_ROOT, Clip

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLIP = "004_paint-shoe-sole-edge"
WEB_PORT, GRPC_PORT = 4100, 4101


def clip_names(args: list[str]) -> list[str]:
    """'all' means every clip with notes in clips/."""
    if args == ["all"]:
        return sorted(p.stem for p in NOTES_ROOT.glob("*.json"))
    return args or [DEFAULT_CLIP]


def main():
    parser = argparse.ArgumentParser(prog="ego-rerun")
    parser.add_argument("command", choices=["build", "serve", "web", "preview", "publish"],
                        help="build: local viewer file · serve: build if needed, then open it · "
                             "web: smaller file for the hosted site · preview: web build, then the site served locally · "
                             "publish: web build, then upload to sample.zygot.ai")
    parser.add_argument("clips", nargs="*", help=f"clip names, or 'all' (default {DEFAULT_CLIP})")
    args = parser.parse_args()
    names = clip_names(args.clips)

    if args.command in ("web", "preview", "publish"):
        built = []
        for name in names:
            print(f"web build of {name}")
            out = build(name, ROOT / "out" / f"{name}_web.rrd", web=True)
            print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")
            built.append((name, out, Clip(name).title))
        if args.command == "preview":
            from ego_rerun.publish import preview

            preview(built)
        elif args.command == "publish":
            from ego_rerun.publish import publish

            publish(built)
        return

    outs = []
    for name in names:
        out = ROOT / "out" / f"{name}.rrd"
        if args.command == "build" or not out.exists():
            print(f"building {name}")
            build(name, out)
            print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")
        outs.append(out)
    if args.command == "serve":
        print(f"open http://localhost:{WEB_PORT}/?url=rerun%2Bhttp%3A%2F%2Flocalhost%3A{GRPC_PORT}%2Fproxy")
        rerun = Path(sys.executable).with_name("rerun")
        os.execv(rerun, [str(rerun), "--serve-web", "--web-viewer-port", str(WEB_PORT), "--port", str(GRPC_PORT),
                         "--server-memory-limit", "6GB", *map(str, outs)])


if __name__ == "__main__":
    main()

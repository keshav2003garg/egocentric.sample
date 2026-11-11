# ego_sample_rerun

Builds Rerun viewer files for the ego sample clips from their finished layer files in `~/Developer/ego_sample/<clip>/`, and publishes them as episodes on sample.zygot.ai. The pipeline that makes those files lives elsewhere; this repo only reads them.

## Clips

| Clip | Title | Notes |
|---|---|---|
| 001_stuff-shoe-with-newspaper | Stuff shoe with newspaper | contact mostly set by review (paper ball not outlined) |
| 002_brush-primer-on-lasted-shoe | Brush primer on lasted shoe | brush tool |
| 003_last-shoe-upper-with-pincers | Last shoe upper with pincers | lasting pincers tool |
| 004_paint-shoe-sole-edge | Paint shoe sole edge | brush tool; richer contact data (see Layers) |
| 005_press-sole-onto-shoe | Press sole onto shoe | |
| 006_apply-solution-along-sole-seam-with-syringe | Apply solution along sole seam | syringe tool |
| 007_screen-print-shoe-upper | Screen print shoe upper | squeegee tool |
| 008_toe-last-shoe-on-machine | Toe last shoe on machine | standing at a machine |
| 009_die-cut-sheet-on-clicker-press | Die-cut sheet on clicker press | cutting die tool |
| 010_bag-shoes-and-put-them-in-the-box | Bag shoes and put them in the box | GoPro |
| 014_position-cutting-die-and-cut-leather | Position cutting die and cut leather | cutting die tool |
| 018_insert-insole-into-shoe | Insert insole into shoe | |
| 019_lace-shoe | Lace shoe | 34 s, the longest |
| 020_trace-sole-line-on-lasted-shoe | Trace sole line on lasted shoe | marker tool |
| 024_write-carton-label | Write carton label | marker tool |

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
```

## Use

```bash
.venv/bin/ego-rerun build                      # writes out/004_paint-shoe-sole-edge.rrd
.venv/bin/ego-rerun serve                      # builds if needed, then serves it
.venv/bin/ego-rerun preview all                # the hosted site, served locally with every clip
```

`serve`: open http://localhost:4100/?url=rerun%2Bhttp%3A%2F%2Flocalhost%3A4101%2Fproxy · `preview`: open http://localhost:4200

Every command takes clip names (`ego-rerun build 019_lace-shoe 024_write-carton-label`) or `all` (every clip in `clips/`); with none it uses 004.

## Layout

- **Left:** the video with masks and hand keypoints; the depth image beside it; the 3D scene below (camera path, hands).
- **Right, top:** "Now": the current step, its description, and what each hand and the tool are doing.
- **Right, below:** one tab per layer (Overview, Steps, Contact, Hands, Masks, Depth, Camera), each with its plots and an "About" note.

## Layers

Each module in `src/ego_rerun/layers/` logs one layer and supplies its tab. A layer whose files aren't there is skipped, so the viewer can be rebuilt at any stage.

Object poses and 3D object models are left out on purpose, so every episode shows the same layers (only 004 has them; its `objects_*` files are not read).

Contact comes in two forms. Clip 004's was measured against 3D object models (thumb/index contact points, brush tip painting/dipping). The other clips' is judged against each object's visible surface from masks and depth, with the tool read as held / in use on an object. `layers/contact.py` handles both.

Object colours are given per clip in mask order (hands are always green and blue), so each clip's legend is short.

| Layer | Files read |
|---|---|
| camera | `<clip>.mp4`, `camera_info.json`, `camera_frames.parquet`, `camera_orientation_1khz.parquet`, `camera_path.parquet`, `camera_path_report.json` |
| depth | `depth.npz`, `depth_report.json` (check numbers; 004's are in its clip notes) |
| masks | `masks.npz`, `masks.parquet` |
| hands | `hands.parquet`, `hands_mesh.npz`, `hands_meta.json` |
| contact | `contact.parquet`, `contact_events.parquet`, `contact_meta.json`, `tool_contact.parquet` |
| steps | `action_steps.parquet`, `task.json` (also drives the live "Now" panel) |

To add a layer, add a module with `log(clip)` and `tab(clip)` and list it in `layers/__init__.py` (`LAYERS` for logging order, `TAB_ORDER` for the tabs).

## Publishing to sample.zygot.ai

This repo is the source of sample.zygot.ai (AWS account 311353783217). The site's bucket, CloudFront and `sample.zygot.ai` address are an SST stack (`sst.config.ts`); the keys and the Cloudflare token are in `.env.production.local` (git-ignored). The DNS record is proxied by Cloudflare, and a custom security rule on the zygot.ai zone ("Bot check on sample.zygot.ai", Managed Challenge) checks visitors are human before the site loads.

```bash
pnpm install && pnpm deploy:prod               # make or update the stack (bucket, CloudFront, domain)
.venv/bin/ego-rerun preview all                # check it locally first
.venv/bin/ego-rerun publish all                # web build of every clip, then upload them as episodes
```

`pnpm remove:prod` deletes the stack. `publish` signs in with the keys in `.env.production.local` and finds the bucket and CloudFront ID in `.sst/outputs.json`, which the deploy writes. It:
- builds a smaller copy of each clip (`ego-rerun web`): re-encoded video, half-size depth images, lighter hand surfaces;
- uploads each to `episodes/<clip>.rrd` and adds or replaces its entry in `episodes.json` (other episodes stay);
- uploads the web viewer from the installed Rerun to `rerun-<version>/`, so the viewer always matches the files;
- uploads `index.html` from `site/index.html`, then clears the CloudFront cache.

The paint-sole-edge episode (`FIRST` in `publish.py`) is listed first and opens by default; the rest follow by clip number.

The page has an episode menu at the top and opens one episode at a time (`?episode=<clip>`, else the first), so a visitor only downloads the episode they look at, and each episode has its own link.

## Clip notes

`clips/<clip>.json` holds what the data files don't: the title, setting, what the work happens on (`surface`, default table; 018 is on the floor), display names for objects, and for 004 the depth check numbers and capture date (the others take the date from the source recording's name). To add a clip, give it a `clips/<clip>.json`; `all` then includes it.

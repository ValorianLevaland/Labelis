# Labelis Render-Only (Napari) – standalone reconstruction + TIFF export

This mini-app is a *render-only* front-end that reuses the Labelis backend:
- robust CSV/TSV localization loading (Labelis `load_localizations`)
- super-resolution reconstruction rendering (Labelis `render_dispatch`)
- exports the rendered reconstruction as TIFF (with pixel size stored as OME PhysicalSizeX/Y + classic TIFF resolution tags)
- writes a small JSON sidecar with render settings and XY offsets

## How to place it (recommended)

Put the folder **next to** your `labelis/` source folder, e.g.

    labelis_upgraded_app/
      labelis/
      labelis_render_only/
        render_only_app.py
        Run_RenderOnly.bat

The scripts auto-add the parent folder to `sys.path`, so you **do not** need `pip install -e .`
as long as `labelis/` is in the same parent directory.

## Run (GUI)

    python render_only_app.py

Or on Windows, double-click:

    Run_RenderOnly.bat

## CLI (headless)

    python render_only_cli.py --input my.csv --output out.tif --px-nm 10 --engine numba

## Notes / gotchas

- By default, XY are shifted so the minimum X/Y become 0 (same as the Labelis GUI/pipeline),
  which keeps image sizes reasonable if your coordinates are large.
- The app uses its own `NUMBA_CACHE_DIR` (under your home folder) to avoid Numba cache collisions
  between different Labelis checkouts/paths (a common cause of `ModuleNotFoundError: labelis` during rendering).

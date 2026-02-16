\
    from __future__ import annotations

    import argparse
    import json
    from pathlib import Path

    import numpy as np
    import tifffile

    from labelis.pipeline.io import load_localizations
    from labelis.pipeline.render import render_dispatch


    def save_render_tiff(
        out_path: str | Path,
        img: np.ndarray,
        *,
        pixel_size_nm: float,
        meta: dict,
    ) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Store physical pixel size as OME metadata in µm (widely understood).
        px_um = float(pixel_size_nm) / 1000.0
        ome_meta = {
            "axes": "YX",
            "PhysicalSizeX": px_um,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": px_um,
            "PhysicalSizeYUnit": "µm",
        }

        # Also store pixels-per-cm in classic TIFF resolution tags.
        px_cm = float(pixel_size_nm) * 1e-7  # 1 nm = 1e-7 cm
        ppcm = (1.0 / px_cm) if px_cm > 0 else 1.0

        tifffile.imwrite(
            out_path,
            np.asarray(img, dtype=np.float32),
            ome=True,
            metadata=ome_meta,
            description=json.dumps(meta, indent=2),
            resolution=(ppcm, ppcm),
            resolutionunit="CENTIMETER",
        )

        # Sidecar JSON (easy to inspect without TIFF tooling)
        out_path.with_suffix(out_path.suffix + ".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


    def main() -> int:
        ap = argparse.ArgumentParser(description="Render a Labelis-style reconstruction from a localization CSV/TSV and save as TIFF.")
        ap.add_argument("--input", "-i", required=True, help="Input localization table (.csv/.tsv)")
        ap.add_argument("--output", "-o", required=True, help="Output TIFF path (.tif/.tiff)")
        ap.add_argument("--px-nm", type=float, default=10.0, help="Rendered pixel size in nm")
        ap.add_argument("--engine", choices=["turbo_bin_blur", "numba", "reference_bruteforce"], default="numba")
        ap.add_argument("--turbo-blur-sigma-px", type=float, default=1.2, help="Only for turbo_bin_blur")
        ap.add_argument("--signal-half-width-px", type=int, default=10, help="Kernel half width (Gaussian stamping engines)")
        ap.add_argument("--sigma-is-variance", action="store_true", help="Interpret sigma column as variance (Labelis/MATLAB-compatible)")
        ap.add_argument("--sigma-is-std", dest="sigma_is_variance", action="store_false", help="Interpret sigma column as std-dev")
        ap.set_defaults(sigma_is_variance=True)
        ap.add_argument("--compat-kernel-crop", action="store_true", help="Match MATLAB boundary-crop quirk")
        ap.add_argument("--no-compat-kernel-crop", dest="compat_kernel_crop", action="store_false")
        ap.set_defaults(compat_kernel_crop=True)
        ap.add_argument("--shift-to-origin", action="store_true", help="Shift XY so min becomes 0 (recommended)")
        ap.add_argument("--no-shift-to-origin", dest="shift_to_origin", action="store_false")
        ap.set_defaults(shift_to_origin=True)

        args = ap.parse_args()

        df = load_localizations(args.input)

        x_offset_nm = float(df["x_nm_"].min())
        y_offset_nm = float(df["y_nm_"].min())
        if args.shift_to_origin:
            df = df.copy()
            df["x_nm_"] = df["x_nm_"] - x_offset_nm
            df["y_nm_"] = df["y_nm_"] - y_offset_nm
        else:
            x_offset_nm = 0.0
            y_offset_nm = 0.0

        img = render_dispatch(
            engine=str(args.engine),
            x_nm=df["x_nm_"].to_numpy(),
            y_nm=df["y_nm_"].to_numpy(),
            sigma_nm=df["sigma_nm_"].to_numpy(),
            pixel_size_nm=float(args.px_nm),
            grid_size_px=None,
            signal_half_width_px=int(args.signal_half_width_px),
            sigma_is_variance=bool(args.sigma_is_variance),
            compat_kernel_crop=bool(args.compat_kernel_crop),
            turbo_blur_sigma_px=float(args.turbo_blur_sigma_px),
        )

        meta = {
            "input": str(Path(args.input).resolve()),
            "output": str(Path(args.output).resolve()),
            "n_localizations": int(len(df)),
            "render_px_size_nm": float(args.px_nm),
            "engine": str(args.engine),
            "signal_half_width_px": int(args.signal_half_width_px),
            "sigma_is_variance": bool(args.sigma_is_variance),
            "compat_kernel_crop": bool(args.compat_kernel_crop),
            "turbo_blur_sigma_px": float(args.turbo_blur_sigma_px),
            "shift_to_origin": bool(args.shift_to_origin),
            "x_offset_nm": float(x_offset_nm),
            "y_offset_nm": float(y_offset_nm),
            "image_shape": [int(img.shape[0]), int(img.shape[1])],
        }

        save_render_tiff(args.output, img, pixel_size_nm=float(args.px_nm), meta=meta)
        print(f"Saved: {args.output}")
        return 0


    if __name__ == "__main__":
        raise SystemExit(main())

"""CLI `tatami-train`."""

from __future__ import annotations

import argparse
import json
import sys

from .embed import DEFAULT_BACKBONE


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tatami-train", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("extract", help="frames de videos + etiquetas por intervalos -> dataset")
    s.add_argument("--videos", required=True, help="directorio con videos")
    s.add_argument("--labels", required=True, help="directorio con .json de intervalos")
    s.add_argument("--out", required=True, help="directorio del dataset")
    s.add_argument("--fps", type=float, default=2.0)
    s.add_argument("--max-side", type=int, default=512)
    s.add_argument("--val-frac", type=float, default=0.15)
    s.add_argument("--test-frac", type=float, default=0.15)
    s.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("split", help="re-asigna splits por video en un dataset existente")
    s.add_argument("dataset")
    s.add_argument("--val-frac", type=float, default=0.15)
    s.add_argument("--test-frac", type=float, default=0.15)
    s.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("embed", help="embeddings con un backbone (cache .npy)")
    s.add_argument("dataset")
    s.add_argument("--backbone", default=DEFAULT_BACKBONE)
    s.add_argument("--batch-size", type=int, default=32)
    s.add_argument("--device", default=None, help="mps|cpu|cuda (default: auto)")

    s = sub.add_parser("train", help="entrena logreg y MLP sobre embeddings")
    s.add_argument("dataset")
    s.add_argument("--out", required=True, help="directorio del run")
    s.add_argument("--backbone", default=DEFAULT_BACKBONE)
    s.add_argument(
        "--targets", nargs="+", default=["position", "top"], choices=["position", "top", "top_side"]
    )
    s.add_argument("--models", nargs="+", default=["logreg", "mlp"], choices=["logreg", "mlp"])
    s.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("eval", help="métricas y matrices de confusión de un run")
    s.add_argument("run")
    s.add_argument("--split", default="test", choices=["train", "val", "test"])
    s.add_argument("--dataset", default=None)

    s = sub.add_parser("export", help="exporta model.joblib + meta.json para tatami-vision")
    s.add_argument("run")
    s.add_argument("--out", required=True)
    s.add_argument("--model", default="logreg", choices=["logreg", "mlp"])

    s = sub.add_parser("pseudolabel", help="pre-etiqueta un video con un VLM (tatami-vision)")
    s.add_argument("video", nargs="?")
    s.add_argument("--out", required=True, help="directorio de salida de etiquetas")
    s.add_argument("--backend", default="ollama")
    s.add_argument("--model", default="qwen3-vl:8b-instruct")
    s.add_argument("--sample-fps", type=float, default=2.0)
    s.add_argument("--from-result", default=None, help="AnalysisResult JSON ya generado")

    s = sub.add_parser("eval-analysis", help="evalúa AnalysisResult(s) contra etiquetas")
    s.add_argument("results", nargs="+", help="result.json de tatami-vision")
    s.add_argument("--labels", required=True, help="etiquetas por intervalos del mismo video")
    s.add_argument("--json", action="store_true", help="métricas completas en JSON")

    s = sub.add_parser("bench-vlm", help="compara variantes de análisis VLM contra etiquetas")
    s.add_argument("video")
    s.add_argument("--labels", required=True)
    s.add_argument(
        "--variant",
        action="append",
        required=True,
        help="nombre:clave=valor,... (kwargs de analyze_video), p.ej. mosaico:tile=4",
    )
    s.add_argument("--out", required=True, help="directorio para resultados y bench.json")
    s.add_argument("--backend", default="ollama")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "extract":
        from .dataset import extract_dataset

        df = extract_dataset(
            args.videos,
            args.labels,
            args.out,
            args.fps,
            args.max_side,
            args.val_frac,
            args.test_frac,
            args.seed,
        )
        print(f"{len(df)} frames -> {args.out}")
        print(df.groupby("split")["video"].agg(["count", "nunique"]).to_string())
    elif args.cmd == "split":
        from .dataset import load_index
        from .split import assign_splits

        df = assign_splits(load_index(args.dataset), args.val_frac, args.test_frac, args.seed)
        df.to_csv(f"{args.dataset}/index.csv", index=False)
        print(df.groupby("split")["video"].agg(["count", "nunique"]).to_string())
    elif args.cmd == "embed":
        from .embed import embed_index

        X = embed_index(args.dataset, args.backbone, None, args.batch_size, args.device)
        print(f"embeddings {X.shape} guardados")
    elif args.cmd == "train":
        from .train import train_run

        info = train_run(
            args.dataset,
            args.out,
            args.backbone,
            tuple(args.targets),
            tuple(args.models),
            args.seed,
        )
        print(json.dumps(info["targets"], indent=2))
    elif args.cmd == "eval":
        from .evaluate import evaluate_run, format_summary

        print(format_summary(evaluate_run(args.run, args.split, args.dataset)))
    elif args.cmd == "export":
        from .export import export_model

        print(json.dumps(export_model(args.run, args.out, args.model), indent=2))
    elif args.cmd == "pseudolabel":
        from .pseudolabel import pseudolabel_video

        lf = pseudolabel_video(
            args.video, args.out, args.backend, args.model, args.sample_fps, args.from_result
        )
        print(f"{len(lf.intervals)} intervalos (source=vlm) para {lf.video}")
    elif args.cmd == "eval-analysis":
        from pathlib import Path

        from .intervals import load_label_file
        from .vlm_eval import evaluate_analysis, format_table

        labels = load_label_file(args.labels)
        rows = [
            {
                "name": Path(r).stem,
                "metrics": evaluate_analysis(json.loads(Path(r).read_text()), labels),
            }
            for r in args.results
        ]
        print(json.dumps(rows, indent=2) if args.json else format_table(rows))
    elif args.cmd == "bench-vlm":
        from .vlm_eval import bench_vlm, format_table

        print(
            format_table(bench_vlm(args.video, args.labels, args.variant, args.out, args.backend))
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

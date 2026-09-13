import argparse
import subprocess
import sys
from pathlib import Path

import srt
import torch
import argostranslate.package
import argostranslate.translate


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def default_model_for_device(device: str) -> str:
    # CPU-only machines get a smaller default model to keep runtime sane
    return "medium" if device == "cuda" else "small"


def transcribe(video_path: Path, model: str, source_lang: str, device: str) -> Path:
    srt_path = video_path.with_suffix(".srt")
    cmd = [
        "whisper", str(video_path),
        "--model", model,
        "--device", device,
        "-f", "srt",
        "--output_dir", str(video_path.parent),
    ]
    if source_lang != "auto":
        cmd += ["--language", source_lang]
    print(f"[1/3] Transcribing with Whisper ({model}) on {device}...")
    subprocess.run(cmd, check=True)
    return srt_path


def translate_srt(srt_path: Path, from_code: str, to_code: str) -> Path:
    print(f"[2/3] Translating {from_code} -> {to_code} with Argos Translate...")
    installed = argostranslate.translate.get_installed_languages()
    if not any(l.code == to_code for l in installed):
        argostranslate.package.update_package_index()
        available = argostranslate.package.get_available_packages()
        pkg = next(p for p in available if p.from_code == from_code and p.to_code == to_code)
        argostranslate.package.install_from_path(pkg.download())

    with open(srt_path, encoding="utf-8") as f:
        subs = list(srt.parse(f.read()))

    for sub in subs:
        sub.content = argostranslate.translate.translate(sub.content, from_code, to_code)

    out_path = srt_path.with_name(f"{srt_path.stem}_{to_code}.srt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))
    return out_path


def burn_subtitles(video_path: Path, srt_path: Path, hardcode: bool) -> Path:
    print(f"[3/3] {'Burning in' if hardcode else 'Embedding'} subtitles with ffmpeg...")
    out_path = video_path.with_name(f"{video_path.stem}_subbed{video_path.suffix}")
    if hardcode:
        vf = f"subtitles={srt_path.name}:force_style='FontName=Noto Sans,FontSize=24'"
        cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vf", vf, str(out_path)]
    else:
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path), "-i", str(srt_path),
            "-c", "copy", "-c:s", "mov_text",
            "-metadata:s:s:0", "language=gre", str(out_path),
        ]
    subprocess.run(cmd, check=True, cwd=video_path.parent)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Transcribe, translate, and subtitle a video locally (CUDA or CPU).")
    parser.add_argument("video", type=Path, help="Path to input video file")
    parser.add_argument("--model", default=None, help="Whisper model (tiny/base/small/medium/large-v3). Auto-picked if omitted.")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"], help="Force device. Auto-detected if omitted.")
    parser.add_argument("--from-lang", default="en", help="Source language code (e.g. en)")
    parser.add_argument("--to-lang", default="el", help="Target language code (e.g. el for Greek)")
    parser.add_argument("--hardcode", action="store_true", help="Burn subtitles into video instead of soft subs")
    args = parser.parse_args()

    if not args.video.exists():
        sys.exit(f"File not found: {args.video}")

    device = args.device or get_device()
    model = args.model or default_model_for_device(device)

    srt_path = transcribe(args.video, model, args.from_lang, device)
    translated_srt = translate_srt(srt_path, args.from_lang, args.to_lang)
    output = burn_subtitles(args.video, translated_srt, args.hardcode)
    print(f"Done: {output}")


if __name__ == "__main__":
    main()
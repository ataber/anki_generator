#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import genanki
from tqdm import tqdm


# ----------------------------
# Utilities
# ----------------------------


def load_sentences(
    path: Path,
    *,
    text_column: Optional[str] = None,
    english_column: Optional[str] = None,
    level_column: str = "Level",
) -> Tuple[List[Tuple[str, str, str]], Optional[str]]:
    """
    Load a sentence corpus for Anki.

    Supports:
      - JSON: list of [Text, English] OR [Level, Text, English]
      - CSV: header row with at least a text column and an English column.

    Returns: (list of (level, text, english), resolved_text_column_name).
    The text column name is None for JSON input.

    Column selection (CSV):
      - level_column defaults to "Level" (if missing, "UNK" is used)
      - english_column defaults to "Natural English Translation", then "English", then "Translation"
      - text_column defaults to the first column that is not the level or English column
        (common values: Greek, Korean, Spanish)
    """
    ext = path.suffix.lower()

    def norm(s: str) -> str:
        return (s or "").strip()

    if ext == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON input must be a list of pairs/triples.")
        out: List[Tuple[str, str, str]] = []
        for idx, item in enumerate(data, start=1):
            if not isinstance(item, list):
                raise ValueError(f"Bad JSON item at {idx}: expected list, got {type(item)}")
            if len(item) == 2:
                txt, eng = norm(item[0]), norm(item[1])
                lvl = "UNK"
            elif len(item) == 3:
                lvl, txt, eng = norm(item[0]), norm(item[1]), norm(item[2])
                lvl = lvl or "UNK"
            else:
                raise ValueError(f"Bad JSON item at {idx}: expected [Text, English] or [Level, Text, English]")
            if not txt or not eng:
                raise ValueError(f"Bad JSON item at {idx}: empty text or English.")
            out.append((lvl, txt, eng))
        return out, None

    if ext == ".csv":
        text = path.read_text(encoding="utf-8-sig")
        reader = csv.DictReader(text.splitlines())
        if not reader.fieldnames:
            raise ValueError("CSV must have a header row.")
        fieldnames = [f.strip() for f in reader.fieldnames if f]

        def pick(row: dict, key: str) -> str:
            if key in row and row[key] is not None:
                return norm(row[key])
            for k in row.keys():
                if k and k.strip().lower() == key.strip().lower():
                    return norm(row.get(k, ""))
            return ""

        # Resolve English column
        if english_column is None:
            for cand in ("Natural English Translation", "English", "Translation"):
                if any(fn.lower() == cand.lower() for fn in fieldnames):
                    english_column = cand
                    break
        if english_column is None:
            raise ValueError(f"Could not find an English column. Available columns: {fieldnames}")

        # Resolve text column
        if text_column is None:
            preferred = ("Greek", "Korean", "Spanish", "Text", "Sentence", "Target")
            for cand in preferred:
                if any(fn.lower() == cand.lower() for fn in fieldnames):
                    text_column = cand
                    break
        if text_column is None:
            excluded = {level_column.lower(), english_column.lower(), "audio", "sound"}
            for fn in fieldnames:
                if fn.strip().lower() not in excluded:
                    text_column = fn
                    break
        if text_column is None:
            raise ValueError(f"Could not determine text column. Available columns: {fieldnames}")

        out: List[Tuple[str, str, str]] = []
        for idx, row in enumerate(reader, start=2):
            lvl = pick(row, level_column) or "UNK"
            txt = pick(row, text_column)
            eng = pick(row, english_column)
            if not txt or not eng:
                raise ValueError(
                    f"Bad CSV row {idx}: expected non-empty '{text_column}' + '{english_column}'. Got: {row}"
                )
            out.append((lvl, txt, eng))
        return out, text_column

    raise ValueError(f"Unsupported input type: {path} (expected .json or .csv)")


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def die(msg: str, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


# ----------------------------
# TTS Backends
# ----------------------------


@dataclass
class TTSConfig:
    backend: str  # "openai" or "piper"
    audio_format: str  # "mp3" or "wav"
    # OpenAI
    openai_model: str
    openai_voice: str
    openai_api_key_env: str
    openai_base_url: Optional[str]
    # Piper
    piper_bin: str
    piper_model: Optional[Path]
    piper_speaker: Optional[int]
    # Common
    speed: float
    instructions: Optional[str]
    retries: int
    retry_sleep: float


def _get_openai_client(cfg: TTSConfig):
    """Create an OpenAI client once for reuse across all TTS calls."""
    api_key = os.getenv(cfg.openai_api_key_env)
    if not api_key:
        die(f"Missing API key. Set {cfg.openai_api_key_env} in your environment.")

    try:
        from openai import OpenAI
    except ImportError:
        die("Missing dependency: openai. Install with: pip install openai")

    client_kwargs = {}
    if cfg.openai_base_url:
        client_kwargs["base_url"] = cfg.openai_base_url

    return OpenAI(api_key=api_key, **client_kwargs)


def synthesize_openai(text: str, out_path: Path, cfg: TTSConfig, client) -> None:
    """Uses OpenAI Audio speech endpoint with a pre-built client."""
    kwargs = dict(
        model=cfg.openai_model,
        voice=cfg.openai_voice,
        input=text,
        response_format=cfg.audio_format,
        speed=cfg.speed,
    )
    if cfg.instructions:
        kwargs["instructions"] = cfg.instructions
    resp = client.audio.speech.create(**kwargs)
    out_path.write_bytes(resp.read())


def synthesize_piper(text: str, out_path: Path, cfg: TTSConfig) -> None:
    """
    Uses Piper CLI:
      echo "..." | piper -m model.onnx --output_file out.wav
    """
    if cfg.audio_format != "wav":
        die("Piper backend outputs WAV. Use --audio-format wav for --tts piper.")

    if not cfg.piper_model:
        die("Piper requires --piper-model /path/to/model.onnx")

    piper_exe = shutil.which(cfg.piper_bin) or cfg.piper_bin
    if not shutil.which(piper_exe) and not Path(piper_exe).exists():
        die(
            f"Could not find Piper executable '{cfg.piper_bin}'. "
            "Install piper-tts or ensure 'piper' is in PATH."
        )

    cmd = [piper_exe, "-m", str(cfg.piper_model), "--output_file", str(out_path)]
    if cfg.piper_speaker is not None:
        cmd += ["--speaker", str(cfg.piper_speaker)]

    proc = subprocess.run(
        cmd,
        input=text.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Piper failed (exit {proc.returncode}). stderr:\n{proc.stderr.decode('utf-8', errors='replace')}"
        )


def synthesize(text: str, out_path: Path, cfg: TTSConfig, client=None) -> None:
    for attempt in range(1, cfg.retries + 1):
        try:
            if cfg.backend == "openai":
                synthesize_openai(text, out_path, cfg, client)
            elif cfg.backend == "piper":
                synthesize_piper(text, out_path, cfg)
            else:
                die(f"Unknown backend: {cfg.backend}")
            return
        except Exception:
            if attempt < cfg.retries:
                time.sleep(cfg.retry_sleep)
            else:
                raise


# ----------------------------
# Anki deck building
# ----------------------------


def build_model(
    model_id: int,
    model_name: str,
    *,
    text_field_name: str = "Text",
    english_field_name: str = "English",
    reverse: bool = False,
) -> genanki.Model:
    """Create an Anki model. Field names are configurable."""
    templates = [
        {
            "name": "Recognition",
            "qfmt": f"{{{{{text_field_name}}}}}<br>{{{{Audio}}}}<div style='font-size:12px;opacity:0.7'>{{{{Level}}}}</div>",
            "afmt": f"{{{{FrontSide}}}}<hr>{{{{{english_field_name}}}}}",
        }
    ]
    if reverse:
        templates.append({
            "name": "Recall",
            "qfmt": f"{{{{{english_field_name}}}}}<div style='font-size:12px;opacity:0.7'>{{{{Level}}}}</div>",
            "afmt": f"{{{{FrontSide}}}}<hr>{{{{{text_field_name}}}}}<br>{{{{Audio}}}}",
        })
    return genanki.Model(
        model_id,
        model_name,
        fields=[
            {"name": "Level"},
            {"name": text_field_name},
            {"name": english_field_name},
            {"name": "Audio"},
        ],
        templates=templates,
    )


def main():
    p = argparse.ArgumentParser(
        description="Generate an Anki deck with TTS audio (OpenAI or Piper) for any language corpus."
    )

    # Input/Output
    p.add_argument(
        "--sentences", type=Path, default=Path("sentences.json"),
        help="Path to corpus (.json or .csv).",
    )

    # Corpus column mapping (CSV)
    p.add_argument("--text-column", type=str, default=None, help="CSV column for target-language text.")
    p.add_argument("--english-column", type=str, default=None, help="CSV column for English translation.")
    p.add_argument("--level-column", type=str, default="Level", help="CSV column for level tag (default: Level).")

    # Anki field labels
    p.add_argument("--anki-text-field", type=str, default="Text", help="Anki model field name for target text.")
    p.add_argument("--anki-english-field", type=str, default="English", help="Anki model field name for English.")

    p.add_argument("--media-dir", type=Path, default=Path("media"), help="Directory for generated audio files.")
    p.add_argument("--output", type=Path, default=Path("deck_WITH_AUDIO.apkg"), help="Output .apkg path.")

    # Card directions
    p.add_argument("--reverse", action="store_true", help="Add reverse cards (English → target language).")

    # Deck config
    p.add_argument("--deck-name", type=str, default="Language Sentences (Audio)")
    p.add_argument("--deck-id", type=int, default=2059400111)
    p.add_argument("--model-name", type=str, default="Language Sentences Model (Audio)")
    p.add_argument("--model-id", type=int, default=1607392320)

    # Processing
    p.add_argument("--limit", type=int, default=0, help="Only process first N sentences (0 = all).")
    p.add_argument("--start-index", type=int, default=1, help="Start numbering audio files at this index.")
    p.add_argument("--audio-prefix", type=str, default=None, help="Prefix for audio filenames (default: derived from text column name, e.g. 'spanish').")
    p.add_argument("--overwrite-audio", action="store_true", help="Regenerate audio even if file exists.")
    p.add_argument("--overwrite-apkg", action="store_true", help="Overwrite output .apkg if it exists.")
    p.add_argument("--dry-run", action="store_true", help="Validate without generating anything.")
    p.add_argument("--workers", type=int, default=4, help="Concurrent TTS requests (default: 4, ignored for piper).")

    # TTS backend selection
    p.add_argument("--tts", choices=["openai", "piper"], default="openai", help="TTS backend.")
    p.add_argument("--audio-format", choices=["mp3", "wav"], default=None, help="Audio format (default: mp3/openai, wav/piper).")
    p.add_argument("--speed", type=float, default=0.9, help="TTS playback speed (default: 0.9). OpenAI supports 0.25-4.0.")
    p.add_argument("--instructions", type=str, default=None, help="System prompt for gpt-4o-mini-tts (e.g. 'Speak clearly in modern Greek with standard Athenian pronunciation').")

    # OpenAI options
    p.add_argument("--openai-model", type=str, default="gpt-4o-mini-tts")
    p.add_argument("--openai-voice", type=str, default="alloy")
    p.add_argument("--openai-api-key-env", type=str, default="OPENAI_API_KEY")
    p.add_argument("--openai-base-url", type=str, default=None)

    # Piper options
    p.add_argument("--piper-bin", type=str, default="piper")
    p.add_argument("--piper-model", type=Path, default=None, help="Path to Piper .onnx voice model.")
    p.add_argument("--piper-speaker", type=int, default=None)

    # Robustness
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--retry-sleep", type=float, default=1.5)

    args = p.parse_args()

    sentences_path = args.sentences.expanduser().resolve()
    media_dir = args.media_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if output_path.exists() and not args.overwrite_apkg and not args.dry_run:
        die(f"Output exists: {output_path} (use --overwrite-apkg to overwrite)")

    sentences, resolved_text_column = load_sentences(
        sentences_path,
        text_column=args.text_column,
        english_column=args.english_column,
        level_column=args.level_column,
    )
    if args.limit and args.limit > 0:
        sentences = sentences[: args.limit]
    if not sentences:
        die("No sentences to process.")

    audio_prefix = args.audio_prefix
    if audio_prefix is None:
        if resolved_text_column:
            audio_prefix = resolved_text_column.lower().replace(" ", "_")
        else:
            audio_prefix = "tts"

    safe_mkdir(media_dir)

    audio_format = args.audio_format or ("mp3" if args.tts == "openai" else "wav")

    cfg = TTSConfig(
        backend=args.tts,
        audio_format=audio_format,
        openai_model=args.openai_model,
        openai_voice=args.openai_voice,
        openai_api_key_env=args.openai_api_key_env,
        openai_base_url=args.openai_base_url,
        piper_bin=args.piper_bin,
        piper_model=args.piper_model.expanduser().resolve() if args.piper_model else None,
        piper_speaker=args.piper_speaker,
        speed=args.speed,
        instructions=args.instructions,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
    )

    # Upfront validation for Piper
    if cfg.backend == "piper":
        if cfg.audio_format != "wav":
            die("Piper backend outputs WAV. Use --audio-format wav for --tts piper.")
        if not cfg.piper_model:
            die("Piper requires --piper-model /path/to/model.onnx")
        piper_exe = shutil.which(cfg.piper_bin) or cfg.piper_bin
        if not shutil.which(piper_exe) and not Path(piper_exe).exists():
            die(f"Could not find Piper executable '{cfg.piper_bin}'. Install piper-tts or ensure 'piper' is in PATH.")

    model = build_model(
        args.model_id,
        args.model_name,
        text_field_name=args.anki_text_field,
        english_field_name=args.anki_english_field,
        reverse=args.reverse,
    )
    deck = genanki.Deck(args.deck_id, args.deck_name)

    # Build work items
    work_items = []
    for n, (level, text, english) in enumerate(sentences):
        i = args.start_index + n
        filename = f"{audio_prefix}_{i}.{audio_format}"
        filepath = media_dir / filename
        work_items.append((i, level, text, english, filename, filepath))

    if args.dry_run:
        total_chars = sum(len(text) for _, _, text, _, _, _ in work_items)
        already_exist = sum(1 for _, _, _, _, _, fp in work_items if fp.exists())
        print("Dry run OK.")
        print(f"- Sentences: {len(sentences)}")
        print(f"- Total characters: {total_chars:,}")
        print(f"- Audio files already exist: {already_exist}")
        print(f"- TTS: {cfg.backend} ({cfg.audio_format}), speed={cfg.speed}")
        if cfg.instructions:
            print(f"- Instructions: {cfg.instructions}")
        print(f"- Media dir: {media_dir}")
        print(f"- Output: {output_path}")
        return

    # Create OpenAI client once (reused across all threads)
    client = _get_openai_client(cfg) if cfg.backend == "openai" else None

    # --- Audio generation phase (concurrent for OpenAI, sequential for Piper) ---
    to_generate = [
        (i, text, filepath)
        for i, _lvl, text, _eng, _fname, filepath in work_items
        if args.overwrite_audio or not filepath.exists()
    ]
    skipped = len(work_items) - len(to_generate)

    if to_generate:
        max_workers = max(1, args.workers) if cfg.backend == "openai" else 1

        def _gen(item):
            idx, txt, fpath = item
            try:
                synthesize(txt, fpath, cfg, client=client)
            except Exception as e:
                raise RuntimeError(f"Failed at item #{idx}: {txt!r}\n{e}") from e

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_gen, item): item for item in to_generate}
            try:
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"TTS={cfg.backend}"):
                    fut.result()
            except Exception:
                for f in futures:
                    f.cancel()
                raise

    # --- Note creation phase ---
    media_files: List[str] = []
    for i, level, text, english, filename, filepath in work_items:
        if not filepath.exists():
            raise RuntimeError(f"Audio file missing after synthesis: {filepath}")
        media_files.append(str(filepath))
        note = genanki.Note(
            model=model,
            fields=[level, text, english, f"[sound:{filename}]"],
            tags=[level],
        )
        deck.add_note(note)

    package = genanki.Package(deck)
    package.media_files = media_files
    package.write_to_file(str(output_path))

    print(f"Done: {output_path}")
    print(f"Cards: {len(sentences)} | Generated: {len(to_generate)} | Skipped: {skipped} | Backend: {cfg.backend}")


if __name__ == "__main__":
    main()

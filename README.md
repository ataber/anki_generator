# Anki Audio Deck Generator

Generate Anki flashcard decks with TTS audio for language learning. Supports Greek, Korean, and any language with an available TTS voice.

## Features

- **Dual TTS backends**: OpenAI (`gpt-4o-mini-tts`) or local Piper
- **Flexible input**: JSON or CSV corpora with auto-detected columns
- **Concurrent generation**: Parallel OpenAI requests (configurable worker count)
- **Resumable**: Skips existing audio files on re-run
- **Customizable speech**: Speed control, voice selection, and system instructions for pronunciation/accent guidance

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For OpenAI TTS, set your API key:

```bash
export OPENAI_API_KEY=sk-...
```

## Quick Start

Generate a deck from the included Greek corpus:

```bash
python make_anki_audio_deck_new.py \
  --sentences sentences.json \
  --output greek_deck.apkg
```

Preview what would happen without generating anything:

```bash
python make_anki_audio_deck_new.py --dry-run
```

## Input Formats

**JSON** — list of `[text, english]` or `[level, text, english]` pairs:

```json
[
  ["Είμαι εδώ.", "I am here."],
  ["Είμαι καλά.", "I am well."]
]
```

**CSV** — header row with auto-detected columns (Greek/Korean/Spanish/Text + English/Translation):

```csv
Level,Greek,Natural English Translation
A1,Είμαι εδώ.,I am here.
```

Column names are auto-detected but can be overridden with `--text-column`, `--english-column`, and `--level-column`.

## Usage

### OpenAI TTS (default)

```bash
python make_anki_audio_deck_new.py \
  --sentences modern_greek_corpus_1000_master.csv \
  --tts openai \
  --openai-voice alloy \
  --speed 0.9 \
  --instructions "Speak clearly in modern Greek with standard Athenian pronunciation" \
  --workers 4 \
  --output greek_1000.apkg
```

### Piper TTS (local, offline)

```bash
python make_anki_audio_deck_new.py \
  --sentences sentences.json \
  --tts piper \
  --piper-model piper/models/el_GR/el_GR-rapunzelina-medium.onnx \
  --audio-format wav \
  --output greek_local.apkg
```

### Key Options

| Flag                | Default                | Description                                |
| ------------------- | ---------------------- | ------------------------------------------ |
| `--sentences`       | `sentences.json`       | Input corpus (`.json` or `.csv`)           |
| `--output`          | `deck_WITH_AUDIO.apkg` | Output `.apkg` path                        |
| `--tts`             | `openai`               | TTS backend (`openai` or `piper`)          |
| `--speed`           | `0.9`                  | Playback speed (OpenAI: 0.25–4.0)          |
| `--instructions`    | —                      | System prompt for gpt-4o-mini-tts          |
| `--openai-voice`    | `alloy`                | OpenAI voice (alloy, nova, shimmer, etc.)  |
| `--workers`         | `4`                    | Concurrent TTS requests (OpenAI only)      |
| `--limit`           | `0` (all)              | Process only the first N sentences         |
| `--start-index`     | `1`                    | Starting index for audio filenames         |
| `--audio-prefix`    | `tts`                  | Prefix for audio filenames                 |
| `--overwrite-audio` | off                    | Regenerate audio even if files exist       |
| `--overwrite-apkg`  | off                    | Overwrite existing output file             |
| `--dry-run`         | off                    | Validate and show stats without generating |

## Included Data

| File                                      | Description                      |
| ----------------------------------------- | -------------------------------- |
| `sentences.json`                          | 500 Greek sentences (JSON pairs) |
| `modern_greek_corpus_1000_master.csv`     | 1000 Greek sentences with levels |
| `korean_high_intermediate_batch1_200.csv` | 200 Korean sentences             |
| `piper/models/el_GR/`                     | Greek Piper voice model          |

# yt-transcribe

A CLI tool to download YouTube videos and transcribe them using OpenAI's Whisper API.

## Features

- 📥 Downloads YouTube videos using `yt-dlp`
- 🎵 Extracts and optimizes audio for Whisper (16kHz mono Opus at 32kbps)
- 🎙️ Transcribes using OpenAI's Whisper API
- 🗣️ Supports speaker diarization with `gpt-4o-transcribe-diarize`
- 📝 Multiple output formats: JSON, text, SRT, VTT, verbose JSON, diarized JSON
- ⚙️ Configurable VAD (Voice Activity Detection) settings

## Prerequisites

- [Node.js](https://nodejs.org/) 26.5.0 or newer
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [ffmpeg](https://ffmpeg.org/)
- OpenAI API key

## Setup

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

```bash
# Basic usage (defaults to gpt-4o-transcribe-diarize with diarized_json output)
node index.ts https://www.youtube.com/watch?v=abc123

# Use a different model
node index.ts https://www.youtube.com/watch?v=abc123 --model whisper-1

# Generate SRT subtitles
node index.ts https://www.youtube.com/watch?v=abc123 --model whisper-1 --format srt

# Plain text output
node index.ts https://www.youtube.com/watch?v=abc123 --model whisper-1 --format text

# Custom output path
node index.ts https://www.youtube.com/watch?v=abc123 -o transcript.json
```

## Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--model` | `-m` | Model to use | `gpt-4o-transcribe-diarize` |
| `--format` | `-f` | Response format | `diarized_json` |
| `--language` | `-l` | Language code (ISO-639-1) | auto-detect |
| `--temperature` | `-t` | Sampling temperature (0-1) | `0` |
| `--output` | `-o` | Output file path | `<video-title>.<ext>` |
| `--no-chunking` | | Disable auto chunking | `false` |
| `--vad-threshold` | | VAD sensitivity (0-1) | `0.5` |
| `--vad-silence` | | VAD silence duration (ms) | `200` |
| `--vad-prefix` | | VAD prefix padding (ms) | `300` |
| `--keep-audio` | | Keep extracted audio file | `false` |
| `--help` | `-h` | Show help message | |

## Available Models

| Model | Description |
|-------|-------------|
| `gpt-4o-transcribe` | Fast transcription, JSON only |
| `gpt-4o-mini-transcribe` | Smaller/faster, JSON only |
| `gpt-4o-transcribe-diarize` | With speaker diarization |
| `whisper-1` | Original Whisper V2, all formats |

## Response Formats

| Format | Description | Supported Models |
|--------|-------------|------------------|
| `json` | Basic JSON | All |
| `text` | Plain text | whisper-1, diarize |
| `srt` | SubRip subtitles | whisper-1 |
| `vtt` | WebVTT subtitles | whisper-1 |
| `verbose_json` | Detailed JSON with timestamps | whisper-1 |
| `diarized_json` | JSON with speaker annotations | diarize only |

## Example Output (diarized_json)

```json
{
  "task": "transcribe",
  "duration": 27.4,
  "text": "Speaker A: Hello everyone...",
  "segments": [
    {
      "type": "transcript.text.segment",
      "id": "seg_001",
      "start": 0.0,
      "end": 4.7,
      "text": "Hello everyone, welcome to the show.",
      "speaker": "A"
    }
  ]
}
```

## License

MIT

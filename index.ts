#!/usr/bin/env bun
/**
 * yt-transcribe - Download YouTube videos and transcribe with OpenAI Whisper
 *
 * Usage:
 *   bun run index.ts <youtube-url> [options]
 *
 * Examples:
 *   bun run index.ts https://www.youtube.com/watch?v=abc123
 *   bun run index.ts https://www.youtube.com/watch?v=abc123 --model whisper-1 --format text
 *   bun run index.ts https://www.youtube.com/watch?v=abc123 --no-chunking
 */

import { $ } from "bun";
import { parseArgs } from "util";
import { existsSync, unlinkSync, mkdirSync } from "fs";
import { join, basename, dirname } from "path";

// ============================================================================
// Constants
// ============================================================================

const TMP_DIR = join(dirname(import.meta.path), "tmp");

function ensureTmpDir(): void {
  if (!existsSync(TMP_DIR)) {
    mkdirSync(TMP_DIR, { recursive: true });
  }
}

// ============================================================================
// Types
// ============================================================================

type Model =
  | "gpt-4o-transcribe"
  | "gpt-4o-mini-transcribe"
  | "gpt-4o-mini-transcribe-2025-12-15"
  | "whisper-1"
  | "gpt-4o-transcribe-diarize";

type ResponseFormat =
  | "json"
  | "text"
  | "srt"
  | "verbose_json"
  | "vtt"
  | "diarized_json";

interface ChunkingStrategyVAD {
  type: "server_vad";
  prefix_padding_ms?: number;
  silence_duration_ms?: number;
  threshold?: number;
}

type ChunkingStrategy = "auto" | ChunkingStrategyVAD;

interface TranscriptionOptions {
  model: Model;
  response_format: ResponseFormat;
  chunking_strategy?: ChunkingStrategy;
  language?: string;
  temperature?: number;
}

interface TranscriptionResponse {
  task?: string;
  duration?: number;
  text: string;
  segments?: Array<{
    type?: string;
    id?: string;
    start: number;
    end: number;
    text: string;
    speaker?: string;
  }>;
  usage?: {
    type: string;
    seconds: number;
  };
}

// ============================================================================
// Configuration
// ============================================================================

const MODELS: Model[] = [
  "gpt-4o-transcribe",
  "gpt-4o-mini-transcribe",
  "gpt-4o-mini-transcribe-2025-12-15",
  "whisper-1",
  "gpt-4o-transcribe-diarize",
];

const RESPONSE_FORMATS: ResponseFormat[] = [
  "json",
  "text",
  "srt",
  "verbose_json",
  "vtt",
  "diarized_json",
];

const DEFAULT_MODEL: Model = "gpt-4o-transcribe-diarize";
const DEFAULT_FORMAT: ResponseFormat = "diarized_json";

// ============================================================================
// Helpers
// ============================================================================

function printHelp(): void {
  console.log(`
\x1b[1myt-transcribe\x1b[0m - Download YouTube videos and transcribe with OpenAI Whisper

\x1b[1mUSAGE:\x1b[0m
  bun run index.ts <youtube-url> [options]

\x1b[1mOPTIONS:\x1b[0m
  -m, --model <model>       Model to use for transcription (default: ${DEFAULT_MODEL})
                            Available: ${MODELS.join(", ")}
  
  -f, --format <format>     Response format (default: ${DEFAULT_FORMAT})
                            Available: ${RESPONSE_FORMATS.join(", ")}
  
  -l, --language <lang>     Language code in ISO-639-1 format (e.g., "en", "es")
  
  -t, --temperature <temp>  Sampling temperature 0-1 (default: 0)
  
  --no-chunking             Disable auto chunking (not recommended for long audio)
  
  --vad-threshold <val>     VAD sensitivity threshold 0-1 (default: 0.5)
  
  --vad-silence <ms>        VAD silence duration in ms (default: 200)
  
  --vad-prefix <ms>         VAD prefix padding in ms (default: 300)
  
  -o, --output <path>       Output file path (default: <video-title>.json)
  
  --keep-audio              Keep the extracted audio file after transcription
  
  -h, --help                Show this help message

\x1b[1mEXAMPLES:\x1b[0m
  # Basic usage with defaults (diarized transcription)
  bun run index.ts https://www.youtube.com/watch?v=abc123

  # Use whisper-1 with plain text output
  bun run index.ts https://www.youtube.com/watch?v=abc123 --model whisper-1 --format text

  # Generate SRT subtitles
  bun run index.ts https://www.youtube.com/watch?v=abc123 --model whisper-1 --format srt

  # Custom VAD settings for noisy audio
  bun run index.ts https://www.youtube.com/watch?v=abc123 --vad-threshold 0.7

\x1b[1mENVIRONMENT:\x1b[0m
  OPENAI_API_KEY    Required. Your OpenAI API key.
`);
}

function log(emoji: string, message: string): void {
  console.log(`${emoji}  ${message}`);
}

function error(message: string, exitCode = 1): never {
  console.error(`\x1b[31m❌ Error:\x1b[0m ${message}`);
  process.exit(exitCode);
}

// ============================================================================
// Core Functions
// ============================================================================

async function downloadVideo(url: string, outputDir: string): Promise<string> {
  log("📥", "Downloading video from YouTube...");

  // Get video title first
  const titleResult =
    await $`yt-dlp --print filename -o "%(title)s" ${url}`.text();
  const title = titleResult.trim().replace(/[/\\?%*:|"<>]/g, "_");

  const outputPath = join(outputDir, `${title}.webm`);

  // Download the video
  await $`yt-dlp -o ${outputPath} ${url}`.quiet();

  if (!existsSync(outputPath)) {
    error(`Failed to download video to ${outputPath}`);
  }

  log("✅", `Downloaded: ${basename(outputPath)}`);
  return outputPath;
}

async function extractAudio(videoPath: string): Promise<string> {
  log("🎵", "Extracting and optimizing audio for Whisper...");

  // Use .webm container with opus audio (OpenAI API supports webm but not raw opus)
  const audioPath = videoPath.replace(/\.[^.]+$/, "_whisper.webm");

  // Extract audio optimized for Whisper:
  // - Mono (Whisper doesn't need stereo)
  // - 16kHz sample rate (Whisper's native rate)
  // - Opus codec at 32kbps (excellent for speech)
  // - VoIP application mode (optimized for voice)
  // - WebM container (supported by OpenAI API)
  await $`ffmpeg -y -i ${videoPath} -vn -ac 1 -ar 16000 -c:a libopus -b:a 32k -application voip -f webm ${audioPath}`
    .quiet();

  if (!existsSync(audioPath)) {
    error(`Failed to extract audio to ${audioPath}`);
  }

  const stats = Bun.file(audioPath);
  const sizeMB = ((await stats.size) / 1024 / 1024).toFixed(2);
  log("✅", `Extracted audio: ${basename(audioPath)} (${sizeMB} MB)`);

  return audioPath;
}

async function transcribe(
  audioPath: string,
  options: TranscriptionOptions
): Promise<TranscriptionResponse> {
  log("🎙️", `Transcribing with ${options.model}...`);

  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    error("OPENAI_API_KEY environment variable is required");
  }

  const formData = new FormData();

  // Add the audio file
  const audioFile = Bun.file(audioPath);
  formData.append("file", audioFile, basename(audioPath));

  // Add required parameters
  formData.append("model", options.model);
  formData.append("response_format", options.response_format);

  // Add optional parameters
  if (options.chunking_strategy) {
    if (typeof options.chunking_strategy === "string") {
      formData.append("chunking_strategy", options.chunking_strategy);
    } else {
      // For object-based chunking strategy, we need to send as JSON
      formData.append(
        "chunking_strategy",
        JSON.stringify(options.chunking_strategy)
      );
    }
  }

  if (options.language) {
    formData.append("language", options.language);
  }

  if (options.temperature !== undefined) {
    formData.append("temperature", options.temperature.toString());
  }

  const response = await fetch(
    "https://api.openai.com/v1/audio/transcriptions",
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
      },
      body: formData,
    }
  );

  if (!response.ok) {
    const errorText = await response.text();
    error(`OpenAI API error (${response.status}): ${errorText}`);
  }

  // Handle different response formats
  const contentType = response.headers.get("content-type") || "";

  if (
    options.response_format === "text" ||
    options.response_format === "srt" ||
    options.response_format === "vtt"
  ) {
    const text = await response.text();
    return { text };
  }

  const result = (await response.json()) as TranscriptionResponse;
  log("✅", "Transcription complete!");

  return result;
}

function getOutputExtension(format: ResponseFormat): string {
  switch (format) {
    case "srt":
      return ".srt";
    case "vtt":
      return ".vtt";
    case "text":
      return ".txt";
    default:
      return ".json";
  }
}

async function saveOutput(
  result: TranscriptionResponse,
  outputPath: string,
  format: ResponseFormat
): Promise<void> {
  let content: string;

  if (format === "text" || format === "srt" || format === "vtt") {
    content = result.text;
  } else {
    content = JSON.stringify(result, null, 2);
  }

  await Bun.write(outputPath, content);
  log("💾", `Saved transcription to: ${outputPath}`);
}

// ============================================================================
// Main
// ============================================================================

async function main(): Promise<void> {
  const { values, positionals } = parseArgs({
    args: Bun.argv.slice(2),
    options: {
      model: { type: "string", short: "m", default: DEFAULT_MODEL },
      format: { type: "string", short: "f", default: DEFAULT_FORMAT },
      language: { type: "string", short: "l" },
      temperature: { type: "string", short: "t" },
      "no-chunking": { type: "boolean", default: false },
      "vad-threshold": { type: "string" },
      "vad-silence": { type: "string" },
      "vad-prefix": { type: "string" },
      output: { type: "string", short: "o" },
      "keep-audio": { type: "boolean", default: false },
      help: { type: "boolean", short: "h", default: false },
    },
    allowPositionals: true,
  });

  if (values.help || positionals.length === 0) {
    printHelp();
    process.exit(values.help ? 0 : 1);
  }

  const youtubeUrl = positionals[0];

  // Validate model
  const model = values.model as Model;
  if (!MODELS.includes(model)) {
    error(`Invalid model: ${model}. Available: ${MODELS.join(", ")}`);
  }

  // Validate response format
  const format = values.format as ResponseFormat;
  if (!RESPONSE_FORMATS.includes(format)) {
    error(
      `Invalid format: ${format}. Available: ${RESPONSE_FORMATS.join(", ")}`
    );
  }

  // Validate format compatibility with model
  if (
    model === "gpt-4o-transcribe-diarize" &&
    !["json", "text", "diarized_json"].includes(format)
  ) {
    error(
      `Model ${model} only supports formats: json, text, diarized_json. Got: ${format}`
    );
  }

  if (
    (model === "gpt-4o-transcribe" || model === "gpt-4o-mini-transcribe") &&
    format !== "json"
  ) {
    error(`Model ${model} only supports format: json. Got: ${format}`);
  }

  // Build chunking strategy
  let chunkingStrategy: ChunkingStrategy | undefined;

  if (!values["no-chunking"]) {
    if (
      values["vad-threshold"] ||
      values["vad-silence"] ||
      values["vad-prefix"]
    ) {
      // Custom VAD settings
      const vadStrategy: ChunkingStrategyVAD = { type: "server_vad" };

      if (values["vad-threshold"]) {
        vadStrategy.threshold = parseFloat(values["vad-threshold"]);
      }
      if (values["vad-silence"]) {
        vadStrategy.silence_duration_ms = parseInt(values["vad-silence"]);
      }
      if (values["vad-prefix"]) {
        vadStrategy.prefix_padding_ms = parseInt(values["vad-prefix"]);
      }

      chunkingStrategy = vadStrategy;
    } else {
      // Default to auto chunking
      chunkingStrategy = "auto";
    }
  }

  // Build options
  const options: TranscriptionOptions = {
    model,
    response_format: format,
    chunking_strategy: chunkingStrategy,
    language: values.language,
    temperature: values.temperature
      ? parseFloat(values.temperature)
      : undefined,
  };

  console.log("\n\x1b[1m🎬 YouTube Transcriber\x1b[0m\n");
  console.log(`   URL:    ${youtubeUrl}`);
  console.log(`   Model:  ${model}`);
  console.log(`   Format: ${format}`);
  console.log(
    `   Chunking: ${chunkingStrategy ? (typeof chunkingStrategy === "string" ? chunkingStrategy : "custom VAD") : "disabled"}`
  );
  console.log();

  // Ensure tmp directory exists for temporary files
  ensureTmpDir();

  try {
    // Step 1: Download video to tmp directory
    const videoPath = await downloadVideo(youtubeUrl, TMP_DIR);

    // Step 2: Extract audio to tmp directory
    const audioPath = await extractAudio(videoPath);

    // Step 3: Transcribe
    const result = await transcribe(audioPath, options);

    // Step 4: Save output
    const outputExt = getOutputExtension(format);
    const defaultOutputPath = videoPath.replace(/\.[^.]+$/, outputExt);
    const outputPath = values.output || defaultOutputPath;

    await saveOutput(result, outputPath, format);

    // Cleanup
    if (!values["keep-audio"]) {
      unlinkSync(audioPath);
      log("🧹", "Cleaned up temporary audio file");
    }

    // Remove the downloaded video (we only needed the audio)
    unlinkSync(videoPath);
    log("🧹", "Cleaned up downloaded video");

    console.log("\n\x1b[32m✨ Done!\x1b[0m\n");

    // Print summary for diarized transcriptions
    if (format === "diarized_json" && result.segments) {
      console.log("\x1b[1mTranscription Preview:\x1b[0m\n");
      const previewSegments = result.segments.slice(0, 5);
      for (const segment of previewSegments) {
        const speaker = segment.speaker || "Unknown";
        const time = `[${segment.start.toFixed(1)}s - ${segment.end.toFixed(1)}s]`;
        console.log(`  \x1b[36m${speaker}\x1b[0m ${time}`);
        console.log(`  ${segment.text}\n`);
      }
      if (result.segments.length > 5) {
        console.log(
          `  ... and ${result.segments.length - 5} more segments\n`
        );
      }
    }
  } catch (err) {
    if (err instanceof Error) {
      error(err.message);
    }
    throw err;
  }
}

main();
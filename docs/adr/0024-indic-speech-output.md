# 0024: English and Hindi speech output with Magpie TTS on the CPU

Date: 2026-11-17
Status: Accepted (built; `phonon` stays the default until the user switches `[tts]`). Replaces the first version of this record, which chose maya2-native.

## Context

Lucy's voice is `phonon` on Ollama (ADR 0015), which speaks English only. The user speaks English, Hindi and the two mixed (ADR 0023) and wants one female voice across English and Hindi (Bengali later).

### First choice, withdrawn: maya2-native

`aruntemme/maya2-native-mlx-8bit` through mlx-audio 0.6.5 was chosen first after a listening test (`checks/tts_listen`, 2026-11-16 and 17) and built as the `maya2` adapter (worker process, streaming, voice Ahana). In the check it streamed its first audio in 0.19 s and read numbers, dates and rupee amounts in either script. In real sessions it lagged, and it was withdrawn from the product the same day:

- Memory: about 3.6 GB of MLX memory (4.7 GB resident with its worker) on a 24 GB machine already holding the chat model (16 GB in Ollama) and the recogniser.
- GPU contention: it runs on the same GPU as the chat model. In the lag diagnosis of real sessions the first sound came 1.2 to 2.6 s after the sentence's text was ready, and playback had 0.3 to 1.5 s gaps while the chat model was still generating; its real-time factor rose from 0.41 idle to 0.82 to 1.43 while the chat model generated, and the chat model fell from 50 to 29 tokens/s.

### Alternatives (checks/README.md, `tts_listen`)

| Option | Result |
|---|---|
| `ARTPARK-IISc/DhVaani-0.5` | measured; whole clip only, clones a reference voice, CC-BY-NC (personal use only), gated |
| `Oomni/mithi-v1-flash` (Charvi) | measured; whole clip only |
| indic-speak | considered; not run |
| IndicF5 | researched, not run: flow matching with a reference clip, whole clip only, too slow for the CPU (inferred from its model card) |
| Indic Parler-TTS | researched, not run: about 0.9 B parameters, prompt-described voices, slow without CUDA (inferred) |
| Supertonic 3 (F1/F2) | measured (`light1`): fastest (real-time factor 0.17, 0.6 GB, CPU), reads Devanagari Hindi, but misreads digits in Hindi (1250 as "25", 7:30 as "6:30") and garbles short romanised replies; no streaming |
| Magpie TTS Multilingual 357M, Q8_0, NeMo-Speech.cpp 0.3.0 on the CPU (Sofia/Aria) | measured (`light1`): real-time factor 0.30 to 0.40 with 4 threads, about 1 GB, chat model 46 to 44 tokens/s while it synthesises, no underruns; with `hi-IN` reads Devanagari and digits (Sparrowhawk normaliser) but spells Latin-script words letter by letter |
| Magpie on Metal | measured: slower load (20 s), competes for the GPU (chat model 42 to 31 tokens/s) |

## Decision

- Lucy's Hindi-capable voice is **Magpie TTS on the CPU**: the `magpie` adapter (`lucy.adapters.tts_magpie`), selected with `[tts] adapter = "magpie"`. `phonon` stays the default in code and in `lucy.example.toml`. The `maya2` adapter, its worker and its tests are removed; `checks/tts_listen` keeps its `maya` engines for comparison.
- **Streaming through the C API.** The runtime's HTTP route (`nemo-speech serve`, `/v1/audio/speech`) buffers the whole clip (its docs say so; first sound 1.5 to 1.8 s for a sentence). The release's C library, `libnemo_speech_tts`, hands PCM to a callback as the codec decodes it (186 ms chunks). `magpie_worker.py`, shipped with the adapter, loads that library with ctypes (standard library only, run with Lucy's own Python as a child process, so a crash in native code cannot take Lucy down) and streams the chunks as JSON lines, like the maya2 worker did. Measured on the CPU, quiet machine, through the worker: first audio 0.13 to 0.18 s for a sentence after warm-up (0.5 to 0.6 s for the first sentence of each language, while its tokenizer loads), real-time factor about 0.25; through the adapter (resampling, with another agent's chat model loaded, load average 3) 0.27 to 0.37 s.
- Process handling is `lucy.adapters.worker_process` (shared with the `indic_flex` recogniser): own process group, stopped on `aclose` and at interpreter exit; restart once if the worker dies before any audio of a sentence; a cancel on early close, honoured at the next chunk.
- Files: runtime, model, codec, tokenizer and normalisation grammars under `data/tools/nemo-speech/` by default (each overridable), installed by hand from the release archive and Hugging Face (docs/voice.md); a missing file is named with where to get it.
- **Language per sentence:** `hi-IN` when the sentence contains Devanagari or is romanised Hindi (converted, below), otherwise `en-US` (`language` forces one). Audio is 22.05 kHz, resampled to 24 kHz with soxr. Voice default Sofia, validated against the model's voices (Aria, Jason, John, Leo, Sofia); seed 0 by default so a sentence sounds the same each time.
- **Reply script.** Magpie reads Hindi only in Devanagari, so the persona now has the chat model write Hindi replies in Devanagari, English loanwords and names included ("मीटिंग", "राहुल"); English replies stay English; the user's own Hindi still reaches the model romanised. The screen shows replies' Devanagari in Latin letters (`lucy.voice.translit.to_latin`, `[display] latin`, default on) in `lucy chat`, `lucy voice` and `lucy listen`; the stored conversation keeps what the model wrote, and speech gets the Devanagari.
- **Fallback for romanised replies** (added 2026-11-17, after the check below): the model still writes some Hindi replies in Latin letters, so the adapter converts a Latin-script sentence that `lucy.core.hinglish.latin_language` judges Hindi to Devanagari (`lucy.core.devanagari.romanised_to_devanagari`: a table of about 600 common romanised Hindi words with spelling variants, English loanwords through the exception list, English-looking words through the English rules, other words through romanised-Hindi rules) and sends it as `hi-IN`; a sentence too short to tell follows the previous sentence's language. `roman_hindi = false` turns it off. Measured in `checks/tts_listen/roman_fallback.py` (checks/README.md).
- **On-screen loanwords:** `lucy.core.loanwords` holds one table of English words Hindi speakers use (weekdays, months, about 125 common loanwords, a few names) with their Devanagari; the respelling reads it one way and `to_latin` the other, so कैलेंडर shows as "calendar" and फ्राइडे as "Friday" instead of "kaileindar" and "phraaide". Real Hindi words are shown as `to_latin` spells them.
- **Safety net for Latin words in a Hindi sentence:** the adapter respells them in Devanagari before synthesis (`lucy.core.devanagari`: a short exception list of common words, letter names for all-capital words, simple spelling rules otherwise); a sentence whose words are mostly Latin goes as `en-US` unchanged. Digits are left to Magpie's normaliser.

## Consequences

- About 1 GB of memory and 4 CPU threads for the voice, no GPU: the chat model keeps its speed (46 to 44 tokens/s in the check).
- First sound about 0.2 to 0.4 s after a sentence is ready, against 0.01 to 0.04 s for `phonon`; warm-up synthesises one phrase per language (a few seconds).
- The model samples; the fixed seed makes a sentence repeatable, but a small change of wording gives a different rendition.
- The rough respelling of leftover Latin words sounds approximately right at best ("Dario" becomes "डैरिऑ"); it only has to beat spelling them out. Names the model leaves in Latin are the likeliest misses.
- With `phonon` (English only) as the voice, Lucy replies in English: `TTSCaps` has a `languages` field (`magpie` declares English and Hindi, every other voice English only), and when the configured `[tts]` voice cannot speak Hindi the persona's reply-language paragraph is replaced by `prompts/language_en.md` (reply in English, and say briefly that the voice cannot speak Hindi when asked). Decided once at startup from the configured adapter class, for text chat too, so the cached prompt prefix stays the same within a session and a conversation continued between `lucy chat` and `lucy voice` gets one instruction; with no `[tts]` section the Hindi paragraph stays.
- Licences: NeMo-Speech.cpp is Apache-2.0; the Magpie model and the NanoCodec are under the NVIDIA Open Model License. NeMo-Speech.cpp's install script edits the shell profile and is not used.

## Not done

- Bengali: the recogniser covers English and Hindi only (ADR 0023); Magpie v2607 has no Bengali.
- Splitting a long first sentence at a clause: not needed while the voice streams.
- The persona alone does not make every Hindi reply Devanagari: on the quiet baseline (2026-11-17, `--reps 2`) 31 of 41 Hindi and mixed replies were, mostly first replies with no lookup whose question carried English words ("Japan ki capital kya hai?" answered "Japan ki capital Tokyo hai."). Two stronger wordings with a worked example raised it to 34 of 40 but made the model answer English questions in Hindi after a lookup (3 of 6 such turns in round 2), so the wording is unchanged and the code fallback above catches the rest.

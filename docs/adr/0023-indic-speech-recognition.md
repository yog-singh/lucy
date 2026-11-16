# 0023: English and Hindi speech recognition with indic-transcribe-flex

Date: 2026-11-17
Status: Accepted (built; Parakeet stays the default until the user switches `[stt]`)

## Context

Parakeet (ADR 0015) understands English only: on Hindi it returns nothing or romanised nonsense ("Mercivil India." for मैं सिविल इंजीनियर). The user speaks English, Hindi and the two mixed, and wants everything Lucy hears written in Latin script, which the chat model reads best.

Measured in `checks/stt_accuracy` (checks/README.md, 2026-11-16 and 17; Apple M5 Pro, 24 GB): the user's 73 labelled English clips and 150 public Hindi clips (Monsoon, other speakers on phones), plus 2 s of silence and of noise.

| Option | English WER | Hindi WER | Notes |
|---|---|---|---|
| Parakeet alone (today) | 9.7 % | none (no Hindi) | 0.06 s per utterance, 2.2 GB |
| Whisper large-v3-turbo, language chosen by the model | 12.6 % | 62.0 % | says "Thank you." on noise |
| Qwen3-ASR 0.6B, language chosen | 16.1 % | 21.2 % | |
| Parakeet for English plus `indic-transcribe-flex` for Hindi, routed by Qwen's language choice (estimated) | 12.6 % | 6.5 % | two or three models, about 7.5 GB; Parakeet's own text cannot route |
| `indic-transcribe-flex` alone, `flex_enhi_mixed_latin` | 16.5 % | 6.2 % (romanised, folded; CER 3.6 %) | 0.35 s per utterance, 2.3 GB GPU |

Designs considered: Parakeet with a router in front (best English, but a misrouted clip is wholly wrong, the router is another model and its language choice is unreliable on short clips); a single multilingual model (Whisper, Qwen: weak Hindi); `indic-transcribe-flex` (a 1.2B fine-tune of NVIDIA Canary 1B v2 for Indian languages and Indian-accented English) in one of its Latin configurations (forced English loops on Hindi; its own romanised mode writes numbers as words and drops punctuation). The user chose the single model in the measured configuration `flex_enhi_mixed_latin`, accepting the English cost.

## Decision

- A second STT adapter, `indic_flex` (`lucy.adapters.stt_indic_flex`), selectable with `[stt] adapter = "indic_flex"`. Parakeet stays the default in code and in `lucy.example.toml`; switching back is one line.
- Configuration as measured: `bodhan-ai/indic-transcribe-flex` at the reviewed revision `ecba9bac`, language identification restricted to English and Hindi, mode "mixed" (English in Latin, digits, punctuation), float16 on MPS, greedy, generation capped at min(512, 10 + 25 tokens per second of audio). Any Devanagari left is written in Latin by `lucy.voice.translit.to_latin` (pure Python). Then the junk guard (`lucy.adapters.stt_indic_flex_guard`): drop below -55 dBFS, at the token cap, above 10 + 25 tokens per second, below -1.0 mean token log-probability, or when the text is only "REDACTED"; a run of 1 to 6 words repeated 4 or more times is kept once. A dropped transcript is empty text, which the voice session treats as no speech, and is logged with its reason and raw text.
- PyTorch and Transformers stay out of Lucy's dependencies. The model runs in a worker process, `indic_flex_worker.py`, shipped in the package: a PEP 723 script with pinned torch 2.11.0, torchaudio 2.11.0, transformers 5.5.3 (and `exclude-newer` for the rest), started with `uv run --script`, so uv builds and caches its own environment. It loads the model once and answers JSON lines (base64 PCM in, text, language, token count, mean log-probability and timings out); whatever the libraries print goes to stderr and into Lucy's debug log.
- The adapter keeps Parakeet's port, capabilities and behaviour: same input checks, the under-0.1 s and -60 dBFS shortcut, lazy start, `warm_up()` (start, load, one transcription) and `prepare()` (a tiny transcription in the worker when the user starts speaking after 3 s idle, the same meaning as for Parakeet: bring the model back into fast memory). Requests are serialised, each has a timeout (20 s; a timed-out worker is killed and restarted on the next utterance), a worker that dies is restarted once and the request retried. The worker runs in its own process group, stopped on close and at interpreter exit. It gets a minimal environment; when the model is in the Hugging Face cache the offline variables are set and no token is passed.

## Consequences

- English accuracy is lower than Parakeet's: 16.5 % against 9.7 % word error on the user's clips, 38 % against 14 % on clips under 2 s. The labels lean towards Parakeet (they were written from its output), so part of the gap is label bias, but not all.
- Replies start later: about 0.35 s per utterance against 0.06 s, up to about 1 s on long ones; warm-up about 8 s (environment start and model load), longer the first time while uv builds the worker's environment (PyTorch is a large download).
- No name biasing: the model takes no prompt or context, so names (Lucy, the user's contacts) cannot be boosted.
- The guard has known holes: on a 1.1 s noise clip (TV) every configuration says "Thank you", which passes; its thresholds were tuned on the clips they were reported on.
- `Transcript.language` is the model's choice between en and hi, which is unreliable on English: it chose Hindi for 55 of the user's 73 English clips (the text is the same, since mixed mode writes English in Latin). Nothing in Lucy reads it today.
- Hindi is written in the transliterator's informal style ("ke lie", "mainne"), less idiomatic than a person's.
- Licence: the model is gated on Hugging Face under the Indic Open Model License; the user must accept it on the model page with their account and log in (`hf auth login` or `HF_TOKEN`) to download the 4.9 GB weights. The adapter reports this when the model is missing or access is refused. The terms were not reviewed in this change.
- Memory: about 2.3 GB on the GPU and a resident peak of about 7 GB while loading (memory-mapped float32 weights, reclaimed afterwards), on top of the chat model.
- A second Python environment on disk (uv's cache), started per run of `lucy voice`/`lucy listen`; uv must be installed. The model's own code (from the reviewed revision) runs in the worker.
- Hindi accuracy rests on other speakers' phone recordings; the user's own Hindi has not been labelled.

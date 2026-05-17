# Encoder Vulnerabilities Transferability

Small research scripts for generating noisy/adversarial audio, transcribing it with Whisper large-v3, answering audio questions with Qwen2-Audio, and evaluating results.

## Setup

```bash
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

If you need gated Hugging Face models, authenticate first:

```bash
huggingface-cli login
```

Download the Qwen model used by the local audio-answer and judge scripts:

```bash
python download_model.py
```

Expected input data lives under `ready/`, especially:

```text
ready/clean/                 # clean WAV files
ready/generated_transcript.csv
```

## Common Commands

Create random-noise WAV folders:

```bash
python add_random_noise.py --clean-dir ready/clean --output-root ready --snr -5 --seed 42
```

Transcribe audio folders with Whisper large-v3:

```bash
python transcribe_whisper_large_v3.py --folders clean random_-5db --device auto
```

Create PGD adversarial WAVs against Whisper:

```bash
python pgd_whisper_attack.py --clean-dir ready/clean --output-dir ready/pgd_clean_whisper_large_v3 --steps 200 --device auto
```

Answer questions from an audio folder with Qwen2-Audio:

```bash
python qwen2_audio_answer.py ready/clean --device auto
```

Measure WER for a transcript CSV:

```bash
python measure_wer.py ready/clean_whisper_large_v3.csv --output ready/clean_wer.csv
```

Judge answer correctness with a local LLM:

```bash
python LLM_judge.py ready/clean_qwen2_audio_answers.csv --output ready/clean_qwen2_audio_answers_judged.csv
```

Use `python <script>.py --help` to see all options.

# Laya playground

A website, four games, a benchmark and an agent skill for [Laya](https://github.com/NandhaKishorM/laya), the open-source decision model: typed questions in, real probabilities out, one forward pass, no generated text. It runs on your own machine.

The live site is at **[brainfunctioncollapse.com/laya](https://brainfunctioncollapse.com/laya)**. It has no model behind it and replays recorded runs. Clone this repository and the same pages run against the real model.

Laya is created by [Nandakishor M](https://github.com/NandhaKishorM) of Convai Innovations. This repository is not his and does not contain the model: it installs his package and downloads his open weights.

## Run it

```bash
git clone https://github.com/wdobry/laya-playground
cd laya-playground
uv venv --python 3.12 && uv pip install laya
.venv/bin/python server.py
```

Then open <http://127.0.0.1:8770>. The first start downloads 2.3 GB of open weights and takes about 90 seconds to load them. Apple silicon, NVIDIA or plain CPU.

You get the whole site, live:

- `/` the landing page, where the model plays Flappy, a lane runner, Tetris and Snake itself, about 30 decisions a second
- `/playground` the editor: write some text and a few typed questions, see every answer with its probabilities, compare all three checkpoints
- `/about` why this exists

The web server itself is the Python standard library and binds to loopback, so nothing is reachable from your network. The only dependency is `laya`.

### Without the model

Any static file server shows the recorded version, exactly as the public site does:

```bash
python3 -m http.server 8771 --bind 127.0.0.1
```

Then open <http://127.0.0.1:8771>. The games replay real recorded decisions and say so; the playground answers its presets from recordings.

## The agent skill

One file teaches a coding agent to add Laya to a project properly: the API, questions that work, thresholds, calibration, and the traps found building this site.

```bash
mkdir -p .claude/skills/laya-integration && curl -fsSL https://brainfunctioncollapse.com/laya/skills/laya-integration/SKILL.md -o .claude/skills/laya-integration/SKILL.md
```

Claude Code reads `.claude/skills` natively. Any agent that accepts a markdown instruction file can use [the file](skills/laya-integration/SKILL.md) as it is.

## The benchmark

Laya and TypeSafe AI's hosted Jev, on the same 500 labelled examples with the same questions and no tuning. Jev is more accurate out of the box. Laya matches it on simple questions, answers several times faster from a laptop, is free, and is yours to fine-tune. The numbers on the site are rendered from [`static/data/versus.json`](static/data/versus.json), never typed by hand.

To reproduce it:

```bash
.venv/bin/python eval/build_dataset.py                       # rebuilds the sampled texts from their public sources
.venv/bin/python server.py                                   # Laya, local: leave it running in another terminal
TYPESAFE_API_KEY=... .venv/bin/python eval/run_eval.py       # writes static/data/versus.json
```

Two of the source datasets restrict redistribution, so the sampled texts and the raw API responses are not in this repository. `eval/provenance.json` records where every example comes from. The harness only measures: nothing from Jev is ever fed into Laya.

## What is in here

| Path | What it is |
| --- | --- |
| `server.py`, `poc.py` | the local model server and the proof of concept it grew from |
| `index.html`, `about.html`, `playground.html` | the three pages |
| `static/` | styles, scripts, and the recorded data the public site replays |
| `static/demos/` | the four games: Flappy, a lane runner, Tetris and Snake. Each one describes its situation in a sentence and asks one typed question |
| `skills/laya-integration/SKILL.md` | the agent skill |
| `eval/` | the benchmark: dataset builder, tasks, runner |
| `tools/` | recorders for the replays, and small site checks |

Tools worth knowing:

- `tools/record_run.mjs` records a real model-driven game run (`ONLY=tetris` records a single game); `tools/verify_replay.mjs` checks a recording replays identically
- `tools/record_presets.py` records the playground's preset answers
- `tools/build_nav.py` stamps the one shared top bar into every page; `tools/build_faq.py` regenerates the FAQ structured data from the visible Q&As
- `tools/check_widows.mjs` fails if any text block ends on a single word, at three widths

## Credits

**Laya** is created by [Nandakishor M](https://github.com/NandhaKishorM) (Convai Innovations) and released under Apache-2.0: [code](https://github.com/NandhaKishorM/laya), [weights](https://huggingface.co/convaiinnovations/laya), [the original paper](https://arxiv.org/abs/2503.23303), [the follow-up](https://arxiv.org/abs/2510.01237). If Laya is useful to you, support its author.

This playground, the Laya vs Jev benchmark and the agent skill are by [brain function collapse](https://brainfunctioncollapse.com).

Not affiliated with or endorsed by TypeSafe AI. Jev is their product, named here only to compare.

## Licence

[MIT](LICENSE) for everything in this repository. Laya itself, its code and its weights, is Apache-2.0 and belongs to its author.

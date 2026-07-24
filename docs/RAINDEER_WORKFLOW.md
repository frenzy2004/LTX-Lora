# Raindeer Workflow

Raindeer is the repeatable LTX 2.3 character-LoRA loop that came out of the first three rounds. It keeps private sources, LoRA weights, provider URLs, and secrets outside Git, while publishing only approved generated proof clips.

## Pattern

1. Record or select one visually consistent character set.
2. Build a short portrait dataset with audio and neutral captions using the `orvo` trigger.
3. Train one LTX 2.3 T2V LoRA for the round.
4. Render the same three proof prompts: two T2V clips and one I2V clip.
5. Compare identity, texture, environment stability, hands, mouth motion, and audio.
6. Keep only generated proof videos that are explicitly approved for publication.
7. Use the result to decide the next dataset move.

## Round Evidence

| Round | Training | Dataset move | Proof clips | Reusable lesson |
|---|---:|---|---|---|
| 1 | 500 steps, about $3.00 | Single tutorial-style source set | [01 T2V](../results/videos/raindeer-round-1-01-t2v.mp4), [02 T2V](../results/videos/raindeer-round-1-02-t2v.mp4), [03 I2V](../results/videos/raindeer-round-1-03-i2v.mp4) | Best photorealistic look; useful reference style. |
| 2 | 500 steps, about $3.00 | New reference footage with more source variation | [01 T2V](../results/videos/raindeer-round-2-01-t2v.mp4), [02 T2V](../results/videos/raindeer-round-2-02-t2v.mp4), [03 I2V](../results/videos/raindeer-round-2-03-i2v.mp4) | Mixed setting/outfit data made the result read more synthetic. |
| 3 | 1000 steps, about $6.00 | Corridor-only subset, same outfit and lighting family | [01 T2V](../results/videos/raindeer-round-3-01-t2v.mp4), [02 T2V](../results/videos/raindeer-round-3-02-t2v.mp4), [03 I2V](../results/videos/raindeer-round-3-03-i2v.mp4) | Consistency matters more than dumping every available clip into training. |

The twin, burger, and other prompt-only side quests are intentionally not part of the three-round training evidence. They are useful render experiments, not training-round evidence. The published burger review proof is tracked separately as a render-only round-1 reuse artifact in [RAINDEER_VIDEOCLAW_BURGER_REVIEW.md](RAINDEER_VIDEOCLAW_BURGER_REVIEW.md).

## Ledger Model

Training is step-priced:

```text
training_cost = steps * $0.006
```

Quality rendering is megapixel-frame priced:

```text
generated_MP = width * height * frames / 1,000,000
render_cost = generated_MP * $0.0027075
```

For the proof renders, the billed class is 1280x720. A 121-frame proof clip is about 111.5136 MP, or about $0.3020. The historical local runner reserved $0.3000 per 5-second proof render; use the MP formula for future ledgers.

Approximate round costs using the MP formula:

| Round | Training | Three 121-frame renders | Total |
|---|---:|---:|---:|
| 1 | $3.0000 | $0.9060 | $3.9060 |
| 2 | $3.0000 | $0.9060 | $3.9060 |
| 3 | $6.0000 | $0.9060 | $6.9060 |

## CLI

Print the default plan:

```bash
python scripts/raindeer.py plan
```

Write a JSON plan:

```bash
python scripts/raindeer.py plan --output private_work/raindeer/plan.json
```

Create a proof manifest from already-approved generated videos:

```bash
python scripts/raindeer.py proof results/videos/raindeer-round-*.mp4 \
  --quality-status raindeer_round_proof \
  --output private_work/raindeer/proof-manifest.json
```

## Operating Rules

- Do not commit source videos, reference images, datasets, provider URLs, LoRA weights, or keys.
- Publish generated proof videos only under `results/videos` and only with exact manifest entries.
- Do not mix visually different rooms, outfits, lighting, or camera distances in one character round unless the goal is robustness testing.
- Prefer a 5-9 minute high-signal source recording for the next serious round: fixed camera, centered face, same outfit, same lighting, clean background, natural mouth movement, and small hand gestures.
- Use prompt-only props/actions for one-off tests. Retrain only when the identity/look fails or when a recurring prop/action must be learned reliably across many shots.

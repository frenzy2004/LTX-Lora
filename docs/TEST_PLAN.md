# Character LoRA pilot test plan

## Objective

Determine whether a private LTX 2.3 LoRA materially improves character consistency across locations and talking-head scenarios while keeping accepted-output cost below the current baseline.

## Dataset gate

- 20–40 clean video clips, each 4–8 seconds.
- Exactly one prominent person per clip.
- Multiple camera distances, head angles, expressions, lighting conditions, and backgrounds.
- No captions, watermarks, rapid edits, or heavy beauty filters.
- Five source clips held out from training.
- Video-only fal archive with matching `.txt` caption sidecars.

## Fixed location matrix

1. Neutral professional studio.
2. Home office.
3. Podcast studio.
4. Modern coffee shop.
5. Conference stage.
6. Library or bookshelf environment.
7. Outdoor urban location.
8. Abstract branded background.

Each location is tested as a close-up and medium shot. Initial previews use one seed; promising cases receive two additional seeds.

## Speech routes

1. LTX-native audiovisual generation for exploratory speech.
2. Character video plus supplied clean audio and a dedicated lip-sync stage for exact wording.

Native generation is not considered exact-script success unless every requested word is reproduced correctly.

## Scorecard

| Metric | Weight |
|---|---:|
| Identity fidelity | 30% |
| Motion and temporal coherence | 20% |
| Prompt and location adherence | 20% |
| Generalization beyond training backgrounds | 15% |
| Artifact rate | 10% |
| Speech/audio quality | 5% |

## Go criteria

- At least 75% of prompts yield an acceptable result within three paid attempts.
- No critical identity drift or background memorization.
- Talking-head movement is natural.
- Exact speech succeeds through the selected audio/lip-sync route.
- Provider spend remains at or below the authorized cap.
- Accepted-output cost is materially below the comparison baseline.

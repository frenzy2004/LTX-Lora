# Character LoRA pilot test plan

## Objective

Determine whether a private LTX 2.3 LoRA materially improves character consistency across locations and talking-head scenarios while keeping accepted-output cost below the current baseline.

## Dataset gate

The following gate covers the current I2V identity pilot. A video-only I2V archive is not sufficient for training a supplied-audio LTX A2V model.

- 20–40 clean video clips, each 4–8 seconds.
- Exactly one prominent person per clip.
- Multiple camera distances, head angles, expressions, lighting conditions, and backgrounds.
- No captions, watermarks, rapid edits, or heavy beauty filters.
- Five source clips held out from training.
- Video-only fal archive with matching `.txt` caption sidecars.
- Landscape containers with centered portrait content are cropped to true 9:16 before training.
- Sidecar captions describe clothing, pose, and environment; the configured trigger phrase is prepended by fal.

## LTX A2V training gate

A proper LTX A2V training experiment requires a new dataset rather than reuse of the current I2V smoke adapter:

- Synchronized close-up video and original clean speech with accurate audio/video alignment.
- One prominent, unobstructed speaker whose mouth, jaw, cheeks, eyes, and full expression remain visible.
- Natural variation in speech sounds and expressions without cuts, dubbed audio, background music, or overlapping speakers.
- Held-out synchronized clips for measuring visual lip-sync and identity independently from training fit.
- Separate authorization for any new paid run. The endpoint lists **$0.006 per step**: 1,000 steps cost $6 and the 2,000-step default costs $12, before external inference and retries.

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
2. Properly trained LTX A2V using synchronized close-up training data.
3. Character video plus supplied clean audio and a dedicated real-video lip-sync stage for exact wording.
4. Sync v3 real-video editing as a separate control and possible production candidate; it is not evidence for the current I2V LoRA/A2V route.

Native generation is not considered exact-script success unless every requested word is reproduced correctly.

Audio waveform preservation and visual lip-sync quality are evaluated separately. A returned audio track can match the supplied recording while the visible mouth, jaw, cheeks, eyes, expression, or facial texture still looks synthetic.

## Indistinguishability gate

Any clip that is obviously AI is rejected regardless of identity similarity, audio PSNR, or aggregate score. Hard-failure signs include deformed or unstable mouth shapes, jaw or cheek warping, unnatural eye or expression motion, and synthetic facial texture. A passing exact-speech result must preserve the requested audio **and** show natural speech-linked facial motion throughout the clip.

## Scorecard

| Metric | Weight |
|---|---:|
| Identity fidelity | 30% |
| Motion and temporal coherence | 20% |
| Prompt and location adherence | 20% |
| Generalization beyond training backgrounds | 15% |
| Artifact rate | 10% |
| Audio/script fidelity | 5% |

## Go criteria

- At least 75% of prompts yield an acceptable result within three paid attempts.
- No critical identity drift or background memorization.
- Talking-head movement and speech-linked mouth, jaw, cheek, eye, and expression motion are natural.
- No synthetic facial texture or other obvious-AI hard failure.
- Exact speech succeeds through the selected audio/lip-sync route.
- Audio/script fidelity passes separately from visual lip-sync quality.
- Provider spend remains at or below the authorized cap.
- Accepted-output cost is materially below the comparison baseline.

## Current gate status

The tested 500-step I2V LoRA plus supplied-audio A2V clip is rejected as obviously AI and does not qualify for production. Its returned audio closely preserves the supplied waveform, but the visual face deformation and synthetic texture fail the independent lip-sync and indistinguishability gates.

The preservation-first real-video control is materially stronger but remains unproven: frame review found the edit localized to the lower face with genuine source texture retained elsewhere, while brief teeth/inner-mouth smoothing could still reveal manipulation. It is **pending blinded review**, not a pass.

Successful completed operations total **$3.2137**; the conservative cap total is **$3.5409**, leaving **$8.4591** under the $12 cap.

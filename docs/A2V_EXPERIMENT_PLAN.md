# Approval-Gated LTX A2V Experiment Plan

**Status:** planned; no A2V training has been submitted
**Hard pilot cap:** $12 total committed spend
**Current conservative total:** $3.5409
**Current remaining budget:** $8.4591

## Why a new run is required

The completed 500-step adapter was trained with the I2V endpoint and `with_audio: false`. It is a valid character/I2V smoke candidate, but it never received synchronized examples linking speech audio to mouth, jaw, cheek, eye, and expression motion. Sending that adapter to an audio-to-video inference endpoint does not make it an A2V-trained LoRA.

The correct LTX mode for “generate video motion that matches this exact audio” is A2V: video is generated, while the supplied audio is frozen as conditioning. [Official LTX training modes](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-trainer/docs/training-modes.md), [fal A2V trainer](https://fal.ai/models/fal-ai/ltx23-trainer-v2/a2v)

## Dataset contract

Each accepted training group must contain:

```text
sample_001_start.png
sample_001_audio.wav
sample_001_end.mp4
sample_001.txt
```

- `_start.png`: the exact first frame of the final target crop.
- `_audio.wav`: speech extracted from the same target segment, with original timing.
- `_end.mp4`: the real synchronized speaking performance the model should learn.
- `.txt`: factual framing, location, motion, and speech description.

The trigger token is sent once as `trigger_phrase`. It is not baked into caption sidecars because the managed trainer prepends it.

### Stop-before-spend gate

Submit nothing unless at least 10 unique groups pass all checks:

- one unobstructed speaker;
- close or medium-close framing;
- mouth, jaw, cheeks, eyes, and expression visible;
- continuous speech-linked movement with no internal cut;
- at least 89 frames at the configured 24 fps;
- start image, audio, and target derived from the same segment;
- A/V starts aligned and duration difference no greater than one frame;
- no overlapping speaker, dubbing, music, watermark, burned captions, or beauty filter;
- five additional synchronized groups held out completely.

If the current source pool does not yield at least 10 passing groups, stop and collect purpose-recorded close talking-head clips. Do not train on distant/full-body clips merely to reach a count.

Current visual audit: the 74-file inventoried pool is dominated by full-body or medium activity shots and does **not yet visibly establish** ten qualifying close-speaking groups with teeth/inner-mouth coverage. The remaining download or new purpose-recorded footage must satisfy this gate before the $6 run can be approved for submission.

## Managed training request

| Field | Planned value |
|---|---:|
| Endpoint | `fal-ai/ltx23-trainer-v2/a2v` |
| Rank | 32 |
| Steps | 1,000 |
| Learning rate | 0.0002 |
| Frames / FPS | 89 / 24 |
| Resolution | high, 9:16 (`544×960`) |
| Auto-scale | false |
| Scene splitting | false |
| Audio normalization | true |
| Preserve pitch | true |
| Built-in validation | two fresh image+audio holdouts |
| Debug dataset | true |

fal lists A2V training at $0.006 per step. The documented default is 2,000 steps ($12); this plan uses 1,000 steps ($6) because the remaining cap must also cover evaluation. More steps are not assumed to fix quality. [fal A2V pricing and schema](https://fal.ai/models/fal-ai/ltx23-trainer-v2/a2v)

## Validation matrix

The provider reel is not sufficient for production approval. After training, use fresh images and matching unseen audio for:

| Category | Minimum evidence |
|---|---:|
| In-distribution framing | 2 clips |
| Unseen scripts | 2 clips |
| Held-out locations | 2 clips |
| Teeth/inner-mouth close view | included in at least 1 clip |

For each output, inspect the full timeline for:

- identity drift;
- synthetic skin or beard texture;
- unstable or anatomically implausible lip shapes;
- teeth/inner-mouth smoothing;
- jaw or cheek warp;
- eye/expression motion unrelated to speech;
- audio timing and exact-script fidelity.

Audio preservation and visible lip-sync are separate gates. A waveform match cannot compensate for an artificial face.

### Final release gate

Mix the candidate outputs with genuine clips, randomize them, and conduct a blinded native-speed review. The requested bar is met only if reviewers cannot reliably identify the A2V outputs as AI. No internal score, PSNR, SSIM, or provider validation reel can substitute for that test. LTX's own training guide recommends a blind real-versus-LoRA evaluation for this kind of objective. [Official LTX LoRA guide](https://ltx.io/blog/training-your-first-lora-on-ltx)

## Budget envelope

| Item | Maximum |
|---|---:|
| Current conservative committed total | $3.5409 |
| 1,000-step A2V training | $6.0000 |
| External A2V validation ceiling | $1.2500 |
| Required safety buffer | $1.2091 |
| **Hard total ceiling** | **$12.0000** |

Reservations are created before submission. Any request that would exceed the remaining cap is blocked locally.

## Approval

No dataset extraction, full captioning, paid A2V training, or new inference begins until the user explicitly approves this plan. Approval authorizes only the 1,000-step A2V run and the $1.25 external-validation ceiling; it does not authorize the extra $2 contingency or any cap increase.

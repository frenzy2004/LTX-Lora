# LTX A2V 1,000-Step Quality Rebuild Evaluation

**Run date:** 2026-07-19

**Decision:** **Rejected for the required “indistinguishable from genuine footage” standard.**
**Useful finding:** the rebuilt LoRA is a substantial improvement on a simple held-out scene, but it is not robust across locations.

This is a sanitized evidence report. It contains generated videos and aggregate technical facts only. It excludes raw source media, private hold-outs, adapter weights, participant identifiers, provider request identifiers, provider URLs, credentials, and local paths.

## What changed from the rejected 400-step route

This was not another reroll of the earlier configuration. The controlled rebuild changed the data, preprocessing gate, training duration, and evaluation design together:

| Item | Rebuild setting |
|---|---:|
| Train / hold-out groups | 13 / 5 |
| Hold-out separation | Source-asset and location separated |
| Speech captions | Explicit A/V speech tag plus factual description |
| Provider-decoded data gate | Inspected before full training |
| Training | 1,000 steps, rank 32, learning rate 0.00015 |
| Bucket | High 9:16, 544×960, 89 frames, 24 fps |
| Inference | LTX-2.3 Quality A2V LoRA, maximum quality, 30 steps |
| LoRA / first-image strength | 0.8 / 1.0 |
| Prompt expansion | Disabled |

The first 100-step provider debug archive decoded with colour inversion. A single controlled counter-test pre-inverted only the visual training inputs; Fal then decoded all 13 video/audio groups at normal colour and the expected bucket. Only after that gate passed was the 1,000-step candidate submitted.

## Generated evidence

Both published files are generated LTX output. They use held-out first images and held-out speech audio that were never included in training.

| Held-out scene | Generated output | Review |
|---|---|---|
| Plain wall | [MP4](videos/a2v-1000step-quality-heldout-wall.mp4) | Identity remains stable and the 89-frame mouth sequence has no gross deformation. This is promising, but it has not passed a blinded playback test and is not approved as indistinguishable. |
| Bright office | [MP4](videos/a2v-1000step-quality-heldout-office.mp4) | **Rejected.** A large invented text overlay appears, laptop text is garbled, and hand/motion detail softens. These are visible AI cues. |

Hashes and media metadata are recorded in the [public manifest](videos/a2v-1000step-quality-evaluation.manifest.json).

## Sound and timing checks

Both retrieved MP4s contain H.264 video and mono 48 kHz AAC audio. Each is 576×960, 89 frames at 24 fps, and 3.708333 seconds long.

The returned audio was compared against the supplied held-out speech after decoding both to mono PCM and aligning 10 ms energy envelopes:

| Scene | Audio-envelope correlation | Measured lag |
|---|---:|---:|
| Plain wall | 0.999986 | 0.00 s |
| Bright office | 0.999705 | 0.00 s |

This establishes that the correct speech recording is present and time-aligned as an audio stream. It does not, by itself, prove perfect phoneme-to-mouth synchronization.

## Visual assessment

The simple wall result is the strongest LTX-only output in this experiment. Identity, hair, beard, garment, framing, and broad mouth motion remain coherent across all 89 frames. Its weaker points are slight facial softening and the absence of an independent normal-speed blind test.

The office result is a hard failure for social-media realism. The model invents prominent text even though prompt expansion was disabled and text/subtitles were explicitly discouraged. It also reconstructs fine scene text poorly and softens a moving hand. A viewer does not need forensic inspection to identify those artifacts.

Therefore the result is not robust across even two held-out locations. One attractive clip cannot support a production claim or a per-customer hosting decision.

## Provider reliability finding

Two additional evaluations were submitted together. Both reported completed compute with no generation error, but Fal's result endpoint persistently returned HTTP 500 and supplied no retrievable video. An identical office evaluation submitted alone completed and downloaded normally.

The local evaluator now enforces one in-flight evaluation at a time. The failed request records are preserved privately and were not resubmitted as creative rerolls. No provider identifiers or URLs are published.

## Cost ledger

Fal currently lists A2V training at $0.006 per step and Quality A2V LoRA inference at $0.0027075 per generated megapixel-frame.

| Component | Known cost (USD) |
|---|---:|
| Original 100-step decoded-data check | 0.600000 |
| Controlled 100-step colour check | 0.600000 |
| 1,000-step candidate | 6.000000 |
| Two retrieved 576×960×89 Quality outputs | 0.266491 |
| **Current controlled sequence known subtotal** | **7.466491** |
| Historical accounted spend before this rebuild | 13.820900 |
| **Historical + current known subtotal** | **21.287391** |

Two requests completed compute but produced no retrievable result; their account-side billing is unknown and is intentionally excluded from the known subtotal. The provider account ledger remains authoritative.

## Decision

Do not deploy this adapter as an “indistinguishable talking head” product and do not run more seeds with the same configuration.

The experiment proves three narrower points:

1. Fal-managed LTX A2V can cheaply train and host a private per-customer LoRA.
2. A simple, reference-anchored scene can preserve identity and supplied audio well.
3. The current route is not robust to visually busier locations and therefore does not meet the required quality bar.

Any further LTX-only experiment must materially change the data: remove visible text/logos from training and validation scenes, use tighter face-dominant crops, and retain the single-flight provider guard. It must still pass a blinded real-versus-generated playback test before an indistinguishability claim.

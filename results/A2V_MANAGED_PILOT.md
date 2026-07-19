# Managed LTX A2V Pilot — Evidence and Decision

## Decision

**Do not approve this native LTX-only configuration for indistinguishable talking-head production or multi-location character generation.**

The managed A2V LoRA training job completed successfully, but the outputs did not meet the required quality bar:

- The trainer's own preview was visibly inverted/glitched.
- A prompt-only LoRA render produced the wrong person.
- A reference-anchored render retained a recognisable likeness, but showed exaggerated synthetic-looking facial motion and remained tied to the source setting.
- Reducing reference strength and prompting for a coworking office did not move the scene; the source setting remained.

This is real execution evidence, not a desk-research conclusion. There is no evidence that the tested pipeline would be indistinguishable from a real social-video talking head.

## Scope

The test used the managed LTX A2V trainer and managed LTX distilled Audio-to-Video-with-LoRA inference only. No external image-generation or lip-sync system was used. Raw inputs, LoRA weights, provider URLs, credentials, and source-media details remain private and are intentionally not included in this repository.

## Completed training

| Item | Value |
|---|---:|
| Training mode | LTX managed A2V LoRA |
| Training steps | 1,000 |
| Training result | Completed; LoRA artifact returned |
| Private LoRA artifact size | 428,150,680 bytes |
| Trainer preview | 544×1072, 24 fps, 37.08 s |
| Training compute estimate | $6.00 (`1,000 × $0.006/step`) |

The returned LoRA and configuration are retained privately; they are not committed to Git.

## Render evidence

All clips below are the direct native LTX outputs. They are included so that the result can be audited rather than described selectively.

| Clip | Native LTX setup | Observed outcome | Decision |
|---|---|---|---|
| [Trainer preview](videos/a2v-managed-trainer-preview.mp4) | Trainer-produced preview | Inverted/negative-looking colour and visible artifacts in sampled frames. | Reject |
| [Prompt-only LoRA control](videos/a2v-managed-pure-lora-control.mp4) | LoRA scale `0.8`; supplied audio; no reference image | Generated a different person. | Reject |
| [Reference-anchor control](videos/a2v-managed-reference-anchor-control.mp4) | LoRA scale `0.8`; reference strength `1.0`; supplied audio | Likeness is recognisable, but motion/expression is visibly synthetic and the reference environment is preserved. | Reject for the required standard |
| [Location-change trade-off](videos/a2v-managed-location-tradeoff-control.mp4) | LoRA scale `0.9`; reference strength `0.55`; coworking-office prompt; supplied audio | Identity remains reference-led, but the requested coworking setting does not appear; source setting persists. | Reject |

The three controlled inference clips are each 576×1024, 89 frames, 24 fps, and approximately 3.708 seconds long.

## Cost record

The official inference price at execution time was `$0.001405` per generated megapixel, rounded up. Each controlled render is:

```text
576 × 1024 × 89 = 52.494336 generated megapixels
53 billed megapixels × $0.001405 = approximately $0.074465 per render
```

| Component | Count | Pricing-model estimate |
|---|---:|---:|
| A2V LoRA training | 1 × 1,000 steps | $6.0000 |
| Controlled native inference | 3 | ~$0.2234 |
| Direct pilot compute total | — | ~$6.2234 |

The run used a conservative `$0.12` ceiling for each submitted inference request. No multi-location fan-out was submitted after the controls failed, avoiding unnecessary spend. Provider invoice totals remain the billing source of truth.

## Quality assessment

The strict acceptance condition was: a real-looking talking head that does not obviously appear AI-generated, retains the authorized subject's identity, speaks naturally, and can move between requested locations.

The tested setup fails that condition for independent reasons:

1. **Prompt-only identity failure:** the LoRA did not reliably represent the intended person without a first-frame reference.
2. **Reference dependence:** the output became recognisable only when the actual reference image dominated generation.
3. **Location control failure:** lowering reference strength did not produce the requested new setting.
4. **Visual realism gap:** sampled frames show exaggerated facial shapes and expressions; this is not sufficient evidence for an indistinguishable result.
5. **Trainer-preview instability:** the returned training preview itself was visually invalid, which is a serious warning sign before any broader roll-out.

## Recommendation

Stop this configuration here. Do not train more customer LoRAs, promise talking-head realism, or deploy it for customer-facing avatar generation based on these results.

Any future experiment should be treated as a new, separately approved evaluation with a fresh quality gate. It should not reuse this result as evidence that native LTX A2V can meet an “indistinguishable” standard.

## Reproducibility metadata

The committed video files are generated outputs only. Their SHA-256 checksums and non-sensitive technical metadata are listed in [the pilot manifest](videos/a2v-managed-pilot.manifest.json). The private execution record contains the completed provider artifacts, original request IDs, LoRA hash, and raw provider responses.

## Primary provider references

- [LTX managed A2V trainer](https://fal.ai/models/fal-ai/ltx23-trainer-v2/a2v/api)
- [LTX distilled A2V LoRA API](https://fal.ai/models/fal-ai/ltx-2.3-22b/distilled/audio-to-video/lora/api)
- [LTX distilled A2V LoRA pricing](https://fal.ai/models/fal-ai/ltx-2.3-22b/distilled/audio-to-video/lora)

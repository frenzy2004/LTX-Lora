# LTX Full-22B A2V LoRA — Held-Out Location Matrix

## Evidence status

This package contains three generated LTX Full-22B audio-to-video LoRA outputs, each made from a different held-out speaking clip. It documents a controlled location-coverage test after the earlier distilled diagnostic was rejected.

The outputs are candidates for blind playback review; they are **not** certified as indistinguishable from real footage and are **not** approved for production release. Frame inspection, file integrity, and a successful provider response cannot prove that ordinary viewers will not recognize generated content.

No external image-generation, animation, lip-sync, or post-processing model was used to make the three clips in this matrix. Each uses LTX audio-to-video conditioning, the private adapter, a held-out first frame, and the held-out clip's speech audio.

## Generated outputs

| Held-out location | Generated output | Technical status |
| --- | --- | --- |
| Location 1 | [MP4](videos/a2v-full22b-lora-location-1.mp4) | Valid H.264/AAC, 576×960, 89 frames, 24 fps, 3.708 s |
| Location 2 | [MP4](videos/a2v-full22b-lora-location-2.mp4) | Valid H.264/AAC, 576×960, 89 frames, 24 fps, 3.708 s |
| Location 3 | [MP4](videos/a2v-full22b-lora-location-3.mp4) | Valid H.264/AAC, 576×960, 89 frames, 24 fps, 3.708 s |

The output hashes, byte counts, and publication-safety assertions are in [the matrix manifest](videos/a2v-full22b-location-matrix.manifest.json).

## Controlled configuration

All three runs used the same production-oriented Full-22B configuration. The held-out first frame and held-out audio changed by location; the model and inference settings did not.

| Setting | Value |
| --- | --- |
| Model path | LTX 2.3 Full 22B Audio-to-Video LoRA |
| Adapter scale | 0.50 |
| Output size | 576×960 vertical |
| Frames / FPS | 89 / 24 |
| Duration | 3.708 seconds |
| Inference steps | 40 |
| Quality mode | Maximum |
| Camera guidance | Static camera |
| Multiscale | Enabled |
| Image and audio conditioning strength | 1.0 / 1.0 |
| Seed | Fixed across the matrix |
| Prompt expansion | Disabled |

Fal documents the Full-22B Audio-to-Video LoRA endpoint and its configuration surface here: [Full-22B A2V LoRA API](https://fal.ai/models/fal-ai/ltx-2.3-22b/audio-to-video/lora/api).

## What this evidence establishes

- Three different held-out speaking locations were rendered with the same Full-22B LoRA configuration.
- Each published file passed local media inspection: one H.264 video stream, one AAC audio stream, the requested 576×960 shape, 89 frames, and a 3.708-second duration.
- The output bundle is reproducible at the configuration level without exposing source media, hosted asset URLs, provider request identifiers, or credentials.
- The switch from the earlier distilled controls to Full-22B is an intentional model-path change, not a hidden change to the source media or test settings.

## What this evidence does not establish

- It does not prove indistinguishability from real video.
- It does not establish acceptance across arbitrary camera moves, durations, or locations beyond these three held-out inputs.
- It does not establish that words, teeth, jaw motion, blinks, hair, and background motion will pass a human blind review at every playback speed or compression level.

The appropriate acceptance test is a blinded, randomized playback comparison against real held-out clips with reviewers not told which clips are generated. LTX's own LoRA guidance recommends blind review for this decision. [LTX LoRA training guide](https://ltx.io/blog/training-your-first-lora-on-ltx)

## Conservative spend snapshot

This is a local spend-control ledger, not a provider invoice. It counts completed requests and one provider result that remains conservatively marked as uncertain.

| Ledger item | USD |
| --- | ---: |
| Completed managed training | 6.0000 |
| Earlier pilot and diagnostic work | 4.5209 |
| Full-22B control for location 2 | 0.1000 |
| Full-22B controls for locations 1 and 3 | 0.2000 |
| **Accounted / reserved total** | **10.8209** |
| **Remaining inside the original 12.0000 cap** | **1.1791** |

No additional inference is authorized by this report. The three files above should be reviewed before spending the remaining cap.

## Publication boundary

Only generated MP4 outputs and non-sensitive integrity metadata are committed. The repository intentionally excludes raw source footage, held-out first frames, source audio, adapter weights, provider response payloads, provider URLs, request identifiers, and all credential material.

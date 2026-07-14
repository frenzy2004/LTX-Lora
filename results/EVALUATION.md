# LTX Character LoRA Pilot — Evaluation Record

This document contains only sanitized aggregate evidence. Provider request identifiers, storage URLs, secrets, source media, reference images, and adapter weights are intentionally excluded.

## Candidate and output settings

- Model family: LTX 2.3 distilled custom-LoRA inference on fal
- Adapter: 500-step I2V LoRA smoke candidate, rank 32
- Identity outputs: 704×1248, 24 fps, 89 frames (3.708 seconds)
- Supplied-audio output: 704×1248, 24 fps, 265 frames (11.042 seconds)
- LoRA scale: 0.8
- Published clips include their returned audio tracks

## Results

| Test | Conditioning | Inference cost | Result |
|---|---|---:|---|
| Prompt-only identity smoke test | Text + LoRA | $0.1099 | Failed identity fidelity. The output was a coherent talking head, but it depicted a different person. |
| Reference-conditioned identity test | First frame + text + LoRA | $0.1099 | Strong identity preservation and coherent facial motion. The source environment was also preserved, showing that location changes require location-specific first frames. |
| Supplied-audio talking-head test | First frame + supplied audio + LoRA | $0.3272 | **Rejected as obviously AI.** Mouth and lip shapes deformed instead of forming stable speech-linked motion; the jaw and cheeks warped; the eyes and overall expression changed unnaturally; and the face developed synthetic texture. The returned audio matched the supplied recording at 165.17 dB PSNR, but waveform preservation does not establish visual lip-sync quality. |
| Preservation-first real-video control | Genuine source video + supplied audio, separate non-LTX lip edit | $1.4667 | **Promising, pending blinded review.** Skin, hair, eyes, head motion, and background retain genuine-footage texture. No gross face warp, seam, halo, or flicker was found, but brief teeth/inner-mouth smoothing remains a possible manipulation tell. This result cannot establish arbitrary-location generation. |

Published clips:

- [Prompt-only identity failure](videos/t2v-lora-prompt-only-identity-failure.mp4)
- [Reference-conditioned identity test](videos/i2v-lora-reference-conditioned.mp4)
- [Supplied-audio talking-head test](videos/a2v-lora-supplied-audio.mp4)
- [Preservation-first real-video control](videos/sync-v3-real-video-control.mp4)

## Supplied-audio rejection

The 11-second supplied-audio clip fails the indistinguishability gate and is not an acceptable production result. Manual review found visibly synthetic mouth motion, unstable lip shapes, jaw stretching, cheek deformation, eye and expression changes that did not track the speech naturally, and generated-looking facial texture. These defects remain apparent even though the subject is recognizable.

The 165.17 dB audio PSNR measures preservation of the supplied audio waveform in the returned file. It does **not** measure phoneme-to-mouth alignment, timing, facial realism, identity stability, or whether the performance looks human. Audio preservation passed; visual lip-sync quality failed.

## Preservation-first control review

The separate real-video control edits an authorized genuine source performance rather than regenerating the full face and scene. A frame-by-frame comparison over all 326 frames found that changes remain concentrated in the lower face:

| Region | PSNR | SSIM |
|---|---:|---:|
| Whole frame | 34.9880 dB | 0.9789 |
| Background | 39.9119 dB | 0.9914 |
| Upper face | 33.6950 dB | 0.9715 |
| Lower face | 26.4457 dB | 0.8882 |
| Mouth core | 23.6868 dB | 0.8136 |

These full-reference metrics show where pixels changed; they cannot establish anatomically correct visemes or human detectability. A separate blur audit measured a 19.1% higher score in the lower-face crop and 12.7% in the mouth crop versus the source, consistent with localized softening. No clear beard popping, jaw/cheek warp, eye deformation, hard mask seam, facial-boundary halo, or gross temporal flicker was found. Potential tells remain at approximately 0.30–0.53 s, 6.77–7.03 s, and 8.80–9.17 s, where teeth or inner-mouth detail becomes unusually smooth or even. The control is therefore not labeled indistinguishable; normal-speed listening and a blinded mixed real-versus-output review remain mandatory.

## Cost ledger

| Operation | Cost |
|---|---:|
| 500-step managed I2V LoRA training | $1.2000 |
| Prompt-only LoRA inference | $0.1099 |
| Reference-conditioned LoRA inference | $0.1099 |
| Supplied-audio LoRA inference | $0.3272 |
| Preservation-first real-video control | $1.4667 |
| **Successful completed-operation total** | **$3.2137** |
| Rejected `.mp4` audio submission retained as a conservative cap debit | $0.3272 |
| **Conservative cap total** | **$3.5409** |
| **Remaining under the $12.00 cap** | **$8.4591** |

The successful completed-operation total includes the I2V training, three completed LTX inference jobs, and the separate preservation-first control. The separate rejected submission failed provider input validation before rendering because `.mp4` is not a supported audio container. It may be non-billable, but the local ledger deliberately continues counting its full projected cost until billing reconciliation. The figures exclude engineering time, storage, and unrelated provider usage.

## Interim conclusion

The 500-step smoke candidate is not viable as a prompt-only identity model. Reference conditioning improves short-clip identity, but the tested I2V LoRA plus supplied-audio A2V route produced an obviously synthetic performance and failed the indistinguishability gate. Native generated speech also remains unsuitable when exact wording is mandatory.

Proper LTX A2V training is a separate experiment, not a parameter adjustment to this I2V smoke adapter. The completed request used the I2V trainer with audio disabled, so it received no synchronized phoneme-to-face supervision. A proper A2V archive requires a start image, synchronized audio, matching target video, and factual caption for each group. The managed endpoint currently lists $0.006 per step; the approval-gated plan limits the next run to 1,000 steps ($6) rather than the 2,000-step default ($12).

The current production-oriented fallback is preservation-first editing of genuine footage recorded in the required location. It is substantially more realistic than the rejected generative clip, but it still needs blinded review and cannot create arbitrary new locations from nothing. Proper A2V training remains **unproven R&D**, not production-approved.

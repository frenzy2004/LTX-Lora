# LTX Character LoRA Pilot — Evaluation Record

This document contains only sanitized aggregate evidence. Provider request identifiers, storage URLs, secrets, source media, reference images, and adapter weights are intentionally excluded.

## Candidate and output settings

- Model family: LTX 2.3 distilled custom-LoRA inference on fal
- Adapter: 500-step I2V LoRA smoke candidate, rank 32
- Identity outputs: 704×1248, 24 fps, 89 frames (3.708 seconds)
- Supplied-audio output: 704×1248, 24 fps, 265 frames (11.042 seconds)
- LoRA scale: 0.8
- Published clips include generated audio

## Results

| Test | Conditioning | Inference cost | Result |
|---|---|---:|---|
| Prompt-only identity smoke test | Text + LoRA | $0.1099 | Failed identity fidelity. The output was a coherent talking head, but it depicted a different person. |
| Reference-conditioned identity test | First frame + text + LoRA | $0.1099 | Strong identity preservation and coherent facial motion. The source environment was also preserved, showing that location changes require location-specific first frames. |
| Supplied-audio talking-head test | First frame + supplied audio + LoRA | $0.3272 | Stable identity for 11 seconds with synchronized facial motion. Expression was somewhat exaggerated. The returned audio matched the supplied recording at 165.17 dB PSNR. |

Published clips:

- [Prompt-only identity failure](videos/t2v-lora-prompt-only-identity-failure.mp4)
- [Reference-conditioned identity test](videos/i2v-lora-reference-conditioned.mp4)
- [Supplied-audio talking-head test](videos/a2v-lora-supplied-audio.mp4)

## Cost ledger

| Operation | Cost |
|---|---:|
| 500-step managed I2V LoRA training | $1.2000 |
| Prompt-only LoRA inference | $0.1099 |
| Reference-conditioned LoRA inference | $0.1099 |
| Supplied-audio LoRA inference | $0.3272 |
| **Rendered-operation subtotal** | **$1.7470** |
| Rejected `.mp4` audio submission retained as a conservative cap debit | $0.3272 |
| **Conservative capped total** | **$2.0742** |
| **Remaining under the $12.00 cap** | **$9.9258** |

The rejected submission failed provider input validation before rendering because `.mp4` is not a supported audio container. It is unlikely to be billable compute, but the local ledger deliberately continues counting its full projected cost until billing reconciliation. The figures exclude engineering time, storage, and unrelated provider usage.

## Interim conclusion

The smoke candidate is not viable as a prompt-only identity model. The practical path is an I2V workflow: create or select an identity-matched first frame for the requested setting, then animate it with the customer LoRA. Supplied-audio A2V preserves the approved recording and identity well enough for further testing, although expression strength needs tuning. Native generated speech remains unsuitable when exact wording is mandatory.

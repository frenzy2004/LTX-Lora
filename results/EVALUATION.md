# LTX Character LoRA Pilot — Evaluation Record

This document contains only sanitized aggregate evidence. Provider request identifiers, storage URLs, secrets, source media, reference images, and adapter weights are intentionally excluded.

## Candidate and output settings

- Model family: LTX 2.3 distilled custom-LoRA inference on fal
- Adapter: 500-step I2V LoRA smoke candidate, rank 32
- Output: 704×1248, 24 fps, 89 frames (3.708 seconds)
- LoRA scale: 0.8
- Published clips include generated audio

## Results

| Test | Conditioning | Inference cost | Result |
|---|---|---:|---|
| Prompt-only identity smoke test | Text + LoRA | $0.1099 | Failed identity fidelity. The output was a coherent talking head, but it depicted a different person. |
| Reference-conditioned identity test | First frame + text + LoRA | $0.1099 | Strong identity preservation and coherent facial motion. The source environment was also preserved, showing that location changes require location-specific first frames. |

Published clips:

- [Prompt-only identity failure](videos/t2v-lora-prompt-only-identity-failure.mp4)
- [Reference-conditioned identity test](videos/i2v-lora-reference-conditioned.mp4)

## Cost ledger

| Operation | Cost |
|---|---:|
| 500-step managed I2V LoRA training | $1.2000 |
| Prompt-only LoRA inference | $0.1099 |
| Reference-conditioned LoRA inference | $0.1099 |
| **Total spent** | **$1.4198** |
| **Remaining under the $12.00 cap** | **$10.5802** |

The reported figures are projected endpoint charges from the pilot's atomic local cost ledger. They exclude engineering time, storage, and any unrelated provider usage.

## Interim conclusion

The smoke candidate is not viable as a prompt-only identity model. The practical path is an I2V workflow: create or select an identity-matched first frame for the requested setting, then animate it with the customer LoRA. Exact scripted speech remains a separate audio-driven evaluation because native generated speech is not guaranteed to reproduce an approved script verbatim.

# Raindeer x VideoClaw Burger Review Proof

This proof is a render-only follow-up to the round-1 Raindeer character LoRA run.
It does not train a new LoRA and does not call a paid gateway during composition.

## Output

- Video: `results/videos/raindeer-videoclaw-round1-burger-review-20s.mp4`
- Contact sheet: `results/videos/raindeer-videoclaw-round1-burger-review-20s-contact.jpg`
- Duration: 20.000 seconds
- Shape: 576x1024, 24 fps, 480 frames

## Sources

- Round-1 burger review render:
  - `outputs/ltx23_orvo_round1_burger/ltx23_orvo_round1_burger_10s_i2v.mp4`
- Round-1 identity bridge render:
  - `outputs/ltx23_orvo_03_i2v.mp4`
- gbro collage B-roll demo inserts:
  - `gbro-collage-broll/assets/demo-teal.gif`
  - `gbro-collage-broll/assets/demo-yellow.gif`
  - `gbro-collage-broll/assets/demo-red.gif`
  - `gbro-collage-broll/assets/demo-purple.gif`

## Edit

The final reel is four five-second beats:

1. Burger review setup, full frame, with a small teal collage insert.
2. Burger review continuation, full frame, with a yellow collage insert.
3. Round-1 Raindeer identity bridge, with a red collage insert.
4. Burger review reprise, lightly cropped in, with a purple collage insert.

The collage layer follows the VideoClaw/gbro idea of editorial B-roll accents, but
the character footage stays locked to the round-1 LTX/Raindeer outputs.

## Cost Ledger

New spend for this proof: **$0.00**.

- LoRA training: $0.00, because the round-1 LoRA was reused.
- LTX generation: $0.00, because no fresh model render was submitted.
- VideoClaw gateway: $0.00, because composition was local and no gateway call was made.
- Local ffmpeg composition: $0.00 provider cost.

If this exact 20-second piece were regenerated fresh as one LTX-2.3 quality render,
the rough model-side cost would be:

- 576x1024 at 24 fps for 20 seconds: 576 * 1024 * 480 = 283.1 megapixels.
- At $0.0027075 per megapixel: about **$0.77**.
- At 720x1280 for 20 seconds: about **$1.20**.

Those are inference estimates only. They do not include any new training steps.

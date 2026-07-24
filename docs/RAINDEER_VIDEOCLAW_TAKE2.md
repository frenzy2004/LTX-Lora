# Raindeer x VideoClaw Take 2 Proof

This proof corrects the Round 1-to-VideoClaw interpretation:

- Raindeer/LTX is the identity source.
- VideoClaw is used for animation/cutaway/post-production energy.
- gbro collage is used as the cutaway style reference, not as identity.

## Output

- Video: `results/videos/raindeer-videoclaw-take2-20s.mp4`
- Contact sheet: `results/videos/raindeer-videoclaw-take2-20s-contact.jpg`
- Duration: 20.053 seconds
- Shape: 720x1280, 30 fps, 600 frames

## Sources

- Existing round-1 Raindeer LoRA and reference frame.
- Existing round-1 burger hook render.
- One fresh LTX/Raindeer continuation render for the bite/verdict segment.
- VideoClaw insert render for the mustard-paper gbro-style burger cutaway.
- VideoClaw retrofit composer for the final 20-second assembly.

## Edit

The final reel is:

1. 0-7s: Raindeer burger hook, on camera.
2. 7-11s: VideoClaw/gbro-style burger cutaway over the transition.
3. 11-20s: Raindeer returns on camera, takes the bite, and gives the verdict.

The VideoClaw insert was cropped and trim-held locally to remove accidental
readable panel text while keeping the generated motion-design beat.

## Cost Ledger

New spend for this take: **$3.33866**.

- New LoRA training: $0.00, because the round-1 LoRA was reused.
- Fresh LTX/Raindeer 10s i2v continuation render: $0.61000.
- VideoClaw insert render: $2.72800.
- VideoClaw transcription for compositor timing: $0.00066.
- Local Remotion/ffmpeg composition, crops, contact sheets, and verification: $0.00 provider cost.

This does not include the already-spent round-1 LoRA training cost.

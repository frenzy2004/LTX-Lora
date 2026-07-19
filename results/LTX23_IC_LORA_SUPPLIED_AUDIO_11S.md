# LTX 2.3 Participant IC-LoRA — 11-second supplied-audio delivery

This package contains the selected LTX-only talking-head result driven by the complete supplied WhatsApp audio.

## Delivery

- `ltx_ic_lora_whatsapp_audio_11s.mp4` — selected 9:16 talking-head video with the supplied speech track.
- `contact_sheet_2fps.png` — full-duration identity and motion review at two frames per second.
- `mouth_identity_sweep_4fps.png` — full-duration mouth/identity review at four frames per second.
- `qa-report.json` — sanitized mechanical, audio, and visual QA results.
- `workflow-manifest.json` — sanitized model, adapter, inference, hash, and cost provenance.

## Selected workflow

1. Decode the complete supplied AAC audio to 48 kHz stereo PCM with FFmpeg, removing only the original AAC priming delay.
2. Generate the audio-conditioned motion driver with LTX 2.3 Pro A2V.
3. Derive RGB and balanced-Canny reference videos from that motion driver.
4. Apply the trained 3,000-step participant IC-LoRA through LTX 2.3 Quality reference-video-to-video at LoRA scale `0.6`, using a real participant first-frame anchor.
5. Remux the complete normalized supplied audio at time zero and retain a short natural video settle after speech ends.
6. Decode-test the final file, run black/freeze checks, compare redacted Whisper transcripts, measure PCM alignment, and visually inspect the full duration at 2 fps and 4 fps.

No non-LTX generative provider or external lip-sync system was used.

## Selection rationale

The no-anchor variant was rejected because it imported a wood-panel background and polo-style clothing from the training distribution. The full-strength anchored variant was rejected because it over-lightened and softened the face. The selected `0.6` anchored result retained the real room, clothing, fuller facial structure, beard, and natural texture while preserving the audio-driven mouth performance.

## Quality interpretation

The final file passes the recorded mechanical, transcript, alignment, and sampled visual checks. The 4-fps sweep shows stable identity and continuous articulation. These checks do not constitute a mathematical guarantee that every viewer will consider the clip indistinguishable from camera footage; final acceptance remains a human perceptual decision.

## Costs

- One-time main IC-LoRA training run: **$17.70**.
- Selected inference path: **$1.48** (`$1.10` LTX 2.3 Pro A2V + `$0.38` LTX 2.3 Quality IC-LoRA V2V).
- Pilot ledger at packaging time: **$27.18** accounted or reserved, including training, experiments, accepted generations, and two failed requests awaiting billing verification.

The adapter weights, source corpus, credentials, signed provider URLs, private transcripts, and request identifiers are intentionally not included in this delivery folder.

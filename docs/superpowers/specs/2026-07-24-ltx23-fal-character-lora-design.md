# LTX-2.3 FAL Character LoRA Output Design

## Goal

Create finished LTX-2.3 character LoRA outputs from the user's portrait source video, following the tutorial workflow: split one talking-head source into short clips, caption each clip, train a character LoRA, then render short sample videos with that LoRA.

## Sources

- User source video: `IMG_3816-001.MOV` from the user's Downloads folder
- Tutorial video: `videoplayback.mp4` from the user's Downloads folder
- Official LTX-2 repository: `LTX-2/` in the workspace
- Budget/helper repository: `LTX-Lora/` in the workspace

## Constraints

- Use only LTX models and LTX/FAL endpoints.
- Use the current LTX-2.3 trainer and inference endpoints.
- Keep a hard local budget cap of USD 25.00.
- Do not use the user's real name in trigger phrases, captions, prompts, filenames, or model labels.
- Use an invented neutral trigger, `orvo`, in captions and render prompts.
- Treat API keys as secrets. Read them from process environment only and do not write them into repo files.
- Save final user-facing videos and manifests under the workspace `outputs/` directory.

## Selected Approach

Use `fal-ai/ltx23-trainer-v2/t2v` for training because the goal is a prompt-summoned talking-head character LoRA. Use the tutorial's dataset structure: many short clips from one scene, mostly captioned as `orvo says, "..."`, with explicit gesture wording only when the clip contains distinct gestures worth controlling.

Use 2000 training steps, rank 32, 9:16 aspect ratio, 24 fps, audio enabled, and medium training resolution. Current FAL pricing for the V2 T2V trainer is approximately `0.006 * steps`, so training is expected to cost about USD 12.00. The remaining budget supports short LTX-2.3 quality inference renders.

## Dataset Preparation

1. Extract source audio and transcribe it locally with Whisper.
2. Split the 7:34 portrait source into approximately 18-24 clips, favoring 2-8 second spans with clean sentence boundaries.
3. Normalize clips to 9:16, 24 fps, H.264/AAC MP4, with short durations suitable for LTX-2.3 training.
4. Generate matching `.txt` sidecar captions:
   - Use `orvo says, "actual spoken words"` for ordinary speech.
   - Add gestures only when they matter, such as `orvo leans forward and says, "..."`
   - Do not include the user's real name.
5. Create a ZIP archive containing the clips and sidecar captions.

## Training Request

Endpoint: `fal-ai/ltx23-trainer-v2/t2v`

Planned input:

```json
{
  "rank": 32,
  "number_of_steps": 2000,
  "learning_rate": 0.0002,
  "number_of_frames": 121,
  "frame_rate": 24,
  "resolution": "medium",
  "aspect_ratio": "9:16",
  "trigger_phrase": "orvo",
  "auto_scale_input": true,
  "split_input_into_scenes": false,
  "with_audio": true,
  "audio_normalize": true,
  "audio_preserve_pitch": true,
  "validation_number_of_frames": 121,
  "validation_frame_rate": 24,
  "validation_resolution": "high",
  "validation_aspect_ratio": "9:16"
}
```

Validation prompts should be short, random-topic talking-head prompts that do not name the user. Example:

- `orvo says, "I think the real question is whether a sandwich can count as architecture."`
- `orvo leans toward the camera and says, "Today I learned that coffee mugs have stronger opinions than most meetings."`

## Inference Plan

After training, use LTX-2.3 LoRA inference only. Prefer `fal-ai/ltx-2.3-quality/text-to-video/lora` for final prompt-only talking-head outputs. If identity is too weak, use `fal-ai/ltx-2.3-quality/image-to-video/lora` with an approved frame from the source video as the first frame, because prior repo evidence showed image-conditioned inference preserves identity better than prompt-only generation.

Generate short 5-second clips first (`121` frames at 24 fps). Use 9:16 portrait output, audio enabled, LoRA scale near `1.0`, and the FAL negative prompt defaults plus identity/artifact negatives.

## Budget

Hard cap: USD 25.00.

Expected spend:

- Training: about USD 12.00 for 2000 steps.
- Final inference: reserve the remaining budget for several short 5-second LTX-2.3 quality renders.

The runner must reserve projected cost in a local ledger before each paid request and persist request IDs before streaming logs or waiting for results.

## Error Handling

- If transcription produces poor speech text, fall back to shorter generic speech captions that still avoid real names.
- If the dataset upload fails, retry upload without submitting training and do not mark the budget reservation consumed.
- If FAL validation rejects parameters, fix schema fields locally and resubmit only after checking the request did not start.
- If prompt-only output fails identity preservation, switch to image-to-video LoRA inference with a selected source still.
- If budget remaining is too low for another render, stop and return the best generated clip plus the ledger.

## Verification

- Verify ZIP contents contain only `.mp4` clips and matching `.txt` captions.
- Check all captions and prompts for real-name leakage before upload.
- Confirm each training clip is playable and has audio.
- Record FAL request IDs, result URLs, downloaded local file paths, and estimated budget entries.
- Inspect final videos locally for audio, portrait orientation, visible subject, and no obvious broken frames before reporting completion.

# Broad A2V Dataset Builder Design

## Status

Approved. This document is the Git-safe implementation projection of the separately approved private 4,000-step training plan. It intentionally excludes participant names, source paths, captions, transcripts, media, API credentials, request identifiers, signed URLs, and model artifacts.

## Goal

Build a deterministic, source-isolated Fal LTX-2.3 A2V dataset from a broad corpus of real participant videos. Every accepted group must contain a face-aware portrait crop, its exact first frame, sample-aligned speech audio, a matching 89-frame target, and a factual caption. The builder must reject technically valid but visually or acoustically unsuitable windows rather than padding the archive.

## Alternatives Considered

### Fixed centre crop

Rejected. Earlier review showed that a single horizontal-to-portrait centre crop cuts or undersizes the participant in off-centre and full-body source shots.

### Reuse the small manually audited bundle

Rejected as the primary route. It is mechanically clean but covers only a small fraction of the available source corpus and cannot answer whether broader identity, angle, expression, clothing, and location coverage improves the held-out result.

### Face-aware broad-corpus builder

Selected. Speech windows are discovered from word-timestamped audio, faces are detected across each proposed interval, one fixed crop is derived from the temporal face envelope, and every rendered group is mechanically and visually audited before upload.

## Architecture

The implementation is split into focused units:

1. `a2v_broad_dataset.py` contains pure selection, split, crop, caption-template, and validation helpers plus an orchestration CLI.
2. `test_a2v_broad_dataset.py` locks the public behavior with synthetic media fixtures and pure-unit tests.
3. Private run metadata supplies source aliases, paths, approved caption style, and the selected trigger token. None of that private metadata is committed.
4. The local canonical dataset remains normal colour. A separate provider-only visual mirror is pre-inverted because a prior controlled test on the same endpoint showed that Fal's preprocessing inverted normal RGB while decoding pre-inverted inputs back to normal colour.
5. A 100-step request with `debug_dataset=true` must prove that Fal decoded normal colour, the expected group count, 544×960 video, 89 frames at 24 fps, 48 kHz audio, and intact captions before the 4,000-step request is allowed.

## Source and Split Rules

- Probe every complete candidate with FFprobe; require one decodable video stream, one decodable audio stream, and at least `89 / 24` seconds.
- Discover synchronized speech windows from word timestamps. Candidate intervals are exactly `89 / 24` seconds and cannot overlap within a source.
- Reserve roughly ten percent of passing source files, with a minimum of five, using seed 42. A holdout source contributes no training window, alternate crop, or neighboring interval.
- Select at least one passing window from each training source when possible. A second window is permitted only when its framing/location bucket differs and it does not overlap the first.
- Never claim a target group count before the visual and audio gates finish.

## Face-Aware Crop

- Detect faces on frames sampled through the complete proposed interval.
- Reject a window when no stable primary face is found, a second similarly sized face is prominent, or detection coverage is below the configured threshold.
- Derive one fixed 9:16 crop from the primary face envelope. A fixed crop avoids synthetic camera jitter while retaining the full hair, jaw, and observed head-motion envelope.
- Scale to 544×960 with Lanczos resampling.
- Reject crops that clip the face envelope or leave the median face too small for the talking-head target.
- Store crop coordinates and aggregate detection metrics only in the private manifest.

## A2V Group Contract

For each accepted basename `<id>`:

- `<id>_start.png`: exact decoded first frame of the final target.
- `<id>_audio.wav`: mono PCM s16le, 48 kHz, exactly aligned to the source interval.
- `<id>_end.mp4`: silent H.264 target, 544×960, 24 fps, exactly 89 frames.
- `<id>.txt`: approved factual caption without transcript content; the private trigger is applied by the provider request.

The archive contains training groups only. Holdouts stay local and private.

## Validation and Review

Mechanical validation is all-or-nothing:

- exact four-file membership;
- unique basenames;
- exact dimensions, frame count, frame rate, audio sample rate, and duration tolerance;
- start PNG pixel equality with the decoded first target frame;
- audio duration and start alignment;
- train/holdout source disjointness;
- archive membership and SHA-256 manifest;
- workspace derived-data ceiling of 8 GiB.

Visual review sheets show sampled real target frames, face boxes/crop envelope, face-size and detection-coverage summaries, and inclusion/exclusion status. They do not show IC-LoRA Canny controls because A2V has no reference-video control input.

## Provider Colour Mirror

The normal-colour local dataset is authoritative. The provider mirror inverts only RGB video/start-image pixels while leaving audio and captions byte-identical. The 100-step debug archive is compared against the normal-colour masters. Any material mismatch blocks the main request; an ambiguous or existing provider request is never resubmitted.

## Training and Telemetry Boundary

- Sanity request: 100 steps, maximum $0.60 at the pinned $0.006/step rate.
- Candidate request: exactly 4,000 steps, rank 32, learning rate 0.0001, maximum $24.00.
- Only one paid request may be in flight.
- Persist every provider-emitted progress, validation, warning, and loss field. Record `provider_loss_not_exposed` when no loss is emitted; never fabricate accuracy.
- Model accuracy is the held-out acceptance rate under identity, mouth/jaw synchronization, temporal stability, and obvious-AI-artifact criteria.

## Security and Publication

Git receives only this design, the reusable builder/tests, sanitized aggregate counts/hashes, provider telemetry without private identifiers, cost summaries, and honest evaluation results. Raw video, audio, captions containing transcript content, face images, private paths, trigger tokens, request IDs, signed URLs, credentials, LoRA weights, and provider artifact URLs remain private.

## Success Criteria

The builder succeeds when every uploaded group passes the local contract and the provider-decoded debug archive matches the normal-colour canonical dataset. That proves input integrity only. The model succeeds only if held-out and exact-audio outputs meet the separately approved visual acceptance rubric; training completion or lower loss is insufficient.

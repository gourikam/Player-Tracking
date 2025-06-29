# 📄 Project Report: Player Re-Identification in a Single Feed

## Objective

Given a 15-second video clip (`15sec_input_720p.mp4`), the goal was to:
- Detect and assign unique IDs to all players.
- Maintain **consistent IDs** even when players **exit and re-enter** the frame.
- Use the provided **YOLOv11 model** for object detection and simulate **real-time tracking**.

---

##  Methodology & Approach

1. **Object Detection (YOLOv11):**
   - Used the provided `best.pt` model, trained on players and balls.
   - Filtered detections with a confidence threshold and size constraints.
   - Referees wearing yellow jerseys were excluded using HSV masking.

2. **Object Tracking (Deep SORT):**
   - Integrated Deep SORT with a `mobilenet` appearance embedder.
   - Tuned parameters (`max_age`, `n_init`, `iou_distance`) for better re-ID consistency.
   - Maintained a mapping from internal Deep SORT ID to a fixed player ID (1–22).

3. **Team Classification:**
   - Determined team color using HSV analysis of red and blue jerseys.
   - Bounding boxes were color-coded: red for one team, blue for the other.
   - Green boxes (unknown classification) were omitted for clarity.

4. **Video Output:**
   - Rendered the video with bounding boxes and player IDs.
   - Adjusted FPS to match original playback speed using OpenCV.

---

## Techniques Tried & Outcomes

| Technique                            | Outcome                                                       |
|-------------------------------------|---------------------------------------------------------------|
| Deep SORT default parameters        | IDs changed frequently when players left and re-entered frame |
| HSV filtering for yellow jerseys    | Successfully filtered out referees                            |
| Mapping internal ID → fixed ID      | Ensured stable IDs (max 22 players)                           |
| FPS doubled (output video)          | Restored video to normal speed after OpenCV delay             |
| Color-based team classification     | Useful for distinguishing teams visually                      |

---

## Challenges Encountered

- **ID Fluctuation**: Deep SORT initially reassigned IDs due to appearance changes or occlusions.
- **Model Size Limits**: The `best.pt` file exceeded GitHub's 100MB limit, had to upload via Google Drive.
- **Team Classification**: Color detection sometimes failed under poor lighting or partial occlusion.

---

## What Remains / Future Improvements

-  **Better Re-ID model**: Use a stronger re-identification model trained on jersey patterns.
-  **Learnable team assignment**: Replace HSV heuristic with a small classifier.
-  **Multi-camera tracking**: Extend solution for multi-view player tracking (future scope).
-  **Real-time optimization**: Reduce inference latency using ONNX or TensorRT.

---

## Summary

Despite real-world constraints (limited video duration and model size), the solution:
- Assigns stable, consistent IDs to players across frames.
- Filters out referees and false positives.
- Demonstrates effective team-level separation using visual cues.

This reflects a robust first-pass solution for real-time **player tracking and re-identification**.


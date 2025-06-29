# 📄 Player Re-Identification – Project Report

## Approach and Methodology

The primary objective of this project was to detect and **re-identify football players** in a 15 second clip, ensuring consistent IDs even when players leave and re enter the frame.

To achieve this:
- I used **YOLOv11** (fine-tuned) for detecting player bounding boxes.
- For re-identification and maintaining consistent IDs, I integrated **Deep SORT**, which relies on appearance features and motion.
- To reduce ID switching, I adjusted Deep SORT parameters (`max_age`, `n_init`, `max_iou_distance`) and added **post-detection filters** to discard false positives like referees.
- Players were also color-classified into teams (red or blue) using HSV thresholding, although only the ID is displayed on screen to avoid clutter.

---

## Techniques Tried & Outcomes

### What Worked Well:
- YOLOv11 was very accurate in detecting players, even in crowded frames.
- Filtering out false positives using bounding box size and yellow jersey detection significantly improved accuracy.
- Deep SORT successfully tracked players when they were briefly occluded or moved fast.
- Capping the total number of IDs to 22 prevented ID overflow.

### Iterations:
- Initially, player IDs kept switching frequently. After tuning `max_age` and `n_init`, tracking became more stable.
- I tested multiple color-space methods (RGB, HSV) for team detection and finally used HSV due to better performance in varying lighting.

---

## Challenges Encountered

- **False Positives:** YOLO occasionally detected patches of grass as players. I resolved this by applying size constraints and confidence filtering.
- **ID Switching:** When players moved too fast or overlapped, Deep SORT sometimes lost the track. To mitigate this, I adjusted `max_age` and required more consistent frames (`n_init`).
- **Model File Size:** The `best.pt` file exceeded GitHub’s 100MB limit, so I provided it via Google Drive.

---

## 🔧 What Remains / Next Steps

If I had more time or resources:
- I would fine-tune the Deep SORT embedder or try **stronger re-ID embeddings** to reduce ID switching.
- Explore **Kalman filter tuning** to further improve re-identification consistency.
- Integrate ball tracking to understand player-ball interactions.
- Explore temporal smoothing or transformer-based tracking.

---

## Final Notes

Despite the tight constraints, I managed to build a fairly robust player tracking and re-identification pipeline using YOLOv11 + Deep SORT. The results are consistent and real-time, simulating how a sports analytics tool might track players across events.

Thank you for the opportunity!

— **Gourika Makhija**


# Player Re-Identification using YOLOv11 and Deep SORT

This project implements **player tracking and re-identification** in a football match using:
- **YOLOv11** for player detection (fine-tuned on football dataset)
- **Deep SORT** for consistent ID assignment across frames
-  Color-based team classification (Red vs Blue jerseys)
-  Referee filtering (based on yellow clothing)

---

## 🔧 Setup & Run Instructions

### 1. Clone the repository
```bash
git clone https://github.com/gourikam/Player-Tracking.git
cd Player-Tracking
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # for Unix/Mac
venv\Scripts\activate     # for Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the YOLOv11 weights
- The trained model file 'best.pt' is too large for GitHub.
- Download it from here: https://drive.google.com/file/d/1NiyvLnjGgroRQNX-yhcn7yH4iDfcou-b/view?usp=drive_link
- Place it in the root of the project directory.

### 5. Run the code
```bash
python player_tracking.py
```

### 6. Output
- Players are tracked with consistent IDs.
- Team colors are indicated via bounding box (🔴 Red / 🔵 Blue).
- Referees are excluded from tracking.
- Output video is saved as player_tracking.mp4.

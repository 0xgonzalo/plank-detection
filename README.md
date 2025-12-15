# Plank Detection System

Real-time plank exercise detection and form analysis using MediaPipe pose estimation. Earn tokens by holding proper plank form!

## Features

- **Real-time pose detection** - Uses MediaPipe's 33-point body landmark model
- **Form scoring** - 0-100 score based on body alignment, hip position, and limb angles
- **Live feedback** - Corrective cues displayed on screen
- **Session tracking** - Tracks total time, good form time, and tokens earned
- **Token system** - Earn 1 token per second of good/perfect form

## Installation

```bash
# Clone the repository
cd plank-detection

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- Webcam
- macOS, Windows, or Linux

## Usage

```bash
python plank_detector.py
```

### Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Reset session |

## How It Works

The system analyzes four key metrics for proper plank form:

| Metric | Weight | Ideal Angle | What It Detects |
|--------|--------|-------------|-----------------|
| Body Alignment | 40% | ~180° | Shoulder-hip-ankle line |
| Hip Angle | 30% | ~180° | Sagging or piking |
| Knee Angle | 15% | ~180° | Bent legs |
| Arm Angle | 15% | ~180° | Elbow position |

### Form States

| State | Score | Color |
|-------|-------|-------|
| Perfect Form | 90-100 | Gold |
| Good Form | 75-89 | Green |
| Needs Adjustment | 50-74 | Orange |
| Poor Form | <50 | Red |
| Not Detected | - | Gray |

## Configuration

Customize detection thresholds by modifying `PlankConfig`:

```python
from plank_detector import PlankDetector, PlankConfig

config = PlankConfig(
    body_alignment_min=160.0,      # Minimum acceptable body angle
    body_alignment_max=195.0,      # Maximum acceptable body angle
    model_complexity=1,            # 0=fast, 1=balanced, 2=accurate
    min_detection_confidence=0.5,  # Pose detection threshold
    smoothing_window=5,            # Frames to average (reduces jitter)
)

detector = PlankDetector(config)
```

## API Usage

```python
from plank_detector import PlankDetector, PlankConfig
import cv2

detector = PlankDetector()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame
    processed_frame, metrics = detector.process_frame(frame)

    # Access metrics
    print(f"Score: {metrics.overall_score}")
    print(f"State: {metrics.state.value}")
    print(f"Feedback: {metrics.feedback}")

    # Get session data
    summary = detector.get_session_summary()
    print(f"Tokens: {summary['tokens_earned']}")

detector.release()
cap.release()
```

## Camera Setup Tips

1. **Position** - Place camera 6-10 feet away, at floor level or slightly elevated
2. **Orientation** - Side view works best (captures full body profile)
3. **Lighting** - Ensure good lighting on your body
4. **Clothing** - Fitted clothes help with landmark detection

```
Good camera angle (side view):

    Camera
      |
      v
   [User doing plank] ------>
```

## Testing

```bash
pytest test_plank_detector.py -v
```

## License

MIT

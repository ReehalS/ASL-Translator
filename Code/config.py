"""Project configuration: keywords and defaults for pipeline runs."""

# Default list of 30 common gloss keywords (includes intro words and common conversational words)
KEYWORDS = [
    'hello', 'good', 'goodbye', 'happy', 'love', 'my', 'name', 'sun', 'deep',
    'i', 'am', 'how', 'are', 'you', 'what', 'where', 'when', 'why', 'who', 'fine',
    'thanks', 'please', 'sorry', 'help', 'food', 'hungry', 'weather', 'morning', 'night', 'today'
]

# Per-class target number of videos to collect
TARGET_PER_CLASS = 40

# Local dataset paths
WLASL_JSON = 'Dataset/WLASL/WLASL_v0.3.json'
WLASL_VIDEOS_DIR = 'Dataset/WLASL/videos'
SUBSET_DIR = 'Dataset/WLASL_subset'
GENERATED_CSV = 'Dataset/Generated_Data/wlasl_pipeline_frames.csv'

WINDOW_SIZE = 16
STRIDE = 4

# Aggressive expansion settings
DOWNLOAD_MISSING = True  # set True to enable yt-dlp downloads for missing videos
MAX_DOWNLOADS_TOTAL = 1200  # cap total downloads in an aggressive run (allow up to 40 per class across 30 classes)
MAX_PER_CLASS_DOWNLOAD = 40  # cap downloads per class

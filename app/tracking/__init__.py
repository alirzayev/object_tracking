# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

__version__ = '10.0.28'

from app.tracking.postprocessing.gsi import gsi
from app.tracking.tracker_zoo import create_tracker, get_tracker_config
from app.tracking.trackers.botsort.bot_sort import BoTSORT
from app.tracking.trackers.bytetrack.byte_tracker import BYTETracker
from app.tracking.trackers.deepocsort.deep_ocsort import DeepOCSort as DeepOCSORT
from app.tracking.trackers.ocsort.ocsort import OCSort as OCSORT
from app.tracking.trackers.strongsort.strong_sort import StrongSORT

TRACKERS = ['bytetrack', 'botsort', 'strongsort', 'ocsort', 'deepocsort']

__all__ = ("__version__",
           "StrongSORT", "OCSORT", "BYTETracker", "BoTSORT", "DeepOCSORT",
           "create_tracker", "get_tracker_config", "gsi")

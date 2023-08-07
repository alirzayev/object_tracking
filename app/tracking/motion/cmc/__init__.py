# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from app.tracking.motion.cmc.ecc import ECC
from app.tracking.motion.cmc.orb import ORB
from app.tracking.motion.cmc.sift import SIFT
from app.tracking.motion.cmc.sof import SparseOptFlow


def get_cmc_method(cmc_method):
    if cmc_method == 'ecc':
        return ECC
    elif cmc_method == 'orb':
        return ORB
    elif cmc_method == 'sof':
        return SparseOptFlow
    elif cmc_method == 'sift':
        return SIFT
    else:
        return None

import torch
from functools import partial
from pathlib import Path

from app.tracking import TRACKERS
from app.tracking.tracker_zoo import create_tracker
from app.tracking.utils import EXAMPLES, ROOT, WEIGHTS
from app.tracking.utils.checks import TestRequirements
from ultralytics import YOLO
from ultralytics.data.utils import VID_FORMATS
from ultralytics.utils.plotting import save_one_box
from app.utils.helpers import write_mot_results

class ObjectTracker:
    def __init__(self,  tracking_method='deepocsort', source='0'):
        """
        Initializes the ObjectTracker class.
        """
        self.reid_model = WEIGHTS / 'osnet_x0_25_msmt17.pt'
        self.tracking_method = tracking_method
        self.source = source
        self.conf = 0.5
        self.iou = 0.7
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.show = False
        self.save = False
        self.classes = [0, 2, 3, 5, 7] # person, car, motorcycle, bus, truck
        self.half = False
        self.show_labels = False
        self.show_conf = False
        self.save_txt = False
        self.visualize = False
        self.save_id_crops = False
        self.save_mot = False
        self.verbose = True

        self.test_requirements = TestRequirements()
        self.test_requirements.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', ))
        self.yolo = YOLO('yolov8n.pt')
        self.yolo.add_callback('on_predict_start', partial(self.on_predict_start, persist=True))

    def on_predict_start(self, predictor, persist=False):
        """
        Callback function called when prediction starts.
        """
        assert self.tracking_method in TRACKERS, \
            f"'{self.tracking_method}' is not supported. Supported ones are {TRACKERS}"

        tracking_config = ROOT / 'tracking' / 'configs' / (self.tracking_method + '.yaml')
        trackers = [create_tracker(self.tracking_method,
                                   tracking_config,
                                   self.reid_model,
                                   self.device,
                                   self.half)]

        predictor.trackers = trackers
        predictor.save_dir = predictor.get_save_dir()


    @torch.no_grad() # improve performance with disabling gradient calculations for the following code block.
    def run(self):
        """
        Runs the object tracking process.
        """
        results = self.yolo.track(
            source=self.source,
            classes=self.classes,
            conf=self.conf,
            iou=self.iou,
            show=self.show,
            stream=False,
            device=self.device,
            show_conf=self.show_conf,
            show_labels=self.show_labels,
            save=self.save,
            save_txt=self.save_txt,
            visualize=self.visualize,
            verbose=self.verbose
        )

        self.process_results(results)

    def process_results(self, results):
        """
        Processes the tracking results.
        """
        for frame_idx, r in enumerate(results):
            if r.boxes.data.shape[1] == 7:
                if self.yolo.predictor.source_type.webcam or self.args.source.endswith(VID_FORMATS):
                    p = self.yolo.predictor.save_dir / 'mot' / (self.args.source + '.txt')
                elif 'MOT16' or 'MOT17' or 'MOT20' in self.args.source:
                    p = self.yolo.predictor.save_dir / 'mot' / (Path(self.args.source).parent.name + '.txt')

                self.yolo.predictor.mot_txt_path = p

                if self.save_mot:
                    write_mot_results(
                        self.yolo.predictor.mot_txt_path,
                        r,
                        frame_idx,
                    )

                if self.save_id_crops:
                    for d in r.boxes:
                        print('args.save_id_crops', d.data)
                        save_one_box(
                            d.xyxy,
                            r.orig_img.copy(),
                            file=(
                                self.yolo.predictor.save_dir / 'crops' /
                                str(int(d.cls.cpu().numpy().item())) /
                                str(int(d.id.cpu().numpy().item())) / f'{frame_idx}.jpg'
                            ),
                            BGR=True
                        )

        if self.save_mot:
            print(f'MOT results saved to {self.yolo.predictor.mot_txt_path}')

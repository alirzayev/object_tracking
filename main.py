from app.tracking.object_tracker import ObjectTracker

if __name__ == "__main__":
    tracker = ObjectTracker(tracking_method='ocsort', source='0') 
    tracker.run()

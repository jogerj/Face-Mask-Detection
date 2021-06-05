from imutils.video import VideoStream
import cv2

def load_video():
    """
    Defines method to load video stream source
    Returns:
        (imutils.VideoStream): Video stream
    """
    print("[INFO] starting video stream...")
    vs = VideoStream("http://192.168.1.89:4747/video").start()
    # vs = VideoStream(0).start()
    # vs = cv2.VideoCapture()
    return vs

def transform(frame):
    """
    Compensate for distortions/transformations of video stream source
    Args:
        frame: Frame to transform

    Returns:
        transformed frame
    """
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    return frame
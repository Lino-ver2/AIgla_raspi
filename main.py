import time
import datetime
from picamera import PiCamera
 
from mmdet.apis import init_detector, inference_detector



def build_model(conf, ckpt):
    model = init_detector(conf, ckpt, device='cpu')  # or device='cuda:0'
    return model


def detection(model ,img):
    pred = inference_detector(model, img)
    return pred


def pred_through_threshold(pred, threshold):
    labels = pred.pred_instances.labels
    scores = pred.pred_instances.scores
    result = labels[scores > threshold]
    return result


def flag_switch(result, target):
    if set(result) - set(target):
        return True
    else:
        return False


def take_picture(save_path, width, height):
    camera = PiCamera()
    camera.capture(save_path, resize=(width, height))
    camera.close()
    return None


def shooting_begins(capture_flag, previous, video_resolution):

    if capture_flag == previous:
        return None
    elif capture_flag:
        video_name = datetime.datetime.now().strftime('%Y年%m月%d日%H-%M-%S')
        camera = PiCamera()
        camera.resolution = video_resolution
        camera.start_recording(f"{video_name}.h264")
        return None
    else:
        camera.stop_recodeing()
    return None


def main(target):

    CONF = './conf/rtmdet_tiny_8xb32-300e_coco.py'
    CHECK_POINT = './weights/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
    PICTURE_PATH = "./tem.jpg"
    PICTURE_RESOLUTION = (960, 960)
    THRESHOLD = 0.6
    VIDEO_RESOLUTION = (640, 480)
    INTERVAL = 1

    capture_flg = False
    model = build_model(CONF, CHECK_POINT)

    while True:
        previous = capture_flg
        take_picture(PICTURE_PATH, *PICTURE_RESOLUTION)
        pred = detection(model, PICTURE_PATH)
        result = pred_through_threshold(pred, THRESHOLD)
        capture_flg = flag_switch(result, target)
        shooting_begins(capture_flg, previous, VIDEO_RESOLUTION)
        time.sleep(INTERVAL)
    
    return None
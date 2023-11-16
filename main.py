import os
import time
import datetime

from picamera import PiCamera
from mmdet.apis import init_detector, inference_detector


def build_model(conf, ckpt):
    # モデルのインスタンス生成
    model = init_detector(conf, ckpt, device='cpu')  # or device='cuda:0'
    return model


def detection(model ,img):
    # モデル推論の実行
    pred = inference_detector(model, img)
    return pred


def pred_through_threshold(pred, threshold):
    # 推論結果を閾値に通して厳選
    labels = pred.pred_instances.labels
    scores = pred.pred_instances.scores
    result = labels[scores > threshold]
    return result


def flag_switch(result, target):
    # target を検出したら True, 検出できなかったら False
    if set(result) - set(target):
        return True
    else:
        return False


def take_picture(save_path, width, height):
    # モデルに入力する写真を撮影
    cap = PiCamera()
    cap.capture(save_path, resize=(width, height))
    cap.close()
    return None


def shoot_video(camera, video_resolution):
    # 撮影開始関数
    video_name = datetime.datetime.now().strftime('%Y年%m月%d日%H-%M-%S')
    camera.resolution = video_resolution
    camera.start_recording(f"{video_name}.h264")


def shooting_begins(camera, capture_flag, previous, video_resolution):
    # 撮影終始の判断
    if capture_flag == previous:
        return 0
    elif capture_flag:  # target を検出したら撮影開始
        shoot_video(camera, video_resolution)
        return 1
    else:  #  target を検出しなくなったら撮影終了
        camera.stop_recodeing()
    return 2


def main(target):

    EXECUTION_SLOT = 120  # プログラムの実行スロット
    CONF = './conf/rtmdet_tiny_8xb32-300e_coco.py'
    CHECK_POINT = './weights/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
    PICTURE_PATH = "./tem.jpg"       # モデルへの入力画像の一時ファイルパス
    PICTURE_RESOLUTION = (960, 960)  # モデルへの入力画像解像度
    THRESHOLD = 0.6                  # モデル出力結果のの閾値
    VIDEO_RESOLUTION = (640, 480)    # 撮影解像度
    SHOOTING_LENGHT = 10             # 最大撮影長さ(sec)
    INTERVAL = 1                     # 実行フレーの間隔間隔

    capture_flg = False                     # target の検出フラグ
    model = build_model(CONF, CHECK_POINT)  # モデルインスタンス  
    camera = PiCamera()                     # raspberry pi のカメラインスタンス
    shooting_counter = 0                    # 撮影長さのカウンター
    excution_counter = 0                    # 実行長さのカウンター

    while True:
        previous = capture_flg
        take_picture(PICTURE_PATH, *PICTURE_RESOLUTION)
        pred = detection(model, PICTURE_PATH)
        result = pred_through_threshold(pred, THRESHOLD)
        capture_flg = flag_switch(result, target)
        playing_status = shooting_begins(camera, capture_flg, previous, VIDEO_RESOLUTION)

        if capture_flg:
            shooting_counter += INTERVAL  # 撮影中であれば秒数をカウント

        if SHOOTING_LENGHT <= shooting_counter:  # 撮影最大長さを超えたら
            camera.stop_recoding()               # 撮影を終了し
            shooting_counter = 0                 # カウントを初期化

        if playing_status == 2:   # 撮影が終了していたら
            shooting_counter = 0  # カウントを初期化

        time.sleep(INTERVAL)
        os.remove(PICTURE_PATH)

        excution_counter += INTERVAL
        if EXECUTION_SLOT <= excution_counter:  # 実行スロットを超えたらプログラムを終了
            break
from flask import Flask, render_template, Response
import cv2
import numpy as np
import time
import datetime
import torch
import torchvision
from torchvision import models
import os
from facenet_pytorch import MTCNN
from collections import Counter

num = 3
app = Flask(__name__)

ESC_KEY = 27
FRAME_RATE = 20
SLEEP_TIME = 1 / FRAME_RATE
mtcnn = MTCNN(keep_all=True, device='cuda:0')

########################여기까지가 원본 변수들
device = torch.device('cuda:0')  # 모델과 이미지를 cuda에 넘겨서 분석할거니까 변수 만들어두고

# 폴더 내에 모델을 둘거니까 주소 가져와서
HERE = os.path.dirname(os.path.abspath(__file__))
# 모델명 조인하고 model_path 완성
model_path = os.path.join(HERE, 'new_model_19_122974.pth')
print('표정분류 모델 불러오기')
# 저장된 모델의 checkpoint(저장되던 당시의 모델의 weight 상태)를 저장해두고
checkpoint = torch.load('./new_model_19_122974.pth')
# 모델의 형태는 기본 뼈대가 resnet34에 class(분류해야 할 카테고리)가 4개로 모델을 만들어와서
model = models.resnet34(num_classes=4)
# resnet34 모델과 형태가 좀 다르니까 strict=False로 설정해서 checkpoint에서의 가중치들을 씌워주고
model.load_state_dict(checkpoint, strict=False)
# 위에서 저장한 cuda로 모델을 넘겨서 gpu 연산이 되도록
model.to(device)
print('이미지 변환 설정')

# Face detection XML load and trained model loading
# EMOTIONS = ['Pleasure', 'Embarrassed', 'Sad', 'Rage']
EMOTIONS = ['Pleasure', 'Embarrassed', 'Rage', 'Sad']
model.to(device)
# cv에서 캡쳐한 이미지는 shape가 rgb(224(width),224(height),3)인데 모델이 받아들일 수 있는 이미지의 형태는 tensor형태(1(샘플 수), 3(채널 수), 224(width),224(height))
# 따라서 모델에 넣기 전에 이미지를 변환해줘야됨. 변환해주기 위한 toTensor객체 만들어서(While문 밖으로 빼도 될것 같은데? TODO 정리하고 나서 빼보자)
toTensor = torchvision.transforms.ToTensor()


# CUDA 메모리가 넘칠때 쓰는 코드라고 했는데 좀 더 알아볼것
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

@app.route('/')
def index():
    """Video streaming home page."""
    now = datetime.datetime.now()
    timeString = now.strftime("%Y-%m-%d %H:%M")
    templateData = {
        'title': 'Image Streaming',
        'time': timeString
    }
    return render_template('index.html', **templateData)


def gen_frames():
    camera = cv2.VideoCapture(0)
    time.sleep(0.2)
    lastTime = time.time() * 1000.0
    model.eval()
    times=[]
    emos=[]
    mosts=[]
    mostEmotion=''
    global EMOTIONS

    while True:
        # 시간 췍
        start = time.time()
        # 꿀뷰 창에서 캡쳐한 이미지를 frame에 저장해서
        # frame = capture.screenshot()
        # 카메라에서 이미지만 불러다가

        _, frame = camera.read()
        boxes=mtcnn.detect(frame)
        # 이미지 분석을 위해 모델 input_shape에 맞춰 224*224 크기로 리스아지
        frame = cv2.resize(frame, (600, 600))
        roi = frame.astype("float") / 255.0
        # 좀 더 정확한 분석결과를 얻기 위해 normalize

        # tensorImage로 변환해야하는데 frame을 변환시켜버리면 cv Image가 아니라 문제가 생김
        # 그래서 roi로 하나 복사해서 변환하고 gpu로 넘길 예정

        roi = cv2.resize(roi, (224, 224))

        # 이미지 형태 변환
        roi = toTensor(roi)
        # 샘플 수 자리 만들어서 모양 확실히 맞춰주기!
        roi = roi.unsqueeze(0)
        # 변환했으면 cuda로 넘기자. 위에서 변환했던 float와 torch.float는 형태가 좀 다르니 변환도 같이
        roi = roi.to(device, dtype=torch.float)

        # 모델을 통해 예측한 값이 배열로 나오니까
        preds = []
        # tensorflow와 다르게 torch는 model이 train mode와 evaluation mode가 따로 있음. 그래서 예측하기 전에 evaluation mode로 변환해줘야됨

        # no_grad를 안해주면 파이썬에서는 기본적으로 gradient(기울기)를 계산 해 주는데 evaluation mode에서는 기울기를 계산 할 필요가 없음. 기울기는 학습에서 쓰이는 개념
        # 그래서 eval()을 적용해 준 이후에 torch.no_grad()를 with statement로 감싸준다
        try:
            # box가 검출이 되면 boxes는 리스트, 검출이 안되면 None type object가 되니까 len이 없어서 여기서 오류가 발생하고 바로 except문으로 들어감
            # 박스가 검출 됐을때만 try문 안에서 진행이 되도록 하기 위한 방법!
            a=len(boxes[0])


            with torch.no_grad():
                # torch 모델의 prediction 방식. model(input)
                preds = model(roi)
                # prediction의 결과값 중 필요한 값들만 가져다 저장
                # TODO preds의 형태와 의미 알아볼 것
                _, predicted = torch.max(preds, 1)
                # preds의 가장 큰 값의 인덱스를 가져와서 EMOTIONS 리스트에서 감정 불러오기
                predicted = EMOTIONS[preds[0].argmax(0)]

                # Assign labeling

            # 프레임에 박스 하나 그려 넣기

            # 프레임에 사각 창 띄우기. putText로 얼굴을 여기에 맞추세요 하는것도 괜찮을지도?
            # 와씨 이걸 모델 돌리기 전에 먼저 넣었더니 또 안될뻔 했네 세상에
            # 않이 모델에 안들어가는 이미지에다가 박스 하나 쳤는데 외 모델이 고장나냐?
            # 이미지 분석 하고 cv 작업 할때는 항상 분석 먼저......
            # frame = cv2.rectangle(frame, (200, 200), (400, 400), (0, 0, 255), 1)

            # TODO 거리 구하는 알고리즘 추가하기

            delta = time.time() - start
            # 확률로 변환하려고 하는데 preds 내의 확률값들의 타입이 torch.float라서 np 연산이 안됨.
            # 변환을 해줘도 gpu에서는 numpy 연산이 안되니까 일단 cpu로 옮겨서
            preds = preds.cpu()
            # 데이터형 변환해주고
            preds = preds.numpy()[0]

            # 점수값을 exponential 함수를 통해 확률값으로 변환해주고
            tr_pred = []
            total = 0
            for pred in preds:
                total = total + np.exp(pred)
            for pred in preds:
                tr_pred.append(np.exp(pred) / total)

            # sort 하면서 index도 킵하기
            # 간단한 기능인데 직접 구현해서 줄이 너무 길어짐. TODO 짧게 만들어보자!
            a = 0
            b = 0
            c = 0
            d = 0
            ia = 0
            ib = 0
            ic = 0
            id = 0
            prob_li = []

            # 리스트 내의 확률값들을 정렬하고, 정렬되기 전의 인덱스값을 저장하기 위한 for문
            # 인덱스값은 나중에 감정을 불러오기 위해 같이 저장
            for i in range(4):
                if tr_pred[i] > d:
                    if tr_pred[i] > c:
                        if tr_pred[i] > b:
                            if tr_pred[i] > a:
                                d = c
                                c = b
                                b = a
                                a = tr_pred[i]
                                id = ic
                                ic = ib
                                ib = ia
                                ia = i
                            else:
                                d = c
                                c = b
                                b = tr_pred[i]
                                id = ic
                                ic = ib
                                ib = i
                        else:
                            d = c
                            c = tr_pred[i]
                            id = ic
                            ic = i
                    else:
                        d = tr_pred[i]
                        id = i
            # sorted_list = sort(prob_li.item())
            
            # 감정진단 시작 이후로 3초를 세기 위한 코드
            # while문 안에서 time.time()값을 저장했다가 비교하려고 하면 자꾸 리셋돼버리니까
            # 리스트를 하나 만들어서 그 안에 시간을 쌓고
            # 비교할때는 times의 첫번째(감정진단 시작 했을 당시에 저장한 값) 시간을 이용해서 비교
            timeFlow=time.time()
            times.append(timeFlow)
            emos.append(EMOTIONS[ia])

            # 화면 내의 안면영역을 탐지해서 3초간 리스트에 담는데 그 3초가 지나고 나면
            # 리스트 안의 최빈값을 감정으로 지정하고 그 결과값을 일단은 txt파일로 저장
            # 나중에는 while문도 여기서 끝내는걸로
            if(timeFlow-times[0]>3):
                mostEmotion=Counter(emos).most_common(1)[0]

                emoProbList=[]
                for i in range(len(emos)):
                    emoProbList.append({emos[i]:tr_pred[i]})
                
            # 일단 3초 보여줬으니까 닫고 다시 시작하는 느낌?
            if(timeFlow-times[0]>6):
                times=[]
                mostEmotion=''

                # for emo in emos:
            print('\r', 'preds=', tr_pred, 'predicted=', predicted, 'Latency=', delta, 'timeFlow : ', timeFlow, end='')


            try:
                # 불러온 감정을 frame에 띄워주기
                p_text = [f'{EMOTIONS[ia]} {round(a * 100, 2)}%',
                          f'{EMOTIONS[ib]} {round(b * 100, 2)}%',
                          f'{EMOTIONS[ic]} {round(c * 100, 2)}%',
                          f'{EMOTIONS[id]} {round(d * 100, 2)}%']


                for i in range(4):
                    cv2.putText(frame, p_text[i], (50, 50 + 25 * i),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 7)
                    cv2.putText(frame, p_text[i], (50, 50 + 25 * i),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # TODO 비식별화 할 때 쓰려고 준비하던 코드. 감정분류 결과에 따라 같은 폴더 내의 PNG 이미지를 가져다가 bounding_box에 씌울 예정
                # img_path=f'{predicted}.png'
                # img_array=np.fromfile(img_path, np.uint8)
                # emo=cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                # mask=emo.cv2.CV_8UC(np.uint8)

                # frame=cv2.copyTo(emo, mask, frame)
                
                # 감정 최빈값을 화면에 나타내주기
                cv2.putText(frame, mostEmotion[0], (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 7)
                cv2.putText(frame, mostEmotion[0], (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

            except:
                pass
        except Exception as e:# cv2.imshow('Frame', frame)
            # 중간에 얼굴이 사라지면 리셋하고 다시 처음부터 시작
            delta=0.001
            emos=[]
            times=[]
            mostEmotion=''
            print(e)
        if delta < SLEEP_TIME:
            time.sleep(SLEEP_TIME - delta)

        # esc 누르면 끗
        key = cv2.waitKey(1) & 0xFF
        if key == ESC_KEY:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='192.168.0.28')
    # app.run(host='112.220.111.68')

    # /*현재 상황, 주의사항등 넣어주기 */
    # / *결과값    제이슨타입으로    array에서    json    변환    함수로 * /
# index.html
# 엔진엑스 test.inviz~~~로 띄우기
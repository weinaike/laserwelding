import cv2
import os

path = 'D:/专利论文项目/97.激光焊接项目/第2批数据20240507'
output_dir = '20240507'
avis = os.listdir(path)

for avi in avis:
    if avi.split('.')[-1] == 'avi':
        cap = cv2.VideoCapture(os.path.join(path, avi))
        if(cap.isOpened() == False):
            print("Error opening video stream or file")
            continue


        # if cap.get(cv2.CAP_PROP_FRAME_WIDTH) < 100:
        #     continue
        print(cap.get(cv2.CAP_PROP_FPS))
        print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        os.makedirs(os.path.join(output_dir, avi.split('.')[0]), exist_ok=True)
        i = 0
        while True:
            ret, frame = cap.read()            
            if not ret:
                break
            i += 1
            frame_name = os.path.join(output_dir, avi.split('.')[0], str(i).zfill(5) + '.jpg')
            cv2.imwrite(frame_name, frame)
            sz = frame.shape
            
            cv2.putText(frame, str(i), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


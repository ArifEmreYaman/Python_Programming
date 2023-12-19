import torch
import cv2
import easyocr
from imutils.video import VideoStream
import threading

class Image_Processing:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='alfa.pt', force_reload=True)
        self.model2 = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5l.pt', force_reload=True)
        self.model2.classes = [2]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model2.to(self.device)
        self.reader = easyocr.Reader(['en'], gpu=True)
        self.cap = VideoStream(src=0).start()
        self.bulunan = 0
        
    def plate_reading(self, crop_img):
        result = self.reader.readtext(crop_img)
        if result:
            text = result[0][1]
            text = ''.join(e for e in text if (e.isalnum() or e.isspace()))
            text2 = str(text)
            text2 = text2.upper()
            text2 = text2.split(" ")
            self.text3 = ""
            for i in text2:
                if i == " ":
                    pass
                else:
                    self.text3 += i
               
            print(self.text3)
            
            

    def main(self):
        while True:
            frame = self.cap.read()
            if frame is None:
                continue
            frame = cv2.resize(frame, (900, 780))
            result = self.model(frame)
            result2 = self.model2(frame)
            boxes = result.xyxy[0].cpu().numpy()
            boxes2 = result2.xyxy[0].cpu().numpy()
            overlay = frame.copy()
            alpha = 0.7
            beta = 0.1

            for box2 in boxes2:
                xa1, ya1, xa2, ya2, conf2, class_idx2 = box2

                if conf2 > beta:
                    cv2.rectangle(overlay, (int(xa1), int(ya1)), (int(xa2), int(ya2), (0, 0, 255), 1))
                    cv2.putText(overlay, "araba", (int(xa1), int(ya1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    ytx = int(xa1 + xa2 / 2)
                    cv2.circle(frame, (int(xa1 + xa2 / 2), int(ya1 + ya2 / 2), 10, (255, 0, 0), -1))
                    sed = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                    frame = sed

           

            for box in boxes:
                x1, y1, x2, y2, conf, class_idx = box

                if conf > beta:
                    crop_img = frame[int(y1):int(y2), int(x1):int(x2)]
                    if self.bulunan == 0:
                        self.bulunan=1
                        okuma = threading.Thread(target=self.plate_reading, args=(crop_img,))
                        okuma.start()

                    cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
                    cv2.putText(overlay, self.text3, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
                    sed = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                    frame = sed

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    program = Image_Processing()
    program.main()

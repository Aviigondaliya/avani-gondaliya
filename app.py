'''
from flask import Flask, render_template, Response, request
from paddleocr import PaddleOCR, draw_ocr
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)

# Define detection and recognition algorithms
det_algo = {
    "east": "/PaddleOCR/inference/det_r50_east",
    "sast": "/PaddleOCR/inference/det_sast_ic15",
    "pse": "/PaddleOCR/inference/det_pse",
    "fce": "/PaddleOCR/inference/det_fce"
}

rec_algo = {
    "srn": "/PaddleOCR/inference/rec_srn",
    "nrtr": "/PaddleOCR/inference/rec_mtb_nrtr",
    "svtr": "/PaddleOCR/inference/rec_svtr_tiny_stn_en",
    "abi": "/PaddleOCR/inference/rec_r45_abinet"
}

# Function to perform OCR on video frames
def start_ocr(det_algo, rec_algo):
    ocr = PaddleOCR(det_model_dir=det_algo,
                    rec_model_dir=rec_algo,
                    use_angle_cls=True,
                    use_gpu=False,
                    rec_char_type='en')

    cap = cv2.VideoCapture(0)

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()

        # Perform OCR on the captured frame
        result = ocr.ocr(frame, cls=True)

        # Check if OCR result is not None and has elements
        if result and len(result) > 0 and result[0] is not None:
            result = result[0]  # Assuming you are interested in the first result
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes = [line[0] for line in result]
            txts = [line[1][0] for line in result]
            scores = [line[1][1] for line in result]
            im_show = draw_ocr(image, boxes, txts, scores, font_path=r'C:\avani\TRP\PaddleOCR\doc\fonts\simfang.ttf')
            frame_with_ocr = cv2.cvtColor(np.array(im_show), cv2.COLOR_RGB2BGR)

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame_with_ocr)
            frame_bytes = buffer.tobytes()

            # Yield the frame in a byte string
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_ocr')
def video_feed():
    det_algo = request.args.get('det_algo')
    rec_algo = request.args.get('rec_algo')

    if det_algo is None or rec_algo is None:
        return "Error: Please provide both detection and recognition algorithms."

    return Response(start_ocr(det_algo=det_algo, rec_algo=rec_algo),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)

'''
from flask import Flask, render_template, Response, request
from paddleocr import PaddleOCR, draw_ocr
import cv2
import numpy as np
from PIL import Image


app = Flask(__name__)

# Define detection and recognition algorithms
det_algo = {
    "east": "/inference/det_r50_east",
    "sast": "/inference/det_sast_ic15",
    "pse": "/inference/det_pse",
    "fce": "/inference/det_fce"
}

rec_algo = {
    "srn": "/inference/rec_srn",
    "nrtr": "/inference/rec_mtb_nrtr",
    "svtr": "/inference/rec_svtr_tiny_stn_en",
    "abi": "/inference/rec_r45_abinet"
}

# Function to perform OCR on video frames
def start_ocr(video_source, det_algo, rec_algo):
    ocr = PaddleOCR(det_model_dir=det_algo,
                    rec_model_dir=rec_algo,
                    use_angle_cls=True,
                    use_gpu=False,
                    rec_char_type='en')

    cap = cv2.VideoCapture(video_source)
    print(video_source)

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()

        # Perform OCR on the captured frame
        result = ocr.ocr(frame, cls=True)

        # Check if OCR result is not None and has elements
        if result and len(result) > 0 and result[0] is not None:
            result = result[0]  # Assuming you are interested in the first result
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes = [line[0] for line in result]
            txts = [line[1][0] for line in result]
            scores = [line[1][1] for line in result]
            im_show = draw_ocr(image, boxes, txts, scores, font_path=r'/PaddleOCR/doc/fonts/simfang.ttf')
            frame_with_ocr = cv2.cvtColor(np.array(im_show), cv2.COLOR_RGB2BGR)

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame_with_ocr)
            frame_bytes = buffer.tobytes()

            # Yield the frame in a byte string
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_ocr', methods=['POST'])
def video_feed():
    video_source = request.form.get('video_source')
    det_algo = request.form.get('det_algo')
    rec_algo = request.form.get('rec_algo')

    if video_source is None or det_algo is None or rec_algo is None:
        return "Error: Please provide video source, detection, and recognition algorithms."

    return Response(start_ocr(video_source, det_algo=det_algo, rec_algo=rec_algo),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)

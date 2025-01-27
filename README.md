# Image Object Recognition Trainer

## Stage 1: Annotation


```python
import os
import cv2
import numpy as np
import nest_asyncio

nest_asyncio.apply()
```


```python
training_data = "./training_data"
image_width = 1080
image_height = 1920
```


```python
images = [f for f in os.listdir(training_data) if f[-4:] == ".jpg"]
len(images)
```




    15112




```python
def is_daylight(image, brightness_threshold=100, blue_threshold=30):
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Extract the Value (brightness) channel
    brightness = hsv[:, :, 2]

    # Calculate average brightness
    avg_brightness = np.mean(brightness)

    # Convert to RGB and calculate average blue intensity
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    blue_channel = rgb[:, :, 2]
    avg_blue_intensity = np.mean(blue_channel)

    # Determine daylight
    is_day = avg_brightness > brightness_threshold and avg_blue_intensity > blue_threshold

    return is_day, avg_brightness, avg_blue_intensity
```


```python
import json
from dataclasses import dataclass
from io import BytesIO
import time

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse, Response


app = FastAPI()
annotations = []
annotation_index = 1500


@dataclass
class Annotation:
    coords: dict[str, tuple[int, int]]
    value: str

    def __post_init__(self):
        if self.value == "":
            self.value = "0"


def genDiceCam():
    global captures, lastCaptureTime
    while True:
        orig = cc.capture()['Dice']
        camImage = orig.copy()
        cv2.putText(camImage, f'Capturing: {captureInProgress}',
            (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3, cv2.LINE_AA)
        annotationStr = ""
        for a in annotations:
            coords = json.loads(a['coords'])
            sp = [int(d) for d in coords['p0']]
            ep = [int(d) for d in coords['p1']]
            if captureInProgress:
                value = a['value']
                w, h = [d1 - d0 for d1, d0 in zip(ep, sp)]
                annotationStr += f"{value} {sp[0]} {sp[1]} {w} {h}\n"
            cv2.rectangle(camImage, sp, ep, (0, 255, 100), 3)
        ret, camImage = cv2.imencode('.jpg', camImage)

        if captureInProgress and (lastCaptureTime is None or (datetime.utcnow() - lastCaptureTime).total_seconds() > 30):
            imageName = f"diceCollector_Cap{datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')}"
            cv2.imwrite(f"captures/{imageName}.jpg", orig)
            with open(f"captures/{imageName}.txt", "w") as f:
                f.write(annotationStr)
            lastCaptureTime = datetime.utcnow()
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')


@app.get('/htmx.min.js')
async def getHTMX():
    return FileResponse("webapp_templates/htmx.min.js", media_type="application/javascript")


@app.get('/bootstrap.min.js')
async def getBootstrapJS():
    return FileResponse("webapp_templates/bootstrap.min.js", media_type="application/javascript")


@app.get('/bootstrap.min.css')
async def getBootstrapCSS():
    return FileResponse("webapp_templates/bootstrap.min.css", media_type="text/css")


@app.get('/')
async def getImageAnnotator():
    read_annotations(images[annotation_index])
    return FileResponse("webapp_templates/image_annotator.html", media_type="text/html")


@app.get('/annotator')
async def getImageAnnotator():
    return FileResponse("webapp_templates/annotator.html", media_type="text/html", headers={"Expires": "0"})


def render_annotations():
    with open("webapp_templates/annotation.html") as f:
        template = f.read()
    return "\n".join([template.format(
        annotation_index=idx,
        annotation_coordinates=json.dumps(annotation.coords),
        annotation_value=annotation.value)
    for idx, annotation in enumerate(annotations)])


def write_annotations(image_path):
    image_annotation_path = f"{training_data}/{image_path[:-4]}.txt"
    image_annotations = []
    for annotation in annotations:
        p0 = annotation.coords['p0']
        p1 = annotation.coords['p1']
        p0_x, p0_y = p0
        p1_x, p1_y = p1
        width, height = [abs(float(d1) - float(d0)) for d0, d1 in zip([p0_x, p0_y], [p1_x, p1_y])]
        c_x = (float(p0_x) + width / 2) / image_width
        c_y = (float(p0_y) + height / 2) / image_height
        width /= image_width
        height /= image_height
        print(f"Writing notation from {annotation.coords}: {c_x} {c_y} {width} {height}")
        image_annotations.append(f"{annotation.value} {c_x} {c_y} {width} {height}\n")
    with open(image_annotation_path, "w") as f:
        f.write("\n".join(image_annotations))


def read_annotations(image_path):
    image_annotation_path = f"{training_data}/{image_path[:-4]}.txt"
    global annotations
    annotations = []
    
    if not os.path.exists(image_annotation_path):
        return

    with open(image_annotation_path) as f:
        stored_image_annotations = f.read()
    print(f"Reading {stored_image_annotations}")
    for anno_line in stored_image_annotations.split("\n"):
        try:
            value, c_x, c_y, width, height = anno_line.split(" ")
        except:
            continue
        c_x = float(c_x)
        c_y = float(c_y)
        width = float(width)
        height = float(height)
        c_x *= image_width
        c_y *= image_height
        width *= image_width
        height *= image_height
        p0 = [c_x - width / 2, c_y - height / 2]
        p1 = [c_x + width / 2, c_y + height / 2]
        annotations.append(Annotation(coords={"p0": p0, "p1": p1}, value=value))


@app.get("/annotation_index")
async def getAnnotationIndex():
    return f"{annotation_index} of {len(images) - 1}"


@app.post('/annotations')
async def setAnnotations(request: Request):
    # Path to your image file
    raw_body = await request.body()  # Get the raw bytes of the request body
    body_str = raw_body.decode("utf-8")  # Decode bytes to a string
    global annotations
    annotations = [Annotation(**a) for a in json.loads(body_str)]  # Parse the JSON data
    print(f"Updated annotations: {annotations}")
    write_annotations(images[annotation_index])
    return Response(render_annotations(), media_type="text/html")


@app.get('/annotations')
async def getAnnotations(request: Request):
    print(f"Current annotations: {annotations}")
    return Response(render_annotations(), media_type="text/html")


@app.get('/clearannotations')
async def clearAnnotations():
    global captureInProgress, annotations
    captureInProgress = False
    annotations = []
    return Response("Success!", media_type="text/html")

@app.get("/image_list")
async def get_image_list():
    rendered = ""
    for idx, image in enumerate(images):
        if idx == annotation_index:
            image = f"<b>{image}<b>"
        rendered = f"{rendered}\n{image}<br>"
    return Response(rendered, media_type="text/html")


def generate_image_stream():
    while True:
        image_name = images[annotation_index]
        image_path = f"training_data/{image_name}"
    
        image = cv2.imread(image_path)
        image_is_daylight, _, _ = is_daylight(image)
        # label is_daylight
        text = f"Daylight: {image_is_daylight:5}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 4
        color = (255, 255, 255)  # White color in BGR
        thickness = 2
        
        # Get the text size to calculate the position dynamically
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
        # Positions for different corners
        top_left = (10, text_height + 20)  # Add some padding for clarity
        
        # Add text to each corner
        cv2.putText(image, text, top_left, font, font_scale, color, thickness)

        for annotation in annotations:
            points = annotation.coords
            # Convert points to integer tuples for OpenCV
            p0 = tuple(map(int, points["p0"]))
            p1 = tuple(map(int, points["p1"]))
            
            # Draw the rectangle
            color = (0, 255, 0)  # Green color in BGR
            thickness = 2        # Thickness of the rectangle border
            cv2.rectangle(image, p0, p1, color, thickness)

        # Encode the image as JPEG
        _, img_encoded = cv2.imencode('.jpg', image)

        # Yield the image bytes
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + img_encoded.tobytes() + b"\r\n")

        # Wait before sending the next frame (simulate dynamic updates)
        time.sleep(0.2)

    
@app.get("/image")
async def get_image():
    return StreamingResponse(generate_image_stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/next")
async def next_image():
    global annotation_index
    annotation_index = min(len(images), annotation_index + 1)
    print(f"Updated annotation_index: {annotation_index}")
    read_annotations(images[annotation_index])
    return FileResponse("webapp_templates/annotator.html", media_type="text/html", headers={"Expires": "0"})
    
@app.get("/prev")
async def prev_image():
    global annotation_index
    annotation_index = max(0, annotation_index - 1)
    print(f"Updated annotation_index: {annotation_index}")
    read_annotations(images[annotation_index])
    return FileResponse("webapp_templates/annotator.html", media_type="text/html", headers={"Expires": "0"})
    
@app.get("/image/{idx}")
async def set_image(idx: int):
    global annotation_index
    annotation_index = min(len(images), max(0, idx))
    print(f"Updated annotation_index: {annotation_index}")
    read_annotations(images[annotation_index])
    return FileResponse("webapp_templates/annotator.html", media_type="text/html", headers={"Expires": "0"})
```


```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
```

    INFO:     Started server process [338071]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://127.0.0.1:5000 (Press CTRL+C to quit)


    INFO:     127.0.0.1:50258 - "GET / HTTP/1.1" 200 OK
    INFO:     127.0.0.1:50258 - "GET /bootstrap.min.js.map HTTP/1.1" 404 Not Found
    INFO:     127.0.0.1:50274 - "GET /bootstrap.min.css.map HTTP/1.1" 404 Not Found
    INFO:     127.0.0.1:50258 - "GET /annotator HTTP/1.1" 200 OK
    INFO:     127.0.0.1:50258 - "GET /image HTTP/1.1" 200 OK
    INFO:     127.0.0.1:50274 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:50274 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:50274 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotations: [Annotation(coords={'p0': [1216.7441860465117, 864.3491379310345], 'p1': [1376.7441860465117, 990.9698275862069]}, value='0')]
    Writing notation from {'p0': [1216.7441860465117, 864.3491379310345], 'p1': [1376.7441860465117, 990.9698275862069]}: 1.2006890611541774 0.4831559806034483 0.14814814814814814 0.06594827586206892
    INFO:     127.0.0.1:50274 - "POST /annotations HTTP/1.1" 200 OK
    Updated annotation_index: 1501
    INFO:     127.0.0.1:50274 - "GET /next HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:50274 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:50284 - "GET /annotation_index HTTP/1.1" 200 OK
    INFO:     127.0.0.1:50288 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 1500
    INFO:     127.0.0.1:50288 - "GET /prev HTTP/1.1" 200 OK
    INFO:     127.0.0.1:50288 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:50284 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:50274 - "GET /image_list HTTP/1.1" 200 OK


    INFO:     Shutting down
    INFO:     Waiting for connections to close. (CTRL+C to force quit)
    INFO:     Finished server process [338071]



```python

```

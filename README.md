# Image Object Recognition Trainer

## Stage 1: Annotation


```python
import os
import cv2
import numpy as np
```


```python
training_data = "./daylight_images"
image_width = 1080
image_height = 1920

images = [f for f in os.listdir(training_data) if f[-4:] == ".jpg"]
len(images)
```




    7020




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
annotation_index = 0


@dataclass
class Annotation:
    coords: dict[str, tuple[int, int]]
    value: str

    def __post_init__(self):
        if self.value == "":
            self.value = "0"


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
            image = f"<b>{image}</b>"
        rendered = f'{rendered}\n<a href="/image/{idx}">{image}</a><br>'
    return Response(rendered, media_type="text/html")


def generate_image_stream():
    while True:
        image_name = images[annotation_index]
        image_path = f"training_data/{image_name}"
    
        image = cv2.imread(image_path)

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
    return FileResponse("webapp_templates/image_annotator.html", media_type="text/html", headers={"Expires": "0"})
```


```python
if __name__ == "__main__":
    import uvicorn
    import nest_asyncio
    nest_asyncio.apply()
    uvicorn.run(app, host="127.0.0.1", port=5000)
```

    INFO:     Started server process [357832]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://127.0.0.1:5000 (Press CTRL+C to quit)


    INFO:     127.0.0.1:51898 - "GET /?annotation_index=50 HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51898 - "GET /annotator HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51898 - "GET /image HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51914 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:51914 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51914 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 50
    INFO:     127.0.0.1:51914 - "GET /image/50 HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51914 - "GET /annotator HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51914 - "GET /image HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51918 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:51920 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51920 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 79
    INFO:     127.0.0.1:51920 - "GET /image/79 HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51920 - "GET /annotator HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51920 - "GET /image HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51918 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:51922 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51922 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 80
    INFO:     127.0.0.1:51922 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51922 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:51922 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51922 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotations: [Annotation(coords={'p0': [1007.5000000000001, 910], 'p1': [1205, 1010]}, value='1')]
    Writing notation from {'p0': [1007.5000000000001, 910], 'p1': [1205, 1010]}: 1.0243055555555556 0.5 0.18287037037037027 0.052083333333333336
    INFO:     127.0.0.1:50132 - "POST /annotations HTTP/1.1" 200 OK
    Updated annotation_index: 81
    INFO:     127.0.0.1:50132 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:50132 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:50144 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:50132 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotations: [Annotation(coords={'p0': [1137.5, 907.5], 'p1': [1310, 997.5]}, value='0')]
    Writing notation from {'p0': [1137.5, 907.5], 'p1': [1310, 997.5]}: 1.1331018518518519 0.49609375 0.1597222222222222 0.046875
    INFO:     127.0.0.1:51206 - "POST /annotations HTTP/1.1" 200 OK
    Updated annotation_index: 80
    Reading 1 1.0243055555555556 0.5 0.18287037037037027 0.052083333333333336
    
    INFO:     127.0.0.1:51206 - "GET /prev HTTP/1.1" 200 OK
    Current annotations: [Annotation(coords={'p0': [1007.5, 910.0], 'p1': [1205.0, 1010.0]}, value='1')]
    INFO:     127.0.0.1:51206 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51206 - "GET /annotation_index HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51206 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 81
    Reading 0 1.1331018518518519 0.49609375 0.1597222222222222 0.046875
    
    INFO:     127.0.0.1:51206 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51206 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: [Annotation(coords={'p0': [1137.5, 907.5], 'p1': [1310.0, 997.5]}, value='0')]
    INFO:     127.0.0.1:51220 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51222 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 0
    INFO:     127.0.0.1:54062 - "GET /image/0 HTTP/1.1" 200 OK
    INFO:     127.0.0.1:54062 - "GET /annotator HTTP/1.1" 200 OK
    INFO:     127.0.0.1:54062 - "GET /image HTTP/1.1" 200 OK
    INFO:     127.0.0.1:54064 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:54078 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:54084 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 1
    INFO:     127.0.0.1:54084 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:54084 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:54078 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:54064 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 2
    INFO:     127.0.0.1:54064 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:54064 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:54078 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:54084 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 3
    INFO:     127.0.0.1:54084 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:54084 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:54078 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:54064 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 4
    INFO:     127.0.0.1:54064 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:54064 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:54078 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:54084 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 5
    INFO:     127.0.0.1:54084 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:54084 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:54078 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:54064 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 6
    Reading 0 1.6157407407407407 0.5201822916666666 0.2916666666666667 0.055989583333333336
    
    INFO:     127.0.0.1:54064 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:54064 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: [Annotation(coords={'p0': [1587.5, 944.9999999999999], 'p1': [1902.5, 1052.5]}, value='0')]
    INFO:     127.0.0.1:54078 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:54084 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 7
    INFO:     127.0.0.1:54084 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:54084 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:54078 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:54064 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 8
    INFO:     127.0.0.1:54064 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:54064 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:54078 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:54084 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 9
    Reading 0 1.3946759259259258 0.5 0.2152777777777778 0.049479166666666664
    
    INFO:     127.0.0.1:54084 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:54084 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: [Annotation(coords={'p0': [1389.9999999999998, 912.5], 'p1': [1622.4999999999998, 1007.5]}, value='0')]
    INFO:     127.0.0.1:54078 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:54064 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 10
    INFO:     127.0.0.1:46830 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:46830 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:46830 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:46830 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 9
    Reading 0 1.3946759259259258 0.5 0.2152777777777778 0.049479166666666664
    
    INFO:     127.0.0.1:46830 - "GET /prev HTTP/1.1" 200 OK
    INFO:     127.0.0.1:46830 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: [Annotation(coords={'p0': [1389.9999999999998, 912.5], 'p1': [1622.4999999999998, 1007.5]}, value='0')]
    INFO:     127.0.0.1:46844 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:46852 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 10
    INFO:     127.0.0.1:33494 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:33494 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:33494 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:33494 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 9
    Reading 0 1.3946759259259258 0.5 0.2152777777777778 0.049479166666666664
    
    INFO:     127.0.0.1:33494 - "GET /prev HTTP/1.1" 200 OK
    INFO:     127.0.0.1:33494 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: [Annotation(coords={'p0': [1389.9999999999998, 912.5], 'p1': [1622.4999999999998, 1007.5]}, value='0')]
    INFO:     127.0.0.1:33508 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:33520 - "GET /image_list HTTP/1.1" 200 OK
    INFO:     127.0.0.1:33520 - "GET /clearannotations HTTP/1.1" 200 OK
    Updated annotations: [Annotation(coords={'p0': [1387.5, 912.5], 'p1': [1625, 1012.5]}, value='1')]
    Writing notation from {'p0': [1387.5, 912.5], 'p1': [1625, 1012.5]}: 1.3946759259259258 0.5013020833333334 0.2199074074074074 0.052083333333333336
    INFO:     127.0.0.1:33520 - "POST /annotations HTTP/1.1" 200 OK
    Updated annotation_index: 10
    INFO:     127.0.0.1:33520 - "GET /next HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:33520 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:33520 - "GET /annotation_index HTTP/1.1" 200 OK
    INFO:     127.0.0.1:33520 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 9
    Reading 1 1.3946759259259258 0.5013020833333334 0.2199074074074074 0.052083333333333336
    
    INFO:     127.0.0.1:33520 - "GET /prev HTTP/1.1" 200 OK
    INFO:     127.0.0.1:33520 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: [Annotation(coords={'p0': [1387.4999999999998, 912.5000000000001], 'p1': [1624.9999999999998, 1012.5000000000001]}, value='1')]
    INFO:     127.0.0.1:41898 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:41912 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 10
    INFO:     127.0.0.1:41912 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:41912 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:41898 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:33520 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 11
    Reading 0 0.7766203703703703 0.486328125 0.5439814814814815 0.12369791666666667
    
    INFO:     127.0.0.1:33520 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:33520 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: [Annotation(coords={'p0': [545.0, 815.0], 'p1': [1132.5, 1052.5]}, value='0')]
    INFO:     127.0.0.1:41898 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:41912 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 12
    INFO:     127.0.0.1:41912 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:41912 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:41898 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:33520 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 13
    INFO:     127.0.0.1:33520 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:33520 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:41898 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:41912 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 14
    INFO:     127.0.0.1:41912 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:41912 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:41898 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:33520 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 15
    INFO:     127.0.0.1:33520 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:33520 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:41898 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:41912 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 16
    INFO:     127.0.0.1:41912 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:41912 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:41898 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:33520 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 17
    INFO:     127.0.0.1:33520 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:33520 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:41898 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:41912 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 18
    INFO:     127.0.0.1:41912 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:41912 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:41898 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:33520 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 19
    INFO:     127.0.0.1:33520 - "GET /next HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:41898 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:33520 - "GET /annotation_index HTTP/1.1" 200 OK
    INFO:     127.0.0.1:41912 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 20
    INFO:     127.0.0.1:41912 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:41912 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:33520 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:41898 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 21
    INFO:     127.0.0.1:41898 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:41898 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:33520 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:41912 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 22
    Reading 0 0.2222222222222222 0.4921875 0.11574074074074074 0.036458333333333336
    
    INFO:     127.0.0.1:41912 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:41912 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: [Annotation(coords={'p0': [177.5, 910.0], 'p1': [302.5, 980.0]}, value='0')]
    INFO:     127.0.0.1:33520 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:41898 - "GET /image_list HTTP/1.1" 200 OK
    INFO:     127.0.0.1:41898 - "GET /clearannotations HTTP/1.1" 200 OK
    Updated annotations: [Annotation(coords={'p0': [180, 910], 'p1': [315, 977.5]}, value='1')]
    Writing notation from {'p0': [180, 910], 'p1': [315, 977.5]}: 0.22916666666666666 0.4915364583333333 0.125 0.03515625
    INFO:     127.0.0.1:45032 - "POST /annotations HTTP/1.1" 200 OK
    Updated annotation_index: 23
    INFO:     127.0.0.1:51358 - "GET /next HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:51358 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51358 - "GET /annotation_index HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51358 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 24
    Reading 0 1.494212962962963 0.5123697916666666 0.5578703703703703 0.08203125000000006
    
    INFO:     127.0.0.1:51358 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51358 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: [Annotation(coords={'p0': [1312.5, 904.9999999999998], 'p1': [1915.0, 1062.5]}, value='0')]
    INFO:     127.0.0.1:51370 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51378 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 25
    Reading 0 1.1956018518518519 0.498046875 0.2013888888888889 0.05859375
    
    INFO:     127.0.0.1:51378 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51378 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: [Annotation(coords={'p0': [1182.5, 900.0], 'p1': [1400.0, 1012.5]}, value='0')]
    INFO:     127.0.0.1:51370 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51358 - "GET /image_list HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51358 - "GET /clearannotations HTTP/1.1" 200 OK
    Updated annotations: [Annotation(coords={'p0': [1190, 912.5], 'p1': [1405, 1010]}, value='1')]
    Writing notation from {'p0': [1190, 912.5], 'p1': [1405, 1010]}: 1.2013888888888888 0.5006510416666666 0.19907407407407407 0.05078125
    INFO:     127.0.0.1:44660 - "POST /annotations HTTP/1.1" 200 OK
    Updated annotation_index: 26
    Reading 0 2.0636574074074074 0.484375 0.8356481481481481 0.13541666666666666
    
    INFO:     127.0.0.1:44660 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:44660 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: [Annotation(coords={'p0': [1777.5, 800.0], 'p1': [2680.0, 1060.0]}, value='0')]
    INFO:     127.0.0.1:44660 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:44660 - "GET /image_list HTTP/1.1" 200 OK
    INFO:     127.0.0.1:44660 - "GET /clearannotations HTTP/1.1" 200 OK
    Updated annotations: [Annotation(coords={'p0': [867.5, 847.5], 'p1': [1752.5, 1060]}, value='0')]
    Writing notation from {'p0': [867.5, 847.5], 'p1': [1752.5, 1060]}: 1.212962962962963 0.4967447916666667 0.8194444444444444 0.11067708333333333
    INFO:     127.0.0.1:51992 - "POST /annotations HTTP/1.1" 200 OK
    Updated annotation_index: 27
    Reading 0 0.7650462962962963 0.4928385416666667 0.298611111111111 0.09505208333333333
    
    0 0.6481481481481481 0.4654947916666667 0.3425925925925926 0.07682291666666667
    
    INFO:     127.0.0.1:51992 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51992 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: [Annotation(coords={'p0': [665.0, 855.0], 'p1': [987.5, 1037.5]}, value='0'), Annotation(coords={'p0': [515.0, 820.0], 'p1': [885.0, 967.5]}, value='0')]
    INFO:     127.0.0.1:51996 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51992 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 28
    Reading 0 0.7118055555555556 0.4915364583333333 0.12268518518518519 0.037760416666666664
    
    INFO:     127.0.0.1:51992 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51992 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: [Annotation(coords={'p0': [702.5, 907.5], 'p1': [835.0, 980.0]}, value='0')]
    INFO:     127.0.0.1:51996 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:47534 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 29
    Reading 0 0.1886574074074074 0.4733072916666667 0.2569444444444444 0.06119791666666661
    
    INFO:     127.0.0.1:47534 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:47534 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: [Annotation(coords={'p0': [65.0, 850.0], 'p1': [342.5, 967.5]}, value='0')]
    INFO:     127.0.0.1:51996 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51992 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 30
    Reading 0 0.6608796296296297 0.4856770833333333 0.13657407407407407 0.044270833333333336
    
    INFO:     127.0.0.1:51992 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51992 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: [Annotation(coords={'p0': [640.0, 890.0], 'p1': [787.5, 975.0]}, value='0')]
    INFO:     127.0.0.1:51996 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:47534 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 31
    INFO:     127.0.0.1:47534 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:47534 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:51996 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51992 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 32
    INFO:     127.0.0.1:51992 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51992 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:51996 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:47534 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 33
    Reading 0 1.349537037037037 0.5130208333333334 0.28703703703703703 0.08072916666666667
    
    INFO:     127.0.0.1:47534 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:47534 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: [Annotation(coords={'p0': [1302.5, 907.5000000000001], 'p1': [1612.5, 1062.5]}, value='0')]
    INFO:     127.0.0.1:51996 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51992 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 34
    INFO:     127.0.0.1:51992 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51992 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:51996 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:47534 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 35
    INFO:     127.0.0.1:47534 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:47534 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:51996 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51992 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 36
    INFO:     127.0.0.1:51992 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51992 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:51996 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:47534 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 37
    INFO:     127.0.0.1:47534 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:47534 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:51996 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51992 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 38
    INFO:     127.0.0.1:51992 - "GET /next HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:51996 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51992 - "GET /annotation_index HTTP/1.1" 200 OK
    INFO:     127.0.0.1:47534 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 39
    INFO:     127.0.0.1:47534 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:47534 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:51992 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51996 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 40
    INFO:     127.0.0.1:51996 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51996 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:51992 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:47534 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 41
    INFO:     127.0.0.1:47534 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:47534 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:51992 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51996 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 42
    INFO:     127.0.0.1:51996 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51996 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:51992 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:47534 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 43
    Reading 0 0.06944444444444445 0.4772135416666667 0.10648148148148148 0.037760416666666664
    
    INFO:     127.0.0.1:47534 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:47534 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: [Annotation(coords={'p0': [17.5, 880.0], 'p1': [132.5, 952.5]}, value='0')]
    INFO:     127.0.0.1:51992 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51996 - "GET /image_list HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51996 - "GET /clearannotations HTTP/1.1" 200 OK
    Updated annotations: [Annotation(coords={'p0': [35, 892.5], 'p1': [127.5, 955]}, value='1')]
    Writing notation from {'p0': [35, 892.5], 'p1': [127.5, 955]}: 0.07523148148148148 0.4811197916666667 0.08564814814814815 0.032552083333333336
    INFO:     127.0.0.1:51996 - "POST /annotations HTTP/1.1" 200 OK
    Updated annotation_index: 44
    Reading 0 1.1944444444444444 0.517578125 0.28703703703703703 0.07421875
    
    INFO:     127.0.0.1:51996 - "GET /next HTTP/1.1" 200 OK
    Current annotations: [Annotation(coords={'p0': [1135.0, 922.5], 'p1': [1445.0, 1065.0]}, value='0')]
    INFO:     127.0.0.1:51996 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51996 - "GET /annotation_index HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51996 - "GET /image_list HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51996 - "GET /clearannotations HTTP/1.1" 200 OK
    Updated annotations: [Annotation(coords={'p0': [1120, 930], 'p1': [1427.5, 1047.5]}, value='1')]
    Writing notation from {'p0': [1120, 930], 'p1': [1427.5, 1047.5]}: 1.1793981481481481 0.5149739583333334 0.2847222222222222 0.061197916666666664
    INFO:     127.0.0.1:51996 - "POST /annotations HTTP/1.1" 200 OK
    Updated annotation_index: 45
    INFO:     127.0.0.1:51996 - "GET /next HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:51996 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:43456 - "GET /annotation_index HTTP/1.1" 200 OK
    INFO:     127.0.0.1:43462 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 46
    INFO:     127.0.0.1:43462 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:43462 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:43456 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51996 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 47
    Reading 0 0.8379629629629629 0.4967447916666667 0.16203703703703715 0.05078125000000006
    
    INFO:     127.0.0.1:51996 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51996 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: [Annotation(coords={'p0': [817.4999999999998, 905.0], 'p1': [992.5, 1002.5]}, value='0')]
    INFO:     127.0.0.1:43456 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:43462 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 48
    INFO:     127.0.0.1:43462 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:43462 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:43456 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51996 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 47
    Reading 0 0.8379629629629629 0.4967447916666667 0.16203703703703715 0.05078125000000006
    
    INFO:     127.0.0.1:51996 - "GET /prev HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51996 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: [Annotation(coords={'p0': [817.4999999999998, 905.0], 'p1': [992.5, 1002.5]}, value='0')]
    INFO:     127.0.0.1:43456 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:43462 - "GET /image_list HTTP/1.1" 200 OK
    INFO:     127.0.0.1:43462 - "GET /clearannotations HTTP/1.1" 200 OK
    Updated annotations: [Annotation(coords={'p0': [825, 902.5], 'p1': [982.5, 1000]}, value='1')]
    Writing notation from {'p0': [825, 902.5], 'p1': [982.5, 1000]}: 0.8368055555555556 0.4954427083333333 0.14583333333333334 0.05078125
    INFO:     127.0.0.1:43462 - "POST /annotations HTTP/1.1" 200 OK
    Updated annotation_index: 48
    INFO:     127.0.0.1:43462 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:43462 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:43462 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:43462 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 49
    INFO:     127.0.0.1:38794 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:38794 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:38798 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:38794 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 50
    INFO:     127.0.0.1:38794 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:38794 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:38798 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:43066 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 51
    INFO:     127.0.0.1:43066 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:43066 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:38798 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:38794 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 52
    INFO:     127.0.0.1:38794 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:38794 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:38798 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:43066 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 53
    INFO:     127.0.0.1:43066 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:43066 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:38798 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:38794 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 54
    INFO:     127.0.0.1:38794 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:38794 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:38798 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:43066 - "GET /image_list HTTP/1.1" 200 OK
    INFO:     127.0.0.1:43066 - "GET /clearannotations HTTP/1.1" 200 OK
    Updated annotations: [Annotation(coords={'p0': [970.0000000000001, 922.5], 'p1': [1225, 1040]}, value='1')]
    Writing notation from {'p0': [970.0000000000001, 922.5], 'p1': [1225, 1040]}: 1.0162037037037037 0.5110677083333334 0.236111111111111 0.061197916666666664
    INFO:     127.0.0.1:55054 - "POST /annotations HTTP/1.1" 200 OK
    Updated annotations: [Annotation(coords={'p0': [970.0000000000001, 922.5], 'p1': [1225, 1040]}, value='1'), Annotation(coords={'p0': [1145, 907.5], 'p1': [1342.5, 1002.5]}, value='1')]
    Writing notation from {'p0': [970.0000000000001, 922.5], 'p1': [1225, 1040]}: 1.0162037037037037 0.5110677083333334 0.236111111111111 0.061197916666666664
    Writing notation from {'p0': [1145, 907.5], 'p1': [1342.5, 1002.5]}: 1.1516203703703705 0.4973958333333333 0.18287037037037038 0.049479166666666664
    INFO:     127.0.0.1:55062 - "POST /annotations HTTP/1.1" 200 OK
    Updated annotation_index: 55
    INFO:     127.0.0.1:55062 - "GET /next HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:55062 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:55062 - "GET /annotation_index HTTP/1.1" 200 OK
    INFO:     127.0.0.1:55062 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotations: [Annotation(coords={'p0': [47.5, 942.5], 'p1': [0, 890]}, value='1')]
    Writing notation from {'p0': [47.5, 942.5], 'p1': [0, 890]}: 0.06597222222222222 0.5045572916666666 0.04398148148148148 0.02734375
    INFO:     127.0.0.1:55070 - "POST /annotations HTTP/1.1" 200 OK
    Updated annotation_index: 56
    INFO:     127.0.0.1:55070 - "GET /next HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:55070 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:55082 - "GET /annotation_index HTTP/1.1" 200 OK
    INFO:     127.0.0.1:55070 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 57
    INFO:     127.0.0.1:55070 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:55070 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:55082 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:43704 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 58
    INFO:     127.0.0.1:43704 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:43704 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:55082 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:55070 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 59
    INFO:     127.0.0.1:55070 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:55070 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:55082 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:43704 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 60
    INFO:     127.0.0.1:43704 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:43704 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:55082 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:55070 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 61
    INFO:     127.0.0.1:56500 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:56500 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:56500 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:56500 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 62
    INFO:     127.0.0.1:56500 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:56500 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:56506 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:56514 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 63
    INFO:     127.0.0.1:56514 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:56514 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:56506 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:56500 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotations: [Annotation(coords={'p0': [607.5, 867.5], 'p1': [780, 980]}, value='1')]
    Writing notation from {'p0': [607.5, 867.5], 'p1': [780, 980]}: 0.6423611111111112 0.4811197916666667 0.1597222222222222 0.05859375
    INFO:     127.0.0.1:49780 - "POST /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:49780 - "GET /clearannotations HTTP/1.1" 200 OK
    Updated annotations: [Annotation(coords={'p0': [602.5, 857.5], 'p1': [780, 982.5]}, value='0')]
    Writing notation from {'p0': [602.5, 857.5], 'p1': [780, 982.5]}: 0.6400462962962963 0.4791666666666667 0.16435185185185186 0.06510416666666667
    INFO:     127.0.0.1:49780 - "POST /annotations HTTP/1.1" 200 OK
    Updated annotation_index: 64
    INFO:     127.0.0.1:49780 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:49780 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:49780 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:49780 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 63
    Reading 0 0.6400462962962963 0.4791666666666667 0.16435185185185186 0.06510416666666667
    
    INFO:     127.0.0.1:49780 - "GET /prev HTTP/1.1" 200 OK
    INFO:     127.0.0.1:49780 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: [Annotation(coords={'p0': [602.5, 857.5], 'p1': [780.0, 982.5]}, value='0')]
    INFO:     127.0.0.1:50420 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:50426 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 62
    INFO:     127.0.0.1:50426 - "GET /prev HTTP/1.1" 200 OK
    INFO:     127.0.0.1:50426 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:50420 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:49780 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 63
    Reading 0 0.6400462962962963 0.4791666666666667 0.16435185185185186 0.06510416666666667
    
    INFO:     127.0.0.1:49780 - "GET /next HTTP/1.1" 200 OK
    Current annotations: [Annotation(coords={'p0': [602.5, 857.5], 'p1': [780.0, 982.5]}, value='0')]
    INFO:     127.0.0.1:50420 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:49780 - "GET /annotation_index HTTP/1.1" 200 OK
    INFO:     127.0.0.1:50426 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 64
    INFO:     127.0.0.1:50426 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:50426 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:49780 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:50420 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 65
    INFO:     127.0.0.1:50420 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:50420 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:49780 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:50426 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 66
    INFO:     127.0.0.1:50426 - "GET /next HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:49780 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:50426 - "GET /annotation_index HTTP/1.1" 200 OK
    INFO:     127.0.0.1:50420 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 67
    INFO:     127.0.0.1:50420 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:50420 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:50426 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:49780 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotations: [Annotation(coords={'p0': [1287.5, 904.9999999999999], 'p1': [1572.5, 1052.5]}, value='0')]
    Writing notation from {'p0': [1287.5, 904.9999999999999], 'p1': [1572.5, 1052.5]}: 1.3240740740740742 0.509765625 0.2638888888888889 0.07682291666666673
    INFO:     127.0.0.1:51620 - "POST /annotations HTTP/1.1" 200 OK
    Updated annotation_index: 68
    INFO:     127.0.0.1:51620 - "GET /next HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:51620 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51620 - "GET /annotation_index HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51620 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 69
    INFO:     127.0.0.1:51620 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51620 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:51632 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51634 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 70
    INFO:     127.0.0.1:51634 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51634 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:51632 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51620 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 71
    INFO:     127.0.0.1:51620 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51620 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:51632 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51634 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 72
    INFO:     127.0.0.1:51634 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51634 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:51632 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51620 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotations: [Annotation(coords={'p0': [320, 910], 'p1': [452.5, 1000]}, value='0')]
    Writing notation from {'p0': [320, 910], 'p1': [452.5, 1000]}: 0.3576388888888889 0.4973958333333333 0.12268518518518519 0.046875
    INFO:     127.0.0.1:51620 - "POST /annotations HTTP/1.1" 200 OK
    Updated annotation_index: 73
    INFO:     127.0.0.1:51620 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51620 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:51620 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51620 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 74
    INFO:     127.0.0.1:51620 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51620 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:37860 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:37876 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 75
    INFO:     127.0.0.1:37876 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:37876 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:37860 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51620 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 76
    INFO:     127.0.0.1:51620 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51620 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:37860 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:37876 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 77
    INFO:     127.0.0.1:37876 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:37876 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:37860 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51620 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 78
    INFO:     127.0.0.1:51620 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51620 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:37860 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:37876 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotations: [Annotation(coords={'p0': [1060, 910], 'p1': [1280, 1032.5]}, value='0')]
    Writing notation from {'p0': [1060, 910], 'p1': [1280, 1032.5]}: 1.0833333333333333 0.505859375 0.2037037037037037 0.06380208333333333
    INFO:     127.0.0.1:37876 - "POST /annotations HTTP/1.1" 200 OK
    Updated annotations: [Annotation(coords={'p0': [1060, 910], 'p1': [1280, 1032.5]}, value='0'), Annotation(coords={'p0': [1175, 900], 'p1': [1360, 1007.5]}, value='1')]
    Writing notation from {'p0': [1060, 910], 'p1': [1280, 1032.5]}: 1.0833333333333333 0.505859375 0.2037037037037037 0.06380208333333333
    Writing notation from {'p0': [1175, 900], 'p1': [1360, 1007.5]}: 1.1736111111111112 0.4967447916666667 0.1712962962962963 0.055989583333333336
    INFO:     127.0.0.1:37876 - "POST /annotations HTTP/1.1" 200 OK
    Updated annotation_index: 79
    INFO:     127.0.0.1:37876 - "GET /next HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:37876 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:37876 - "GET /annotation_index HTTP/1.1" 200 OK
    INFO:     127.0.0.1:37876 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 80
    Reading 1 1.0243055555555556 0.5 0.18287037037037027 0.052083333333333336
    
    INFO:     127.0.0.1:52420 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:52420 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: [Annotation(coords={'p0': [1007.5, 910.0], 'p1': [1205.0, 1010.0]}, value='1')]
    INFO:     127.0.0.1:52420 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:52420 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 81
    Reading 0 1.1331018518518519 0.49609375 0.1597222222222222 0.046875
    
    INFO:     127.0.0.1:52420 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:52420 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: [Annotation(coords={'p0': [1137.5, 907.5], 'p1': [1310.0, 997.5]}, value='0')]
    INFO:     127.0.0.1:52428 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:52436 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 82
    INFO:     127.0.0.1:52436 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:52436 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:52428 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:52420 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotations: [Annotation(coords={'p0': [340, 907.5], 'p1': [440, 967.5]}, value='0')]
    Writing notation from {'p0': [340, 907.5], 'p1': [440, 967.5]}: 0.3611111111111111 0.48828125 0.09259259259259259 0.03125
    INFO:     127.0.0.1:37442 - "POST /annotations HTTP/1.1" 200 OK
    Updated annotation_index: 83
    INFO:     127.0.0.1:37442 - "GET /next HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:37442 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:37442 - "GET /annotation_index HTTP/1.1" 200 OK
    INFO:     127.0.0.1:37442 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotations: [Annotation(coords={'p0': [1590, 932.5], 'p1': [1890, 1067.5]}, value='0')]
    Writing notation from {'p0': [1590, 932.5], 'p1': [1890, 1067.5]}: 1.6111111111111112 0.5208333333333334 0.2777777777777778 0.0703125
    INFO:     127.0.0.1:37450 - "POST /annotations HTTP/1.1" 200 OK
    Updated annotation_index: 84
    INFO:     127.0.0.1:37450 - "GET /next HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:37450 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:37466 - "GET /annotation_index HTTP/1.1" 200 OK
    INFO:     127.0.0.1:37450 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 85
    INFO:     127.0.0.1:37450 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:37450 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:37466 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:56420 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 86
    INFO:     127.0.0.1:56420 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:56420 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:37466 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:37450 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 87
    INFO:     127.0.0.1:37450 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:37450 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:37466 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:56420 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 88
    INFO:     127.0.0.1:56420 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:56420 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:37466 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:37450 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 89
    INFO:     127.0.0.1:37450 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:37450 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:37466 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:56420 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 90
    INFO:     127.0.0.1:56420 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:56420 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:37466 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:37450 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 91
    INFO:     127.0.0.1:37450 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:37450 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:37466 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:56420 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 92
    INFO:     127.0.0.1:56420 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:56420 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:37466 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:37450 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 93
    INFO:     127.0.0.1:37450 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:37450 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:37466 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:56420 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 94
    INFO:     127.0.0.1:56420 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:56420 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:37466 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:37450 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 95
    INFO:     127.0.0.1:37450 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:37450 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:37466 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:56420 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotations: [Annotation(coords={'p0': [755, 805], 'p1': [1222.5, 997.5]}, value='0')]
    Writing notation from {'p0': [755, 805], 'p1': [1222.5, 997.5]}: 0.9155092592592593 0.4694010416666667 0.43287037037037035 0.10026041666666667
    INFO:     127.0.0.1:56420 - "POST /annotations HTTP/1.1" 200 OK
    Updated annotation_index: 96
    INFO:     127.0.0.1:56420 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:56420 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:56420 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:56420 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotations: [Annotation(coords={'p0': [1350, 875], 'p1': [1670, 1057.5]}, value='0')]
    Writing notation from {'p0': [1350, 875], 'p1': [1670, 1057.5]}: 1.3981481481481481 0.5032552083333334 0.2962962962962963 0.09505208333333333
    INFO:     127.0.0.1:56420 - "POST /annotations HTTP/1.1" 200 OK
    Updated annotation_index: 97
    INFO:     127.0.0.1:56420 - "GET /next HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:56420 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51002 - "GET /annotation_index HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51010 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 98
    INFO:     127.0.0.1:51010 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51010 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:51002 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:56420 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 99
    INFO:     127.0.0.1:56420 - "GET /next HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:51002 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:56420 - "GET /annotation_index HTTP/1.1" 200 OK
    INFO:     127.0.0.1:51010 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotations: [Annotation(coords={'p0': [207.5, 897.5], 'p1': [320, 960]}, value='1')]
    Writing notation from {'p0': [207.5, 897.5], 'p1': [320, 960]}: 0.24421296296296297 0.4837239583333333 0.10416666666666667 0.032552083333333336
    INFO:     127.0.0.1:35048 - "POST /annotations HTTP/1.1" 200 OK
    Updated annotation_index: 100
    INFO:     127.0.0.1:35048 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:35048 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:35048 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:35048 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotations: [Annotation(coords={'p0': [97.5, 900], 'p1': [240, 985.0000000000001]}, value='1')]
    Writing notation from {'p0': [97.5, 900], 'p1': [240, 985.0000000000001]}: 0.15625 0.4908854166666667 0.13194444444444445 0.04427083333333339
    INFO:     127.0.0.1:35048 - "POST /annotations HTTP/1.1" 200 OK
    Updated annotation_index: 101
    INFO:     127.0.0.1:35048 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:35048 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:35064 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:35072 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotations: [Annotation(coords={'p0': [150, 895], 'p1': [250.00000000000003, 957.5]}, value='1')]
    Writing notation from {'p0': [150, 895], 'p1': [250.00000000000003, 957.5]}: 0.18518518518518517 0.482421875 0.09259259259259262 0.032552083333333336
    INFO:     127.0.0.1:35072 - "POST /annotations HTTP/1.1" 200 OK
    Updated annotation_index: 102
    INFO:     127.0.0.1:35072 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:35072 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:35064 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:35048 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotations: [Annotation(coords={'p0': [1145, 935], 'p1': [1405, 1037.5]}, value='1')]
    Writing notation from {'p0': [1145, 935], 'p1': [1405, 1037.5]}: 1.1805555555555556 0.513671875 0.24074074074074073 0.053385416666666664
    INFO:     127.0.0.1:35048 - "POST /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:35048 - "GET /clearannotations HTTP/1.1" 200 OK
    Updated annotations: [Annotation(coords={'p0': [1142.5, 917.5000000000001], 'p1': [1412.5, 1042.5]}, value='1')]
    Writing notation from {'p0': [1142.5, 917.5000000000001], 'p1': [1412.5, 1042.5]}: 1.1828703703703705 0.5104166666666666 0.25 0.0651041666666666
    INFO:     127.0.0.1:35048 - "POST /annotations HTTP/1.1" 200 OK
    Updated annotation_index: 103
    INFO:     127.0.0.1:35048 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:35048 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:35048 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:35048 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotations: [Annotation(coords={'p0': [342.5, 902.5], 'p1': [447.5, 960]}, value='0')]
    Writing notation from {'p0': [342.5, 902.5], 'p1': [447.5, 960]}: 0.36574074074074076 0.4850260416666667 0.09722222222222222 0.029947916666666668
    INFO:     127.0.0.1:34548 - "POST /annotations HTTP/1.1" 200 OK
    Updated annotation_index: 104
    INFO:     127.0.0.1:34548 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:34548 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:34558 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:34548 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotations: [Annotation(coords={'p0': [562.5, 900], 'p1': [677.5, 967.5]}, value='0')]
    Writing notation from {'p0': [562.5, 900], 'p1': [677.5, 967.5]}: 0.5740740740740741 0.486328125 0.10648148148148148 0.03515625
    INFO:     127.0.0.1:34548 - "POST /annotations HTTP/1.1" 200 OK
    Updated annotation_index: 105
    INFO:     127.0.0.1:34548 - "GET /next HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:34548 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:34558 - "GET /annotation_index HTTP/1.1" 200 OK
    INFO:     127.0.0.1:53934 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 106
    INFO:     127.0.0.1:53934 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:53934 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:34558 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:34548 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 107
    INFO:     127.0.0.1:34548 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:34548 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:34558 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:53934 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 108
    INFO:     127.0.0.1:53934 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:53934 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:34558 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:34548 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 109
    INFO:     127.0.0.1:34548 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:34548 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:34558 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:53934 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotations: [Annotation(coords={'p0': [75, 970], 'p1': [0, 900]}, value='0')]
    Writing notation from {'p0': [75, 970], 'p1': [0, 900]}: 0.10416666666666667 0.5234375 0.06944444444444445 0.036458333333333336
    INFO:     127.0.0.1:53934 - "POST /annotations HTTP/1.1" 200 OK
    Updated annotation_index: 110
    INFO:     127.0.0.1:53934 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:53934 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:34558 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:34548 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 111
    INFO:     127.0.0.1:34548 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:34548 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:34558 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:53934 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 112
    INFO:     127.0.0.1:53934 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:53934 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:34558 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:34548 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 113
    INFO:     127.0.0.1:34548 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:34548 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:34558 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:53934 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 114
    INFO:     127.0.0.1:53934 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:53934 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:34558 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:34548 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotations: [Annotation(coords={'p0': [372.5, 897.5], 'p1': [532.5, 1000]}, value='0')]
    Writing notation from {'p0': [372.5, 897.5], 'p1': [532.5, 1000]}: 0.41898148148148145 0.494140625 0.14814814814814814 0.053385416666666664
    INFO:     127.0.0.1:34548 - "POST /annotations HTTP/1.1" 200 OK
    Updated annotation_index: 115
    INFO:     127.0.0.1:34548 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:34548 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:34558 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:53934 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 116
    INFO:     127.0.0.1:53934 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:53934 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:34558 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:34548 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotation_index: 117
    INFO:     127.0.0.1:34548 - "GET /next HTTP/1.1" 200 OK
    INFO:     127.0.0.1:34548 - "GET /annotation_index HTTP/1.1" 200 OK
    Current annotations: []
    INFO:     127.0.0.1:34558 - "GET /annotations HTTP/1.1" 200 OK
    INFO:     127.0.0.1:53934 - "GET /image_list HTTP/1.1" 200 OK
    Updated annotations: [Annotation(coords={'p0': [497.49999999999994, 907.5], 'p1': [657.5, 1002.5]}, value='1')]
    Writing notation from {'p0': [497.49999999999994, 907.5], 'p1': [657.5, 1002.5]}: 0.5347222222222222 0.4973958333333333 0.1481481481481482 0.049479166666666664
    INFO:     127.0.0.1:53934 - "POST /annotations HTTP/1.1" 200 OK



```python

```

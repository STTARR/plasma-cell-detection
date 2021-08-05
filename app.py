"""API using Flask. Run in development mode with:
set FLASK_APP=app.py
flask run
"""

from flask import Flask, json, jsonify, request, abort, send_from_directory
from werkzeug.utils import secure_filename

from pathlib import Path
import datetime
import torch
import torchvision.transforms as transforms
from PIL import Image
from base64 import b64encode
from io import BytesIO

from mmdetect.model.unetvgg import UNetVGG
from mmdetect.model.postprocess import model_output_to_points

# Parameters
UPLOAD_DIR = r"web/uploads"
MODEL_PATH = r"model/trained_model.pth"

# Setup
app = Flask(__name__)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)
MODEL = UNetVGG("vgg13_bn", 3)
MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
MODEL.to(DEVICE)
MODEL.eval()


def get_timestamped_filename(filename):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return f"{timestamp}_{secure_filename(filename)}"


def scale_image(im_pil, scale):
    w, h = round(im_pil.width * scale), round(im_pil.height * scale)
    return im_pil.resize((w, h), resample=1)


def process_image(im_pil):
    # Image transform
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # Run through network
    with torch.no_grad():  # disable autograd for inference
        im = tform(im_pil).to(DEVICE).float().unsqueeze(0)
        out = MODEL(im)[0]
        out = torch.nn.functional.softmax(out.cpu(), dim=0).detach()
    blue, brown = model_output_to_points(out)
    return blue, brown


@app.route("/")
def index():
    return send_from_directory("", "web/index.html")


@app.route("/run", methods=["POST"])
def count():
    if request.method == "POST":
        if "image" not in request.files:
            print(request.files)
            abort(400, "No image.")  # Bad request
        mpp = float(request.form.get("mpp"))
        # Our images are 0.25 MPP (Aperio 40x)
        # Daniel's camera images double that size, so should be 0.125 MPP
        # scale is basically how much to resample (so for Daniel scale should equal 0.5)
        scale = mpp / 0.25

        image = request.files["image"]
        # Save a copy of all uploaded images
        image.save(str(Path(UPLOAD_DIR) / get_timestamped_filename(image.filename)))

        # Send image to model
        im = Image.open(image.stream).convert("RGB")
        blue, brown = process_image(scale_image(im, scale))
        blue = blue / scale
        brown = brown / scale

        # Clear CUDA memory
        torch.cuda.empty_cache()
        
        return jsonify({
            # Separate coordinates into x/y since easier to handle on JS end
            "negative": {"x": blue[:,1].tolist(), "y": blue[:,0].tolist()},
            "positive": {"x": brown[:,1].tolist(), "y": brown[:,0].tolist()},
        })


@app.route("/test", methods=["POST"])
def test():
    if request.method == "POST":
        if "image" not in request.files:
            print(request.files)
            abort(400, "No image.")  # Bad request
        
        print(request.form)

        return jsonify({
            # Separate coordinates into x/y since easier to handle on JS end
            "negative": {"x": [0], "y": [0]},
            "positive": {"x": [100], "y": [100]}
        })
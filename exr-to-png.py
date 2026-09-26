from flask import Flask, request, send_file, render_template_string
import OpenEXR
import Imath
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>EXR → PNG</title>

    <style>
        body {
            margin: 0;
            min-height: 100vh;
            background: #111;
            color: #eee;
            font-family: Arial, sans-serif;

            display: flex;
            justify-content: center;
            align-items: center;
        }

        .container {
            width: min(900px, 92vw);
            text-align: center;
        }

        h1 {
            margin-bottom: 8px;
        }

        .subtitle {
            color: #999;
            margin-bottom: 25px;
        }

        #dropzone {
            border: 2px dashed #666;
            border-radius: 14px;
            padding: 60px 20px;
            cursor: pointer;
            transition: 0.2s;
        }

        #dropzone.dragover {
            border-color: white;
            background: #1c1c1c;
        }

        #previewArea {
            display: none;
        }

        #preview {
            display: block;
            max-width: 100%;
            max-height: 70vh;
            margin: 0 auto 20px auto;
            border-radius: 5px;
            background: black;
        }

        .buttons {
            display: flex;
            gap: 12px;
            justify-content: center;
        }

        button {
            padding: 11px 22px;
            border: 0;
            border-radius: 8px;
            font-size: 15px;
            cursor: pointer;
        }

        #download {
            background: #eee;
            color: #111;
        }

        #newImage {
            background: #333;
            color: #eee;
        }

        #filename {
            color: #aaa;
            margin-bottom: 12px;
        }
    </style>
</head>

<body>

<div class="container">

    <h1>EXR → PNG</h1>
    <div class="subtitle">
        Raw linear conversion — no tone mapping, no normalization
    </div>

    <div id="dropzone">
        Drop an <b>.exr</b> image here<br><br>
        or click to select one
    </div>

    <input
        id="fileInput"
        type="file"
        accept=".exr"
        hidden
    >

    <div id="previewArea">
        <div id="filename"></div>

        <img id="preview">

        <div class="buttons">
            <button id="download">Download PNG</button>
            <button id="newImage">Upload New</button>
        </div>
    </div>

</div>

<script>

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");

const previewArea = document.getElementById("previewArea");
const preview = document.getElementById("preview");
const filename = document.getElementById("filename");

const downloadButton = document.getElementById("download");
const newImageButton = document.getElementById("newImage");

let pngBlob = null;
let originalName = "";


dropzone.addEventListener("click", () => {
    fileInput.click();
});


dropzone.addEventListener("dragover", e => {
    e.preventDefault();
    dropzone.classList.add("dragover");
});


dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
});


dropzone.addEventListener("drop", e => {
    e.preventDefault();

    dropzone.classList.remove("dragover");

    if (e.dataTransfer.files.length) {
        uploadFile(e.dataTransfer.files[0]);
    }
});


fileInput.addEventListener("change", () => {
    if (fileInput.files.length) {
        uploadFile(fileInput.files[0]);
    }
});


async function uploadFile(file) {

    if (!file.name.toLowerCase().endsWith(".exr")) {
        alert("Please select an EXR image.");
        return;
    }

    originalName = file.name;

    const formData = new FormData();
    formData.append("file", file);

    dropzone.innerHTML = "Converting...";

    try {

        const response = await fetch("/convert", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const text = await response.text();
            throw new Error(text);
        }

        pngBlob = await response.blob();

        const imageURL = URL.createObjectURL(pngBlob);

        preview.src = imageURL;

        filename.textContent =
            originalName + " → " +
            originalName.replace(/\.exr$/i, ".png");

        dropzone.style.display = "none";
        previewArea.style.display = "block";

    }

    catch (error) {

        alert("Could not convert EXR:\n\n" + error.message);

        dropzone.innerHTML =
            'Drop an <b>.exr</b> image here<br><br>or click to select one';
    }
}


downloadButton.addEventListener("click", () => {

    if (!pngBlob)
        return;

    const url = URL.createObjectURL(pngBlob);

    const a = document.createElement("a");

    a.href = url;
    a.download = originalName.replace(/\.exr$/i, ".png");

    document.body.appendChild(a);

    a.click();

    a.remove();

    URL.revokeObjectURL(url);
});


newImageButton.addEventListener("click", () => {

    previewArea.style.display = "none";
    dropzone.style.display = "block";

    dropzone.innerHTML =
        'Drop an <b>.exr</b> image here<br><br>or click to select one';

    preview.src = "";
    pngBlob = null;

    fileInput.value = "";
});

</script>

</body>
</html>
"""


def read_exr(file_obj):
    """
    Read RGB values directly from an OpenEXR file as float32.

    No normalization.
    No exposure adjustment.
    No tone mapping.
    No gamma correction.
    """

    # OpenEXR wants an actual filename, so save uploaded bytes temporarily.
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(
        suffix=".exr",
        delete=False
    ) as temp:

        temp.write(file_obj.read())
        temp_path = temp.name

    try:
        exr = OpenEXR.InputFile(temp_path)

        header = exr.header()

        dw = header["dataWindow"]

        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        pixel_type = Imath.PixelType(
            Imath.PixelType.FLOAT
        )

        channels = header["channels"]

        # ------------------------------------------------
        # RGB
        # ------------------------------------------------

        if all(c in channels for c in ("R", "G", "B")):

            r = np.frombuffer(
                exr.channel("R", pixel_type),
                dtype=np.float32
            )

            g = np.frombuffer(
                exr.channel("G", pixel_type),
                dtype=np.float32
            )

            b = np.frombuffer(
                exr.channel("B", pixel_type),
                dtype=np.float32
            )

            image = np.stack(
                [r, g, b],
                axis=-1
            )

        # ------------------------------------------------
        # Grayscale / Y
        # ------------------------------------------------

        elif "Y" in channels:

            y = np.frombuffer(
                exr.channel("Y", pixel_type),
                dtype=np.float32
            )

            image = np.stack(
                [y, y, y],
                axis=-1
            )

        else:
            raise ValueError(
                "EXR does not contain R/G/B or Y channels."
            )

        image = image.reshape(
            height,
            width,
            3
        )

        return image

    finally:
        try:
            exr.close()
        except:
            pass

        os.remove(temp_path)


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/convert", methods=["POST"])
def convert():

    if "file" not in request.files:
        return "No file supplied.", 400

    uploaded = request.files["file"]

    try:
        image = read_exr(uploaded)

    except Exception as e:
        return str(e), 400

    # ----------------------------------------------------
    # IMPORTANT
    #
    # There is deliberately NO:
    #
    #   image /= image.max()
    #
    # NO Reinhard
    # NO exposure
    # NO gamma
    # NO percentile scaling
    #
    # Floating point EXR values are simply clipped because
    # standard PNG cannot contain values outside its integer
    # display range.
    #
    #   < 0 → 0
    #   > 1 → 1
    # ----------------------------------------------------

    image = np.clip(image, 0.0, 1.0)

    # Convert directly to 8-bit PNG.
    image = (image * 255.0).round().astype(np.uint8)

    pil_image = Image.fromarray(
        image,
        mode="RGB"
    )

    buffer = io.BytesIO()

    pil_image.save(
        buffer,
        format="PNG"
    )

    buffer.seek(0)

    return send_file(
        buffer,
        mimetype="image/png"
    )


if __name__ == "__main__":

    print()
    print("EXR → PNG converter")
    print("-------------------")
    print("Open:")
    print("http://127.0.0.1:5000")
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
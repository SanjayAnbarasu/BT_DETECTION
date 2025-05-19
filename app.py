import os
import cv2
import numpy as np
import keras._tf_keras.keras as k
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, send_file
from keras._tf_keras.keras.models import Model
import json
import google.generativeai as genai
import base64
from PIL import Image
from io import BytesIO
from flask import send_file
import io
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as RLImage, Table, TableStyle, Spacer
from reportlab.lib import colors
import os
import cv2
import os
genai.configure(api_key=os.getenv("API_KEY"))

# Initialize Flask App
app = Flask(__name__)

# Define Upload Folder
UPLOAD_FOLDER = 'upload_folder'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load Model
MODEL_PATH = 'model.keras'  # Update with your actual model path
model = tf.keras.models.load_model(MODEL_PATH)

# Class Labels
CLASS_DICT = {0: 'Glioma', 1: 'Meningioma', 2: 'No Tumor', 3: 'Pituitary'}

# Configure Gemini API
genai.configure(api_key="AIzaSyD8PdEerXFWttZMY_80fiIsy_K8ciIDnaI")


# Configure API key
genai.configure(api_key="AIzaSyBKu9wEhNmxRNI0lGEPE2nv48s0E4yo7Ek")

def get_disease_info(disease_name):
    prompt = f"""
Act as a medical expert. Provide structured information about the disease "{disease_name}". 
Your response must be in valid JSON format without extra text. Structure it as:

{{
    "Definition": "Brief paragraph and definition of the disease .",
    "Causes": ["Cause 1", "Cause 2", "Cause 3","Cause 4", "Cause 5", "Cause 6",  ],
    "Controls": ["Control 1", "Control 2", "Control 3","Control 54", "Control 5", "Control 7"],
    "Treatments": ["Treatment 1", "Treatment 2", "Treatment 3","Treatment 4", "Treatment 5", "Treatment 6"]
}}


    """

    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    response = model.generate_content(prompt)

    try:
        json_start = response.text.find("{")
        json_end = response.text.rfind("}")
        json_data = response.text[json_start:json_end + 1]
        data = json.loads(json_data)
        return data
    except json.JSONDecodeError:
        return {"error": "Failed to parse AI response", "raw_response": response.text}


def VizGradCAM(model, image, filename, interpolant=0.5):
    last_conv_layer = next(
        x for x in model.layers[::-1] if isinstance(x, k.layers.Conv2D)
    )
    target_layer = model.get_layer(last_conv_layer.name)

    img = np.expand_dims(image, axis=0)
    prediction = model.predict(img)
    prediction_idx = np.argmax(prediction)
    predicted_label = CLASS_DICT[prediction_idx]

    with tf.GradientTape() as tape:
        gradient_model = Model([model.inputs], [target_layer.output, model.output])
        conv2d_out, prediction = gradient_model(img)
        loss = prediction[:, prediction_idx]
    
    gradients = tape.gradient(loss, conv2d_out)
    output = conv2d_out[0]
    weights = tf.reduce_mean(gradients[0], axis=(0, 1))
    activation_map = np.zeros(output.shape[0:2], dtype=np.float32)
    
    for idx, weight in enumerate(weights):
        activation_map += weight * output[:, :, idx]
    
    activation_map = cv2.resize(
        activation_map.numpy(), (image.shape[1], image.shape[0])
    )
    activation_map = np.maximum(activation_map, 0)
    activation_map = (activation_map - activation_map.min()) / (
        activation_map.max() - activation_map.min()
    )
    activation_map = np.uint8(255 * activation_map)
    heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)
    blended = np.uint8(image * interpolant + heatmap * (1 - interpolant))

    # Save the processed image without prediction label
    filename_without_ext = os.path.splitext(filename)[0]
    result_filename = f"{filename_without_ext}.png"
    result_image_path = os.path.join(UPLOAD_FOLDER, result_filename)
    
    cv2.imwrite(result_image_path, blended)

    _, buffer = cv2.imencode('.png', blended)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    
    return encoded_image, predicted_label, result_filename


@app.route("/", methods=["GET", "POST"])
def upload_file():
    raw_image = None
    result_image = None
    prediction = None
    result_filename = None

    if request.method == "POST":
        file = request.files.get("file")
        captured_image = request.form.get("captured_image")  # Base64 image

        if file and file.filename != "":  # If an image is uploaded
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                return "Error: Unable to decode image. Please upload a valid image file."
            
            filename = file.filename

        elif captured_image:  # If an image is captured
            try:
                img_data = base64.b64decode(captured_image.split(",")[1])
                image = np.array(Image.open(BytesIO(img_data)))  # Convert to NumPy array
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to OpenCV format
                filename = "captured_image.png"
                print("Captured image processed successfully.")  # Debugging print
            except Exception as e:
                print("Error processing captured image:", str(e))
                return "Error processing captured image", 400

        else:
            return "No image provided"

        # Resize image
        image = cv2.resize(image, (240, 240))

        # Encode original image to display
        _, raw_buffer = cv2.imencode('.png', image)
        raw_image = base64.b64encode(raw_buffer).decode('utf-8')

        # Process image with Grad-CAM
        result_image, prediction, result_filename = VizGradCAM(model, image, filename)

    return render_template(
        "index.html",
        raw_image=raw_image,
        result_image=result_image,
        prediction=prediction,
        result_filename=result_filename
    )



@app.route("/get_disease_info", methods=["POST", "GET"])
def get_disease():
    if request.method == "GET":
        return jsonify({"error": "Please use POST method to get disease information."})

    data = request.json
    disease_name = data.get("disease", "").strip()
    
    if not disease_name:
        return jsonify({"error": "Disease name is required"}), 400

    # If the disease is "No tumor", return only the label
    if disease_name.lower() == "no tumor":
        return jsonify({"prediction": "No tumor"})

    result = get_disease_info(disease_name)
    if not result or "Definition" not in result:
        return jsonify({"error": "No data found. Try a different disease."})

    return jsonify(result)




@app.route("/download_processed_image/<filename>/<label>")
def download_processed_image(filename, label):
    result_image_path = os.path.join(UPLOAD_FOLDER, filename)

    if not label:
        label = "Unknown"

    if not os.path.exists(result_image_path):
        return "No processed image available for download.", 404

    # Overlay the label onto the PNG as before
    image = cv2.imread(result_image_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
    text_x = image.shape[1] - text_size[0] - 10
    text_y = image.shape[0] - 10
    cv2.putText(image, label, (text_x, text_y), font, font_scale,
                (255, 255, 255), font_thickness, cv2.LINE_AA)
    cv2.imwrite(result_image_path, image)

    # Fetch disease info JSON
    report_data = {}
    if label.lower() != "no tumor":
        report_data = get_disease_info(label)
    definition = report_data.get("Definition", "No definition available.")
    causes     = report_data.get("Causes", [])
    controls   = report_data.get("Controls", [])
    treatments = report_data.get("Treatments", [])

    # Create PDF in memory using Platypus for better layout
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=20,
        leftMargin=20,
        topMargin=20,
        bottomMargin=20
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'title',
        parent=styles['Heading1'],
        fontSize=26,
        alignment=1,  # Centered
        spaceAfter=20,
        leading=28
    )
    subtitle_style = ParagraphStyle(
        'subtitle',
        parent=styles['Heading2'],
        textColor=colors.red,
        fontSize=16,
        spaceAfter=12,
        leading=20
    )
    normal_style = ParagraphStyle(
        'normal',
        parent=styles['BodyText'],
        fontSize=11,
        leading=18,
        spaceAfter=10
    )

    elements = []
    # Title Heading
    elements.append(Paragraph("Brain Tumor Prediction Report", title_style))
    elements.append(Spacer(1, 16))

    # Top section: image and report title & definition side by side
    img = RLImage(result_image_path, width=240, height=240)
    text_flow = []
    text_flow.append(Spacer(5, 1))
    text_flow.append(Paragraph(label, subtitle_style))
    text_flow.append(Spacer(1, 10))
    text_flow.append(Paragraph(definition, normal_style))
    right_table = text_flow

    data = [[img, right_table]]
    table = Table(data, colWidths=[250, 300])
    table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 30))

    # Bottom section: Causes, Controls, Treatments as columns
    headers = ['Causes', 'Controls', 'Treatments']
    rows = []
    max_len = max(len(causes), len(controls), len(treatments))
    for i in range(max_len):
        row = []
        row.append(Paragraph(causes[i], normal_style) if i < len(causes) else '')
        row.append(Paragraph(controls[i], normal_style) if i < len(controls) else '')
        row.append(Paragraph(treatments[i], normal_style) if i < len(treatments) else '')
        rows.append(row)

    table_data = [headers] + rows
    info_table = Table(table_data, colWidths=[180, 180, 180])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgreen),
        ('TEXTCOLOR', (0,0), (-1,0), colors.darkblue),
        ('ALIGN',(0,0),(-1,-1),'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgreen),
    ]))

    elements.append(info_table)

    doc.build(elements)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="brain_tumor_report.pdf",
        mimetype="application/pdf"
    )




    return "No processed image available for download.", 404


if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for, session
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
app.secret_key = "plant_secret_key"

# Load trained model
model = tf.keras.models.load_model("plant_disease_model.h5")

CLASS_NAMES = [
    "Apple Scab","Apple Black Rot","Apple Cedar Rust","Apple Healthy",
    "Blueberry Healthy","Cherry Powdery Mildew","Cherry Healthy",
    "Corn Gray Leaf Spot","Corn Common Rust","Corn Northern Leaf Blight","Corn Healthy",
    "Grape Black Rot","Grape Esca","Grape Leaf Blight","Grape Healthy",
    "Peach Bacterial Spot","Peach Healthy",
    "Potato Early Blight","Potato Late Blight","Potato Healthy",
    "Tomato Bacterial Spot","Tomato Early Blight","Tomato Late Blight",
    "Tomato Leaf Mold","Tomato Septoria Leaf Spot",
    "Tomato Spider Mites","Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus","Tomato Mosaic Virus","Tomato Healthy"
]

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        session["user"] = email
        return redirect(url_for("home"))
    return render_template("login.html")

@app.route("/home")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    path = os.path.join("static", file.filename)
    file.save(path)

    img = image.load_img(path, target_size=(224,224))
    img = image.img_to_array(img)/255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    result = CLASS_NAMES[np.argmax(prediction)]

    return render_template("result.html", result=result, image=path)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)

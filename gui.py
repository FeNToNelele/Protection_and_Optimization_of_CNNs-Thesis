import tkinter
from pathlib import Path

from tkinter import Tk, Canvas, Entry, Button, PhotoImage, Label
from tkinter.filedialog import askopenfile

import subprocess

import os
import tensorflow as tf
from keras import Sequential
from keras.models import load_model
from tensorflow.python.keras import models
import cv2 as cv
import numpy as np
from PIL import ImageTk, Image


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"./assets/frame0")
CACHE_PATH = OUTPUT_PATH / Path(r"./cache/model_choice.txt")

model_choice = open(CACHE_PATH, "w")
model_choice.close()

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


window = Tk()
window.title("GUI for image classification")
window.geometry("1700x1024")
window.configure(bg = "#F4F4F9")


canvas = Canvas(
    window,
    bg = "#F4F4F9",
    height = 1024,
    width = 1700,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
canvas.create_rectangle(
    0.0,
    0.0,
    850.0,
    1024.0,
    fill="#7FA9C6",
    outline="")

canvas.create_rectangle(
    53.0,
    870.0,
    173.0,
    920.0,
    fill="#E9F09D",
    outline="")

canvas.create_text(
    53.0,
    239.0,
    anchor="nw",
    text="Image ",
    fill="#272932",
    font=("Heebo Regular", 32 * -1)
)

canvas.create_text(
    53.0,
    316.0,
    anchor="nw",
    text="Path:",
    fill="#272932",
    font=("Heebo Light", 24 * -1)
)

canvas.create_text(
    53.0,
    650.0,
    anchor="nw",
    text="Template window size",
    fill="#272932",
    font=("Heebo Light", 24 * -1)
)

canvas.create_text(
    53.0,
    708.0,
    anchor="nw",
    text="Search window size",
    fill="#272932",
    font=("Heebo Light", 24 * -1)
)

canvas.create_text(
    489.0,
    657.0,
    anchor="nw",
    text="px",
    fill="#272932",
    font=("Heebo Light", 15 * -1)
)

canvas.create_text(
    489.0,
    715.0,
    anchor="nw",
    text="px",
    fill="#272932",
    font=("Heebo Light", 15 * -1)
)

canvas.create_text(
    46.0,
    563.0,
    anchor="nw",
    text="Noise reduction settings",
    fill="#272932",
    font=("Heebo Regular", 32 * -1)
)

canvas.create_text(
    407.0,
    569.0,
    anchor="nw",
    text="(optional)",
    fill="#272932",
    font=("Heebo Light", 24 * -1)
)

img_btn_load_image = PhotoImage(
    file=relative_to_assets("btn_load_image.png"))
btn_load_image = Button(
    image=img_btn_load_image,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: load_image(),
    relief="flat"
)
btn_load_image.place(
    x=265.0,
    y=380.0,
    width=160.0,
    height=58.0
)

canvas.create_rectangle(
    589.0,
    305.0,
    729.0,
    363.0,
    fill="#7FAAC7",
    outline="")

img_btn_browse_image_path = PhotoImage(
    file=relative_to_assets("btn_browse_image_path.png"))
btn_browse_image_path = Button(
    image=img_btn_browse_image_path,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: change_image(),
    relief="flat"
)
btn_browse_image_path.place(
    x=589.0,
    y=305.0,
    width=140.0,
    height=58.0
)

template_window, search_window, noise, color_noise = 7, 21, 3, 3

img_btn_help_template_window = PhotoImage(
    file=relative_to_assets("btn_help_template_window.png"))
btn_help_template_window = Button(
    image=img_btn_help_template_window,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: tkinter.messagebox.showinfo("Template window",
                                                "Template patch size in pixels that is used to compute weights.\n"
                                                "Recommended value: 7 px."),
    relief="flat"
)
btn_help_template_window.place(
    x=527.0,
    y=657.0,
    width=30.0,
    height=20.0
)

img_btn_help_search_window = PhotoImage(
    file=relative_to_assets("btn_help_search_window.png"))
btn_help_search_window = Button(
    image=img_btn_help_search_window,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: tkinter.messagebox.showinfo("Search window",
                                                "The window size in pixels that is used to compute average for given pixel.\n"
                                                "Recommended value: 21 px.\n"
                                                "Note: Greater value leads to longer denoising time."),
    relief="flat"
)
btn_help_search_window.place(
    x=527.0,
    y=716.0,
    width=30.0,
    height=20.0
)

img_btn_help_noise = PhotoImage(
    file=relative_to_assets("btn_help_noise.png"))
btn_help_noise = Button(
    image=img_btn_help_noise,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: tkinter.messagebox.showinfo("Noise reduction",
                                                "Strength of noise reduction.\n"
                                                "Recommended value: 3.\n"
                                                "Note: The greater the value, the less detail the image preserves."),
    relief="flat"
)
btn_help_noise.place(
    x=503.0,
    y=774.0,
    width=30.0,
    height=20.0
)

img_btn_help_color_noise = PhotoImage(
    file=relative_to_assets("btn_help_color_noise.png"))
btn_help_color_noise = Button(
    image=img_btn_help_color_noise,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: tkinter.messagebox.showinfo("Color noise reduction",
                                                "Strength of color noise reduction.\n"
                                                "Recommended value: 3.\n"
                                                "Note: Values above 10 start to distort colors on image."),
    relief="flat"
)
btn_help_color_noise.place(
    x=503.0,
    y=825.0,
    width=30.0,
    height=20.0
)

canvas.create_rectangle(
    503.0,
    97.0,
    728.0,
    157.0,
    fill="#7FAAC7",
    outline="")

img_btn_model_summary = PhotoImage(
    file=relative_to_assets("btn_model_summary.png"))
btn_model_summary = Button(
    image=img_btn_model_summary,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print_summary(),
    relief="flat"
)
btn_model_summary.place(
    x=502.9955749511719,
    y=97.0,
    width=225.00442504882812,
    height=60.0
)

canvas.create_rectangle(
    53.0,
    97.0,
    233.0,
    157.0,
    fill="#7FAAC7",
    outline="")

img_btn_change_model = PhotoImage(
    file=relative_to_assets("btn_change_model.png"))
btn_change_model = Button(
    image=img_btn_change_model,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: change_model(),
    relief="flat"
)
btn_change_model.place(
    x=53.0,
    y=97.0,
    width=180.0,
    height=60.0
)

img_btn_predict = PhotoImage(
    file=relative_to_assets("btn_predict.png"))
btn_predict = Button(
    image=img_btn_predict,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: predict(),
    relief="flat"
)
btn_predict.place(
    x=249.0,
    y=935.0,
    width=186.0,
    height=58.0
)

img_btn_clear_all = PhotoImage(
    file=relative_to_assets("btn_clear_all.png"))
btn_clear_all = Button(
    image=img_btn_clear_all,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: clear_all(),
    relief="flat"
)
btn_clear_all.place(
    x=53.0,
    y=870.0,
    width=120.0,
    height=50.0
)


img_tb_image_path = PhotoImage(
    file=relative_to_assets("tb_image_path.png"))
entry_bg_1 = canvas.create_image(
    345.0,
    334.0,
    image=img_tb_image_path
)
tb_image_path = Entry(
    bd=0,
    bg="#F4F4F9",
    fg="#000716",
    highlightthickness=0
)
tb_image_path.place(
    x=130.0,
    y=309.0,
    width=430.0,
    height=48.0
)

img_btn_apply_filters = PhotoImage(
    file=relative_to_assets("btn_apply_filters.png"))
btn_apply_filters = Button(
    image=img_btn_apply_filters,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: denoise_image(),
    relief="flat"
)
btn_apply_filters.place(
    x=527.0,
    y=567.0,
    width=120.0,
    height=50.0
)


pi_tb_template_window = PhotoImage(
    file=relative_to_assets("tb_template_window.png"))
img_tb_template_window = canvas.create_image(
    431.0,
    665.0,
    image=pi_tb_template_window
)
tb_template_window = Entry(
    bd=0,
    bg="#F4F4F9",
    fg="#000716",
    highlightthickness=0
)
tb_template_window.place(
    x=381.0,
    y=645.0,
    width=100.0,
    height=38.0
)
tb_template_window.insert(0, str(template_window))


pi_tb_search_window = PhotoImage(
    file=relative_to_assets("tb_search_window.png"))
img_tb_search_window = canvas.create_image(
    431.0,
    726.0,
    image=pi_tb_search_window
)
tb_search_window = Entry(
    bd=0,
    bg="#F4F4F9",
    fg="#000716",
    highlightthickness=0
)
tb_search_window.place(
    x=381.0,
    y=706.0,
    width=100.0,
    height=38.0
)
tb_search_window.insert(0, str(search_window))

canvas.create_text(
    53.0,
    766.0,
    anchor="nw",
    text="Noise reduction strength",
    fill="#272932",
    font=("Heebo Light", 24 * -1)
)

pi_tb_noise_reduction = PhotoImage(
    file=relative_to_assets("tb_noise_reduction.png"))
img_tb_noise_reduction = canvas.create_image(
    431.0,
    784.0,
    image=pi_tb_noise_reduction
)
tb_noise = Entry(
    bd=0,
    bg="#F4F4F9",
    fg="#000716",
    highlightthickness=0
)
tb_noise.place(
    x=381.0,
    y=764.0,
    width=100.0,
    height=38.0
)
tb_noise.insert(0, str(noise))

canvas.create_text(
    53.0,
    818.0,
    anchor="nw",
    text="Color noise reduction strength",
    fill="#272932",
    font=("Heebo Light", 24 * -1)
)

pi_tb_color_noise = PhotoImage(
    file=relative_to_assets("tb_color_noise.png"))
img_tb_color_noise = canvas.create_image(
    431.0,
    836.0,
    image=pi_tb_color_noise
)
tb_color_noise = Entry(
    bd=0,
    bg="#F4F4F9",
    fg="#000716",
    highlightthickness=0
)
tb_color_noise.place(
    x=381.0,
    y=816.0,
    width=100.0,
    height=38.0
)
tb_color_noise.insert(0, str(color_noise))

canvas.create_rectangle(
    51.0,
    288.0,
    113.0,
    290.0,
    fill="#272932",
    outline="")

canvas.create_rectangle(
    44.0,
    618.0,
    277.0,
    620.0,
    fill="#272932",
    outline="")

canvas.create_rectangle(
    840.0000162124634,
    -10.0,
    850.0000610351562,
    1024.0,
    fill="#A3C4D6",
    outline="")

model = Sequential()
model_summary = str()
model_path = str()
model_choice = str()

ensemble_models = []

def change_text(label: Label, value: str):
    label.config(text=value)

def load_single_model():
    global model_path
    absolute_path = model_path
    try:
        global model, model_summary

        model = models.load_model(absolute_path, compile=False)
        tkinter.messagebox.showinfo("Success", "Model has been loaded successfully.")

        model_summary_stream = []
        model.summary(print_fn=lambda x: model_summary_stream.append(x))
        model_summary = "\n".join(model_summary_stream)
    except Exception as ex:
        print(ex)
        tkinter.messagebox.showwarning("Info", ex)


def load_ensemble_models():
    global ensemble_models

    MODEL_PATH = OUTPUT_PATH / Path("../../../../saved_models/ensemble_training")

    animal_classifier = load_model(str(MODEL_PATH) + "/animal_recogniser/model.h5")
    car_classifier = load_model(str(MODEL_PATH) + "/car_recogniser/model.h5")
    human_classifier = load_model(str(MODEL_PATH) + "/human_recogniser/model.h5")

    ensemble_models.append(animal_classifier)
    ensemble_models.append(car_classifier)
    ensemble_models.append(human_classifier)

    tkinter.messagebox.showinfo("Success", "Models have been loaded successfully.")


def change_model():
    global model_path, model_choice, ensemble_models

    ensemble_models = []
    MODEL_SEL_PATH = OUTPUT_PATH / Path("../../model-selection/build/gui.py")

    subprocess.run("python "+ str(MODEL_SEL_PATH), capture_output=True, text=True)
    model_choice = open(CACHE_PATH, "r").readline()

    if model_choice == "single":
        accepted_image_extensions = [("*.h5", "H5 File")]
        path = tkinter.filedialog.askopenfile(mode='r', filetypes=accepted_image_extensions, title="Choose a model")
        if path:
            absolute_path = os.path.abspath(path.name)
            model_path = absolute_path
            load_single_model()
    elif model_choice == "ensemble":
        load_ensemble_models()
    else:
        tkinter.messagebox.showwarning("Warning", "No model was selected.")

change_model()


def change_image():
    accepted_image_extensions = [("PNG File", "*.png"), ("JPG File", "*.jpg"), ("JPEG File", "*.jpeg"),
                                              ("BMP File", "*.bmp"), ("GIF File", "*.gif")]

    image_path = tkinter.filedialog.askopenfile(mode='r', filetypes=accepted_image_extensions, title="Choose an image")
    if image_path == "":
        tkinter.messagebox.showwarning("Warning", "No image was selected.")
    else:
        tb_image_path.delete(0, tkinter.END)
    absolute_path = os.path.abspath(image_path.name)
    tb_image_path.insert(0, absolute_path)


def print_summary():
    global model_summary

    message_window = tkinter.Toplevel(window)
    message_window.title("Model summary")
    tkinter.Message(message_window, text=model_summary, bg="white").pack()

def clear_all():
    if len(tb_template_window.get()) != 0:
        tb_template_window.delete(0, tkinter.END)
        tb_template_window.insert(0, "0")
    if len(tb_search_window.get()) != 0:
        tb_search_window.delete(0, tkinter.END)
        tb_search_window.insert(0, "0")
    if len(tb_noise.get()) != 0:
        tb_noise.delete(0, tkinter.END)
        tb_noise.insert(0, "0")
    if len(tb_color_noise.get()) != 0:
        tb_color_noise.delete(0, tkinter.END)
        tb_color_noise.insert(0, "0")


def convert_filters_to_int():
    global template_window, search_window, noise, color_noise

    try:
        template_window = int(tb_template_window.get())
        search_window =  int(tb_search_window.get())
        noise = int(tb_noise.get())
        color_noise = int(tb_color_noise.get())
        return True
    except ValueError as ex:
        return False

original_image = np.ndarray((0, 0))
denoised_image = np.ndarray((0, 0))
image_for_canvas = None
label_for_image = None

def load_image():
    global original_image, image_for_canvas, label_for_image
    absolute_path = tb_image_path.get()

    if absolute_path == "":
        tkinter.messagebox.showwarning("Error", "Please select an image.")
        return

    if acceptable_image_format(absolute_path):
        original_image = cv.imread(absolute_path)
        load_image_for_canvas(original_image)
    else:
        tkinter.messagebox.showwarning("Error", "Image extension is not supported by Tensorflow..")


def load_image_for_canvas(bgr_image: np.ndarray):
    rgb_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
    resized_image = Image.fromarray(rgb_image).resize((512, 512))
    photoimage = ImageTk.PhotoImage(resized_image)

    label_for_image = tkinter.Label(image=photoimage)
    label_for_image.image = photoimage
    label_for_image.place(x=1019, y=200)

def acceptable_image_format(absolute_path: str):
    accepted_image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".gif"]
    for extension in accepted_image_extensions:
        if absolute_path.find(extension):
            return True
    return False


def denoise_image():
    global original_image, image_for_canvas, denoised_image, template_window, search_window, noise, color_noise, label_for_image

    if convert_filters_to_int():
        denoised_image = cv.fastNlMeansDenoisingColored(src=original_image, templateWindowSize=template_window,
                                                        searchWindowSize=search_window,
                                                        h=noise, hColor=color_noise)
    else:
        tkinter.messagebox.showinfo("Error", "Please use integers only.")

    load_image_for_canvas(denoised_image)

labels = ["animal", "car", "human"]
yhats = []

def predict():

    if tb_image_path.get() == "":
        tkinter.messagebox.showerror("Error", "Please load an image.")
    elif not acceptable_image_format(tb_image_path.get()):
        tkinter.messagebox.showinfo("Info", "Image is invalid. Check path and extension.")
    elif model_path == "" and model_choice == "single":
        tkinter.messagebox.showerror("Error", "Please load a model.")
    elif ensemble_models == [] and model_choice == "ensemble":
        tkinter.messagebox.showerror("Error", "Please load a model.")
    denoise_image()

    if (model_choice == "single"):
        config = model.get_config()
    else:
        config = ensemble_models[0].get_config()

    width, height = list(config["layers"][0]["config"]["batch_input_shape"])[1:3]

    input_image = tf.image.resize(denoised_image, (width, height))

    if model_choice == "single":
        predict_with_single_model(input_image)
    elif model_choice == "ensemble":
        predict_with_ensemble_models(input_image)
    else:
        tkinter.messagebox.showinfo("Info", "No model was selected.")


lbl_prediction_label = Label(window, text="", font=("Heebo Light", 24 * -1))
lbl_prediction_label.place(x=1150, y=750)
lbl_prediction_label.pack()
lbl_probability = Label(window, text="", font=("Heebo Light", 24 * -1))
lbl_probability.place(x=1200, y=810)
lbl_probability.pack()
lbl_all_probability = Label(window, text="", font=("Heebo Light", 24 * -1))
lbl_all_probability.place(x= 1080, y=860)
lbl_all_probability.pack()

def get_article(label: str):
    if label == "animal":
        return "an"
    else:
        return "a"


def predict_with_single_model(input_image):
    global yhats, labels, lbl_prediction_label, lbl_probability, lbl_all_probability

    prediction = model.predict(np.expand_dims(input_image / 255, 0))
    yhats = prediction.tolist()[0]
    print(yhats)
    label = labels[yhats.index(max(yhats))]
    probability = max(yhats)

    change_text(lbl_prediction_label, "This image shows {} {}.".format(get_article(label), label))
    lbl_prediction_label.place(x=1150, y=750)
    change_text(lbl_probability, "Probability: {}".format(round(probability, 2)))
    lbl_probability.place(x=1200, y=810)
    change_text(lbl_all_probability, "Animal: {}, Car: {}, Human: {}".format(round(yhats[0], 3),
                                                                             round(yhats[1], 3),
                                                                             round(yhats[2], 3)))
    lbl_all_probability.place(x=1080, y=870)


def predict_with_ensemble_models(input_image):
    global yhats, labels, lbl_prediction_label, lbl_probability, lbl_all_probability, ensemble_models
    yhats = []

    for model in ensemble_models:
        yhats.append(model.predict(np.expand_dims(input_image / 255, 0)).tolist()[0])

    for yhat in yhats:
        print(yhat)

    yhat_maximum = yhats[0][0]
    yhat_maximum_index = 0
    for i in range(1, len(yhats)):
        if yhats[i][0] > yhat_maximum:
            yhat_maximum = yhats[i][0]
            yhat_maximum_index = i

    label = labels[yhat_maximum_index]

    change_text(lbl_prediction_label, "This image shows {} {}".format(get_article(label), label))
    lbl_prediction_label.place(x=1150, y=750)
    change_text(lbl_probability, "Probability: {}".format(round(yhat_maximum, 2)))
    lbl_probability.place(x=1200, y=810)
    change_text(lbl_all_probability, "Animal: {}, Car: {}, Human: {}".format(round(yhats[0][0], 3),
                                                                             round(yhats[1][0], 3),
                                                                             round(yhats[2][0], 3)))
    lbl_all_probability.place(x=1080, y=870)


window.resizable(False, False)
window.mainloop()

from pathlib import Path

from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"./assets/frame0")
CACHE_PATH = OUTPUT_PATH / Path(r"../../menu/build/cache/model_choice.txt")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

window = Tk()

window.geometry("640x480")
window.configure(bg = "#7FAAC7")
window.title("Model selection")

canvas = Canvas(
    window,
    bg = "#7FAAC7",
    height = 480,
    width = 640,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
canvas.create_rectangle(
    0.0,
    0.0,
    640.0,
    120.0,
    fill="#F4F4F9",
    outline="")

canvas.create_text(
    194.0,
    34.0,
    anchor="nw",
    text="Choose a method",
    fill="#272932",
    font=("Heebo Regular", 32 * -1)
)

canvas.create_text(
    264.0,
    158.0,
    anchor="nw",
    text="I want to use",
    fill="#272932",
    font=("Heebo Light", 20 * -1)
)

canvas.create_rectangle(
    208.0,
    240.0,
    433.0,
    300.0,
    fill="#7FAAC7",
    outline="")

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
btn_ensemble = Button(
    image=button_image_1,
    borderwidth=-1,
    highlightthickness=0,
    command=lambda: save_selection(False),
    relief="flat"
)
btn_ensemble.place(
    x=208.0,
    y=240.0,
    width=225.0,
    height=60.0
)

canvas.create_rectangle(
    229.0,
    336.0,
    412.0,
    396.0,
    fill="#7FAAC7",
    outline="")

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
btn_single_model = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: save_selection(True),
    relief="flat"
)
btn_single_model.place(
    x=229.0,
    y=336.0,
    width=183.0,
    height=60.0
)

canvas.create_rectangle(
    -10.0,
    119.0,
    640.0,
    129.0,
    fill="#A3C4D6",
    outline="")

def save_selection(is_single_model: bool):
    file = open(CACHE_PATH, "w")

    if is_single_model:
        file.write("single")
    else:
        file.write("ensemble")
    window.quit()

window.resizable(False, False)
window.mainloop()

import tkinter as tk
import cv2 as cv # use cv2 to show
import numpy as np
from tkinter import filedialog
from PIL import Image, ImageTk


def choose_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if file_path:
        load_image(file_path)

def load_image(file_path):

    # image = Image.open(file_path)
    # img = img.resize((400, 300))

    img = ImageTk.PhotoImage(Image.open(file_path))
    image_label.config(image=img)
    image_label.image = img

root = tk.Tk()
root.title("Image viewer")
root.geometry("500x400")

# choose button
choose_button = tk.Button(root, text="Choose Image", command=choose_image)
choose_button.pack(pady=10)

# img = ImageTk.PhotoImage(Image.open("src/messi.jpeg"))
# canvas.create_image(20, 20, anchor=NW, image=img)

image_label = tk.Label(root)
image_label.pack()

root.mainloop()

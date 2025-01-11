import tkinter as tk
from tkinter import filedialog, XView
from PIL import Image, ImageTk, ImageOps, ImageDraw
import cv2  # use cv2 to show
import numpy as np

from libs.PartialConv.test import predict_PConv

GEOMETRY_X=1400
GEOMETRY_Y=600

IMG_SIZE = 512

def load_image(file_path):
    img = cv2.imread(file_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return Image.fromarray(img_rgb)

def display_image(canvas: tk.Canvas, img):
    img = ImageTk.PhotoImage(img)

    # display image inside canvas
    canvas.delete("all")  # Clear the previous image (if any)
    canvas.create_image(0,0, anchor=tk.NW, image=img)
    canvas.image = img  # Keep a reference to the image to avoid garbage collection

class Engine:
    def __init__(self):

        # image
        self.org_img = None
        self.img = None
        self.mask = None
        self.masked_img = None

        # mode
        self.edit_mode = False

        # scroll bar
        self.v_scrollbar = None
        self.h_scrollbar = None

        # ROOT
        self.root = tk.Tk()
        self.root.title("Image viewer")
        self.root.geometry(f"{GEOMETRY_X}x{GEOMETRY_Y}")

        # Create a frame for the main content (image display)
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # CANVA-1 - Image for masking
        self.title1 = tk.Label(self.main_frame, text="Masking", font=("Lato", 14, "bold"), bg="lightgray")
        self.title1.place(x=0, y=0)
        self.canvas1 = tk.Canvas(self.main_frame, width=IMG_SIZE, height=IMG_SIZE)
        self.canvas1.place(x=0, y=20)
        self.canvas1.bind("<Button-1>", self.erase_image)
        self.canvas1.bind("<B1-Motion>", self.erase_image)

        # CANVA-2 - Predict image
        self.title2 = tk.Label(self.main_frame, text="Predict", font=("Lato", 14, "bold"), bg="lightgray")
        self.title2.place(x=600, y=0)
        self.canvas2 = tk.Canvas(self.main_frame, width=IMG_SIZE, height=IMG_SIZE)
        self.canvas2.place(x=600, y=20)

        # SIDE FRAME - BUTTONS
        self.side_frame = tk.Frame(self.root, width=200, bg="lightgrey")
        self.side_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.side_frame.pack_propagate(False)

        self.choose_btn = tk.Button(self.side_frame, text="Choose Image", command=self.choose_image)
        self.choose_btn.pack(pady=10)

        # Button to toggle edit mode
        self.edit_btn = tk.Button(self.side_frame, text=f"Edit Mode = {self.edit_mode}", command=self.toggle_edit_mode)
        self.edit_btn.pack(pady=10)

        # Button to reset masking
        self.reset_btn = tk.Button(self.side_frame, text="Reset", command=self.reset)
        self.reset_btn.pack(pady=10)

        # Button to predict
        self.pred_btn = tk.Button(self.side_frame, text="Predict", command=self.predict)
        self.pred_btn.pack(pady=10)

    def mainloop(self):
        # display
        self.root.mainloop()

    # -----------------------------------------------------------------------------------------------------------------
    def erase_image(self, event):
        if self.edit_mode and self.img:
            # Convert canvas coordinates to image coordinates
            img_x = event.x
            img_y = event.y

            # Draw a small white circle on the image to "erase" that region
            eraser_size = 10

            draw1 = ImageDraw.Draw(self.img)
            draw1.ellipse((img_x - eraser_size, img_y - eraser_size, img_x + eraser_size, img_y + eraser_size),
                         fill="white")

            draw2 = ImageDraw.Draw(self.mask)
            draw2.ellipse((img_x - eraser_size, img_y - eraser_size, img_x + eraser_size, img_y + eraser_size),
                          fill="black")

            # Update the canvas with the modified image
            display_image(self.canvas1, self.img)

    def reset(self):
        self.img = self.org_img
        display_image(self.canvas1, self.img)

    # -----------------------------------------------------------------------------------------------------------------
    def choose_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            # load image from file path
            self.org_img = load_image(file_path)
            self.img = load_image(file_path)

            # create empty mask alias with image
            self.mask = Image.fromarray(np.ones((self.img.size[1], self.img.size[0], 3), dtype=np.uint8) * 255)
            display_image(self.canvas1, self.img)

    # -----------------------------------------------------------------------------------------------------------------
    def toggle_edit_mode(self):
        self.edit_mode = not self.edit_mode
        self.edit_btn.config(text=f"Edit Mode = {self.edit_mode}")

    # -----------------------------------------------------------------------------------------------------------------
    def predict(self):
        predict = predict_PConv(self.org_img, self.mask)
        display_image(self.canvas2, predict)
        # print(type(predict))
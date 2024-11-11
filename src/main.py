import tkinter as tk
from tkinter import filedialog
import cv2  # use cv2 to show
import numpy as np
from PIL import Image, ImageTk, ImageOps, ImageDraw

# Initialize a variable to store the current image
current_image = None
eraser_size = 10  # Default eraser size
edit_mode = False


def choose_image():
    global current_image
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if file_path:
        # load_image(file_path)
        current_image = load_image(file_path)
        display_image(current_image)


def load_image(file_path):
    # image = Image.open(file_path)
    # img = img.resize((400, 300))

    # Read the image using OpenCV
    img = cv2.imread(file_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (750, 750))

    return Image.fromarray(img_rgb)


def display_image(img):
    img = ImageTk.PhotoImage(img)

    # display image inside canvas
    canvas.delete("all")  # Clear the previous image (if any)
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    x = (canvas_width - 750) // 2  # Center horizontally
    y = (canvas_height - 750) // 2  # Center vertically
    canvas.create_image(x, y, anchor=tk.NW, image=img)
    canvas.image = img  # Keep a reference to the image to avoid garbage collection


def apply_grayscale():
    global current_image
    if current_image:
        # Convert to grayscale
        grayscale_image = ImageOps.grayscale(current_image)
        display_image(grayscale_image)
        current_image = grayscale_image  # Update the current image to grayscale


# Function to toggle edit mode
def toggle_edit_mode():
    global edit_mode
    edit_mode = not edit_mode
    edit_btn.config(text="Disable Edit Mode" if edit_mode else "Enable Edit Mode")


def erase_image(event):
    global current_image
    if edit_mode and current_image:
        # Calculate click position relative to the image
        canvas_x = (canvas.winfo_width() - 750) // 2
        canvas_y = (canvas.winfo_height() - 750) // 2

        # Erase only if the click is within the image bounds
        if canvas_x <= event.x <= canvas_x + 750 and canvas_y <= event.y <= canvas_y + 750:
            # Convert canvas coordinates to image coordinates
            img_x = event.x - canvas_x
            img_y = event.y - canvas_y

            # Draw a small white circle on the image to "erase" that region
            draw = ImageDraw.Draw(current_image)
            eraser_size = 10
            draw.ellipse((img_x - eraser_size, img_y - eraser_size, img_x + eraser_size, img_y + eraser_size),
                         fill="white")

            # Update the canvas with the modified image
            display_image(current_image)


root = tk.Tk()
root.title("Image viewer")
root.geometry("1200x800")

# Create a frame for the main content (image display)
main_frame = tk.Frame(root)
main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Create a canvas to display images inside the main frame
canvas = tk.Canvas(main_frame)
canvas.pack(fill=tk.BOTH, expand=True)
canvas.bind("<Button-1>", erase_image)
canvas.bind("<B1-Motion>", erase_image)

# Side frame - buttons
side_frame = tk.Frame(root, width=300, bg="lightgrey")
side_frame.pack(side=tk.RIGHT, fill=tk.Y)
side_frame.pack_propagate(False)

choose_btn = tk.Button(side_frame, text="Choose Image", command=choose_image)
choose_btn.pack(pady=10)

# Button to toggle edit mode
edit_btn = tk.Button(side_frame, text="Enable Edit Mode", command=toggle_edit_mode)
edit_btn.pack(pady=10)

grayscale_btn = tk.Button(side_frame, text="Grayscale", command=apply_grayscale)
grayscale_btn.pack(pady=10)

inpainting_btn = tk.Button(side_frame, text="Inpainting", command=apply_grayscale)
inpainting_btn.pack(pady=10)

root.mainloop()

import os
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from time import sleep
from tqdm import tqdm
import sys
sys.path.append("..")
from Yolo import predict

# Specify the directory containing the images
image_dir = 'RawData'
save_dir = '/Users/tobiasvontobel/Polybox/Master/Semester_4/Semester_Project/captcha_project/data/train'

# Define the class directories
class_dirs = ['bicycle', 'bridge', 'bus', 'car', 'chimney', 'crosswalk', 'hydrant', 'motorcycle', 'mountain', 'other', 'palm', 'traffic', 'stairs']
folder_dirs = ['Bicycle', 'Bridge', 'Bus', 'Car', 'Chimney', 'Crosswalk', 'Hydrant', 'Motorcycle', 'Mountain', 'Other', 'Palm', 'Traffic Light', 'Stairs']

# Create the main window
root = tk.Tk()

# Create a label for the image
image_label = tk.Label(root)
image_label.pack()



# Create a dictionary to store the buttons
buttons = {}


# Create a list to store the image filenames
image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
image_filenames.sort() #make sure that images are sorted after name, such that all images with same classes come after each other
current_image_index = 0
pbar = tqdm(total=len(image_filenames))

# Create a mapping from class names to folder names
class_to_folder = {class_dir: folder_dir for class_dir, folder_dir in zip(class_dirs, folder_dirs)}

# Function to update the image and button highlight
def update_image_and_button():
    global current_image_index

    # Get the current filename
    filename = image_filenames[current_image_index]

    absolute_path = os.path.abspath(os.path.join(image_dir, filename))
    predict.predict_tile_with_shap(absolute_path)

    # Open the image
    img = Image.open(os.path.join(image_dir, filename))
    
    # Resize the image
    img = img.resize((120, 120))
    
    # Update the image label
    tk_img = ImageTk.PhotoImage(img)
    image_label.config(image=tk_img)
    image_label.image = tk_img

    # Get the class from the filename
    class_name = filename.split('_')[0]

    # Reset all button colors
    for button in buttons.values():
        button.config(fg='black', activeforeground='black')

    # Highlight the button that corresponds to the class of the current image
    if class_name in class_to_folder:
        folder_name = class_to_folder[class_name]
        if folder_name in buttons:
            buttons[folder_name].config(fg='red', activeforeground='red')

# Create a function to handle button clicks
def handle_button_click(folder_dir):
    global current_image_index

    # Save the image in the specified folder directory
    filename = image_filenames[current_image_index]
    img = Image.open(os.path.join(image_dir, filename))
    new_filename = folder_dir

    # Create the directory if it does not exist
    os.makedirs(os.path.join(save_dir, folder_dir), exist_ok=True)

    extension = '.png'
    i = 0
    while os.path.exists(os.path.join(save_dir, folder_dir, new_filename + extension)):
        i += 1
        new_filename = filename[:-4] + '_' + str(i)

    img.save(os.path.join(save_dir, folder_dir, new_filename + extension), 'PNG')

    # Delete the image from the original folder
    os.remove(os.path.join(image_dir, filename))

    # Move to the next image
    current_image_index += 1
    if current_image_index < len(image_filenames):
        update_image_and_button()
    
    # Update the progress bar
    pbar.update(1)

# Clear all existing buttons
for button in buttons.values():
    button.destroy()
buttons.clear()

# Create a frame for the folder buttons
button_frame = tk.Frame(root)
button_frame.pack()

# Create buttons for the folders
for i, folder_dir in enumerate(folder_dirs):
    button = tk.Button(button_frame, text=folder_dir, command=lambda folder_dir=folder_dir: handle_button_click(folder_dir))
    button.grid(row=i//5, column=i%5)  # Arrange the buttons in rows of 5
    buttons[folder_dir] = button

def delete_image():
    global current_image_index

    # Delete the current image
    filename = image_filenames[current_image_index]
    os.remove(os.path.join(image_dir, filename))

    # Remove the filename from the list
    del image_filenames[current_image_index]

    # If there are still images left, update the image and button
    if image_filenames:
        update_image_and_button()

    # Update the progress bar
    pbar.update(1)

# Create a button to delete the current image
delete_button = tk.Button(root, text='Delete', command=delete_image)
delete_button.pack(pady=50)

# Start the updates
update_image_and_button()

# Start the main loop
root.mainloop()
pbar.close()
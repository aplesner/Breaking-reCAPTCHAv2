from selenium import webdriver 
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys 
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from time import sleep
import os
import requests
from PIL import Image
from io import BytesIO
import random
import tensorflow as tf
import tensorflow.keras as keras 
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from Yolo import predict
from datetime import datetime
import time
from tensorflow.keras.optimizers.legacy import Adam
from tqdm import tqdm


# Constants
CAPTCHA_URL = "https://www.google.com/recaptcha/api2/demo"
THRESHOLD = 0.2
COLLECT_TYPE1 = False
COLLECT_TYPE2 = True
CLASSES = ["bicycle", "bridge", "bus", "car", "chimney", "crosswalk", "hydrant", "motorcycle", "other", "palm", "stairs", "traffic"]
EXCLUDED_CLASSES = ["car"]
YOLO_CLASSES = ['bicycle', 'bridge', 'bus', 'car', 'chimney', 'crosswalk', 'hydrant', 'motorcycle', 'mountain', 'other', 'palm', 'traffic']
YOLO_CLASSES = ['bicycle', 'bridge', 'bus', 'car', 'chimney', 'crosswalk', 'hydrant', 'motorcycle', 'mountain', 'other', 'palm', 'traffic', 'stairs']

MODEL = "YOLO" # "YOLO"

image_count = 0
total_images = 100

#suppress tensorflow logs
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

# Check if data dir is present
data_dir = os.path.join(os.getcwd(), "data/RawData")
os.makedirs(data_dir, exist_ok=True) 
    


# main functions

def getModel():
    f = tf.keras.utils.get_file(
                                fname="model.h5",
                                origin="https://github.com/ajmandourah/Deep-reCaptcha/releases/download/model/efficentM2.h5",
                                cache_dir=data_dir,
                                cache_subdir='model')
    model = keras.models.load_model(f, compile=False)
    model.compile(optimizer=Adam(),
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                 metrics=["accuracy"])
    return model

def getFirstModel():
    model_path = "data/model/first_model.h5"
    model = keras.models.load_model(model_path, compile=False)
    model.compile(optimizer=Adam(),
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                 metrics=["accuracy"])
    return model


def sliceImage(path):
    from PIL import Image
    img = np.array(Image.open(path))

    ys = img.shape[0] // 3
    xs = img.shape[1] // 3

    topLeft = img[0:ys,0:xs]
    topMid = img[0:ys,xs:xs*2]
    topRight = img[0:ys,xs*2:xs*3]
    midLeft = img[ys:ys*2,0:xs]
    midmid = img[ys:ys*2,xs:xs*2]
    midRight = img[ys:ys*2,xs*2:xs*3]
    bottomLeft = img[ys*2:ys*3,0:xs]
    bottomMid = img[ys*2:ys*3,xs:xs*2]
    bottomRight = img[ys*2:ys*3,xs*2:xs*3]
    
    return [topLeft, topMid, topRight, midLeft, midmid, midRight, bottomLeft, bottomMid, bottomRight]

def predict_tile(tile, model):    
    #resize the image

    i = img_to_array(tile)
    to_predict = i.reshape((-1,224,224,3))
    prediction = model.predict(to_predict)
    #return a list of the prediction array, the class name with highest probability and its index 
    return [ prediction, CLASSES[np.argmax(prediction)], np.argmax(prediction)  ]

COUNT = 0
def process_tile(i, model, captcha_object, class_index, driver):
    global COUNT
    global image_count  

    xpath = "//td[contains(@tabindex, '" + str(i+4)+ "')]"
    matched_tile = driver.find_element(By.XPATH, xpath)

    # Add the timestamp to the filename
    timestamp = str(time.time())
    filename = f"tile_{COUNT}_{timestamp}.png"  # Save the screenshot as .png

    # Use the filename when saving and loading the image
    matched_tile.screenshot(os.path.join(data_dir, filename))
    img = Image.open(os.path.join(data_dir, filename)).convert('RGB')
    img = img.resize(size=(224,224))

    # Save the image as .jpg
    jpg_filename = filename.replace('.png', '.jpg')
    img.save(os.path.join(data_dir, jpg_filename))
    image_count += 1

    # Delete the .png file
    os.remove(os.path.join(data_dir, filename))

    if MODEL == "YOLO":
        result = predict.predict_tile(os.path.join(data_dir, jpg_filename))
        current_object_probability = result[0][class_index]
        object_name = YOLO_CLASSES[result[2]]
    else:
        result = predict_tile(img, model)
        current_object_probability = result[0][0][class_index]
        object_name = CLASSES[result[2]]

    #rename image
    os.rename(os.path.join(data_dir, jpg_filename), os.path.join(data_dir, object_name + "_" + jpg_filename))

    compare_probability = result[2] * THRESHOLD
    
    COUNT += 1
    if object_name in captcha_object.text or current_object_probability > THRESHOLD:
        #matched_tile.click()
        return True
    else:
        return False
    
def process_type2(driver):
    global image_count
    try:
        save_path = "data/RawDataType2"
        xpath_image = "/html/body/div/div/div[2]/div[2]/div/table/tbody/tr[1]/td[1]/div/div[1]/img"

        xpath_text = "/html/body/div/div/div[2]/div[1]/div[1]/div/strong"
        captcha_text = driver.find_element(By.XPATH, xpath_text)
        captcha_text = captcha_text.text

        img = driver.find_element(By.XPATH, xpath_image)
        img_url = img.get_attribute("src")
        response = requests.get(img_url, stream=True)
        # Check if the request was successful
        if response.status_code == 200:
            # Get the current timestamp and convert it to a string
            timestamp = str(time.time())
            # Open a file in write-binary mode and save the image to it
            with open(os.path.join(save_path, "image_" + captcha_text + "_" + timestamp + ".png"), 'wb') as f:
                f.write(response.content)
                image_count += 1
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def run():
    #getting the model from github
    model = getFirstModel()

    # Setup WebDriver
    driver = webdriver.Firefox()


    # Set window size to half the screen width
    driver.set_window_size(600, 600)

    driver.get(CAPTCHA_URL)

    recapcha_frame = driver.find_element(By.XPATH, "//iframe[@title='reCAPTCHA']")

    driver.switch_to.frame(recapcha_frame)

    driver.find_element(By.CLASS_NAME, "recaptcha-checkbox-border").click()
    sleep(4)
    driver.switch_to.default_content()
    WebDriverWait(driver, 20).until(EC.frame_to_be_available_and_switch_to_it((By.XPATH,"//iframe[@title='recaptcha challenge expires in two minutes']")))


    #The main loop
    solved = False

    while not solved:
        reload = driver.find_element(By.ID, "recaptcha-reload-button")
        title_wrapper = driver.find_element(By.ID, 'rc-imageselect')
        
        #Skipping if it is a 4x4 captcha...will be implementing it later on .
        if "squares" in title_wrapper.text and COLLECT_TYPE2:
            process_type2(driver)
            break
        
        if "none" in title_wrapper.text and COLLECT_TYPE1:
            dynamic_captcha = True
        elif COLLECT_TYPE1:
            dynamic_captcha = False
        else:
            reload.click()
            continue

        #Get the object of the captcha where we suppose to look for
        captcha_object = title_wrapper.find_element(By.TAG_NAME, 'strong')
        for i in EXCLUDED_CLASSES:
            if i in captcha_object.text:
                continue
        
        #get the class index of the captcha object if found.
        class_index = CLASSES.index("other")
        for i in CLASSES:
            if i in captcha_object.text:
                if MODEL == "YOLO":
                    try:
                        class_index = YOLO_CLASSES.index(i)
                    except:
                        class_index = CLASSES.index("other")
                        continue
                else:
                    class_index = CLASSES.index(i)   
        
        
        #first run of solving the captcha       
        to_check = []
        for i in range(9):
            process_tile(i, model, captcha_object, class_index, driver)
        solved = True
    driver.close()

# Create a progress bar
pbar = tqdm(total=total_images)

while image_count < total_images:
    image_count_before = image_count
    run()
    image_count_after = image_count
    pbar.update(image_count_after - image_count_before)

pbar.close()
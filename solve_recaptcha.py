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
import tensorflow as tf
import tensorflow.keras as keras 
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from models.YOLO_Classification import predict
from models.YOLO_Segment import predict as predict_segment
import time
import csv
from datetime import datetime
from IP import vpn
from selenium.webdriver import ActionChains
from pynput.mouse import Button, Controller
import random
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.common.exceptions import WebDriverException
import traceback


# Constants
CAPTCHA_URL = "https://www.google.com/recaptcha/api2/demo"
THRESHOLD = 0.2
USE_TOP_N_STRATEGY = False
N = 3
CLASSES = ["bicycle", "bridge", "bus", "car", "chimney", "crosswalk", "hydrant", "motorcycle", "other", "palm", "stairs", "traffic"]
YOLO_CLASSES = ['bicycle', 'bridge', 'bus', 'car', 'chimney', 'crosswalk', 'hydrant', 'motorcycle', 'mountain', 'other', 'palm', 'traffic']
MODEL = "YOLO" # "YOLO"
TYPE1 = True #one time image selection
TYPE2 = True #segmentation problem
TYPE3 = True #dynamic captcha
ENABLE_LOGS = True
ENABLE_VPN = False
ENABLE_MOUSE_MOVEMENT = True
ENABLE_NATURAL_MOUSE_MOVEMENT = True
ENABLE_COOKIES = True
PATH_TO_FIREFOX_PROFILE = '.../Application Support/Firefox/Profiles/wtjovf77.default-release'

def set_variables(variables):
    global CAPTCHA_URL, THRESHOLD, USE_TOP_N_STRATEGY, N, CLASSES, YOLO_CLASSES, MODEL, TYPE1, TYPE2, TYPE3, ENABLE_LOGS, ENABLE_VPN, ENABLE_MOUSE_MOVEMENT, ENABLE_NATURAL_MOUSE_MOVEMENT, ENABLE_COOKIES
    if 'CAPTCHA_URL' in variables:
        CAPTCHA_URL = variables['CAPTCHA_URL']
    if 'THRESHOLD' in variables:
        THRESHOLD = variables['THRESHOLD']
    if 'USE_TOP_N_STRATEGY' in variables:
        USE_TOP_N_STRATEGY = variables['USE_TOP_N_STRATEGY']
    if 'N' in variables:
        N = variables['N']
    if 'CLASSES' in variables:
        CLASSES = variables['CLASSES']
    if 'YOLO_CLASSES' in variables:
        YOLO_CLASSES = variables['YOLO_CLASSES']
    if 'MODEL' in variables:
        MODEL = variables['MODEL']
    if 'TYPE1' in variables:
        TYPE1 = variables['TYPE1']
    if 'TYPE2' in variables:
        TYPE2 = variables['TYPE2']
    if 'TYPE3' in variables:
        TYPE3 = variables['TYPE3']
    if 'ENABLE_LOGS' in variables:
        ENABLE_LOGS = variables['ENABLE_LOGS']
    if 'ENABLE_VPN' in variables:
        ENABLE_VPN = variables['ENABLE_VPN']
    if 'ENABLE_MOUSE_MOVEMENT' in variables:
        ENABLE_MOUSE_MOVEMENT = variables['ENABLE_MOUSE_MOVEMENT']
    if 'ENABLE_NATURAL_MOUSE_MOVEMENT' in variables:
        ENABLE_NATURAL_MOUSE_MOVEMENT = variables['ENABLE_NATURAL_MOUSE_MOVEMENT']
    if 'ENABLE_COOKIES' in variables:
        ENABLE_COOKIES = variables['ENABLE_COOKIES']

#suppress tensorflow logs
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

# Check if data dir is present
data_dir = os.path.join(os.getcwd(), "data")
os.makedirs(data_dir, exist_ok=True)
    
def getFirstModel():
    model_path = "models/Base_Line/first_model.h5"
    model = keras.models.load_model(model_path, compile=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                 metrics=["accuracy"])
    return model



def generate_bezier_curve(p0, p1, p2, p3, num_points=100):
    """ Generate points along a Bezier curve using four control points. """
    curve = []
    for t in np.linspace(0, 1, num_points):
        point = (1-t)**3 * p0 + 3 * (1-t)**2 * t * p1 + 3 * (1-t) * t**2 * p2 + t**3 * p3
        curve.append(point)
    return curve



def move_mouse_in_curve(mouse, start_pos, end_pos):
    """ Move the mouse in a curve from start_pos to end_pos. """
    # Generate random control points for the Bezier curve
    ctrl1 = start_pos + np.random.rand(2) * 100
    ctrl2 = end_pos + np.random.rand(2) * 100

    # Create the Bezier curve
    curve = generate_bezier_curve(np.array(start_pos), ctrl1, ctrl2, np.array(end_pos))

    # Move the mouse through the curve points
    for point in curve:
        mouse.position = point
        time.sleep(random.uniform(0.003, 0.01))




mouse = Controller()

def click_element(driver, element, offset=True):
    # Execute JavaScript to get the absolute position of the element
    if ENABLE_MOUSE_MOVEMENT:
        x = driver.execute_script('return arguments[0].getBoundingClientRect().left + window.pageXOffset', element)
        y = driver.execute_script('return arguments[0].getBoundingClientRect().top + window.pageYOffset', element)

        # Add half the size of the element to get the center
        if offset:
            x += 110
            y += 220
        else:
            x += 30
            y += 440
        

        if ENABLE_NATURAL_MOUSE_MOVEMENT:
            start_pos = np.array(mouse.position)
            end_pos = np.array([x, y])
            move_mouse_in_curve(mouse, start_pos, end_pos)
        else:
            # Move the mouse to the element and click on it
            mouse.position = (x, y)
        sleep(0.1)
        mouse.click(Button.left, 1)
        sleep(0.2)
    else:
        element.click()


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
    print("processing tile with class index ", str(class_index))

    xpath = "//td[contains(@tabindex, '" + str(i+4)+ "')]"
    matched_tile = driver.find_element(By.XPATH, xpath)

    # Add the timestamp to the filename
    filename = f"tile_{COUNT}.jpg"

    # Use the filename when saving and loading the image
    matched_tile.screenshot(os.path.join(data_dir, filename))
    img = Image.open(os.path.join(data_dir, filename)).convert('RGB')
    img = img.resize(size=(224,224))

    if MODEL == "YOLO":
        result = predict.predict_tile(os.path.join(data_dir, filename))
        current_object_probability = result[0][class_index]
        object_name = YOLO_CLASSES[result[2]]
    else:
        result = predict_tile(img, model)
        current_object_probability = result[0][0][class_index]
        object_name = CLASSES[result[2]]

    #rename image
    os.rename(os.path.join(data_dir, filename), os.path.join(data_dir, object_name + "_" + filename))

    print(str(COUNT) + ": The AI predicted tile to be ", object_name, "and probability is",current_object_probability )
    
    COUNT += 1
    if USE_TOP_N_STRATEGY:
        # Use the top-N strategy
        top_n_indices = sorted(range(len(result[0])), key=lambda i: result[0][i], reverse=True)[:N]
        if class_index in top_n_indices:
            click_element(driver, matched_tile)
            return True
    else:
        # Use the threshold strategy object_name in captcha_object.text or
        if current_object_probability > THRESHOLD:
            print(current_object_probability, " > ", THRESHOLD)
            click_element(driver, matched_tile)
            return True

    #print(" not a match .. skipping!")
    return False



def solve_type2(driver):
    save_path = "temp"
    os.makedirs(save_path, exist_ok=True)
    xpath_image = "/html/body/div/div/div[2]/div[2]/div/table/tbody/tr[1]/td[1]/div/div[1]/img"

    xpath_text = "/html/body/div/div/div[2]/div[1]/div[1]/div/strong"
    captcha_text = driver.find_element(By.XPATH, xpath_text)
    captcha_text = captcha_text.text

    log("Type2", captcha_text)

    #get the class index of the captcha object if found.
    for i in CLASSES:
        if i in captcha_text:
            class_index = CLASSES.index(i)
            #print("class index is ", str(class_index))

    img = driver.find_element(By.XPATH, xpath_image)
    img_url = img.get_attribute("src")
    response = requests.get(img_url, stream=True)
    # Check if the request was successful
    if response.status_code == 200:
        # Get the current timestamp and convert it to a string
        timestamp = str(time.time())
        # Open a file in write-binary mode and save the image to it
        filename = f"image_{captcha_text}_{timestamp}.png"  # Save the screenshot as .png
        with open(os.path.join(save_path, filename), 'wb') as f:
            f.write(response.content)
        success, grid = predict_segment.predict(class_index, os.path.join(save_path, filename))
        #print(grid)

        xpath_tiles = "/html/body/div/div/div[2]/div[2]/div/table/tbody"
        tiles_to_click = [(i+1, j+1) for i in range(4) for j in range(4) if grid[i][j] == 1]
        for i, j in tiles_to_click:
            tile = driver.find_element(By.XPATH, xpath_tiles + f"/tr[{i}]/td[{j}]")
            click_element(driver, tile)
            sleep(0.5)
    click_element(driver, driver.find_element(By.ID, "recaptcha-verify-button"))
    sleep(0.5)



def open_browser_with_captcha():
    if ENABLE_VPN:
        vpn.connect()
        print("VPN connected")

    if ENABLE_COOKIES:
        options = Options()
        options.profile = PATH_TO_FIREFOX_PROFILE
        # Initialize the WebDriver with the specified options
        driver = webdriver.Firefox(options=options)
        print("init with cookies")
    else:
        driver = webdriver.Firefox()

    driver.maximize_window()

    for _ in range(10):
        try:
            driver.get(CAPTCHA_URL)
            WebDriverWait(driver, 10).until(lambda d: d.execute_script('return document.readyState') == 'complete')

            sleep(1)
            recapcha_frame = driver.find_element(By.XPATH, "//iframe[@title='reCAPTCHA']")

            driver.switch_to.frame(recapcha_frame)

            driver.find_element(By.CLASS_NAME, "recaptcha-checkbox-border").click()
            #click_element(driver, driver.find_element(By.CLASS_NAME, "recaptcha-checkbox-border"), offset=False)
            sleep(4)
            driver.switch_to.default_content()
            WebDriverWait(driver, 3).until(EC.frame_to_be_available_and_switch_to_it((By.XPATH,"//iframe[@title='recaptcha challenge expires in two minutes']")))
            print("opened the browser with the captcha")
            return driver
        except WebDriverException:
            print("An error occurred. Reloading the page and trying again.")
            driver.refresh()
    print("Failed to open the browser with the captcha after 10 attempts.")
    return None



def get_class_index(captcha_object):
    # Get the class index of the captcha object if found.
    for i in CLASSES:
        if i in captcha_object.text:
            if MODEL == "YOLO":
                class_index = YOLO_CLASSES.index(i)
            else:
                class_index = CLASSES.index(i)
            #print("class index is ", str(class_index))
    return class_index



def handle_dynamic_captcha(driver, model, captcha_object, class_index, to_check):
    if len(to_check) < 1:
        click_element(driver, driver.find_element(By.ID, "recaptcha-verify-button"))
        sleep(1)
    else:
        while True:
            # Loop until no results found
            for i in to_check:
                if process_tile(i, model, captcha_object, class_index, driver):
                    to_check.append(i)
                else:
                    to_check.remove(i)
                sleep(2)

            if len(to_check) < 1:
                #verify
                click_element(driver, driver.find_element(By.ID, "recaptcha-verify-button"))
                sleep(2)
                #check if the text select more appears
                #<div class="rc-imageselect-error-select-more" style="display:none">Please select all matching images.</div>
                try:
                    # Wait for the error message to appear
                    error_message = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "rc-imageselect-error-select-more"))
                    )

                    # Check if the style is 'none'
                    if 'none' in error_message.get_attribute('style'):
                        print("The 'select more images' text did not appear.")
                    else:
                        print("The 'select more images' text appeared.")
                        driver.find_element(By.ID, "recaptcha-reload-button").click()

                except:
                    print("The 'select more images' text did not appear.")

                break



def captcha_is_solved(driver):
    #wait 1 second
    sleep(1)
    try:
        driver.switch_to.default_content()
        # Switch to the iframe using its XPath
        iframe = driver.find_element(By.XPATH, '/html/body/div[1]/form/fieldset/ul/li[5]/div/div/div/div/iframe')
        driver.switch_to.frame(iframe)
        checkbox = driver.find_element(By.XPATH, '//*[@id="recaptcha-anchor"]')
        if checkbox.get_attribute('aria-checked') == 'true':
            print("captcha is solved")
            return True
        else:
            print("captcha is not solved yet")
            return False
    except:
        #print("captcha element not found")
        return False
    finally:
        # Switch back to the main content
        driver.switch_to.default_content()
        WebDriverWait(driver, 20).until(EC.frame_to_be_available_and_switch_to_it((By.XPATH,"//iframe[@title='recaptcha challenge expires in two minutes']")))
    


def solve_classification_type(driver, model, dynamic_captcha):
     #Get the object of the captcha where we suppose to look for
    captcha_object = driver.find_element(By.ID, 'rc-imageselect').find_element(By.TAG_NAME, 'strong')
    #print("The object to look for is ", captcha_object.text)
    class_index = get_class_index(captcha_object) 
    
    if dynamic_captcha:
        log("dynamic", captcha_object.text)
    else:
        log("Type1", captcha_object.text)
        
        
    #first run of solving the captcha       
    to_check = []
    for i in range(9):
        if process_tile(i, model, captcha_object, class_index, driver):
            to_check.append(i)
            
    if dynamic_captcha:
        handle_dynamic_captcha(driver, model, captcha_object, class_index, to_check)   
    else:
        click_element(driver, driver.find_element(By.ID, "recaptcha-verify-button"))
        



# Global variable to store the log filename for the current session
log_filename = None
session_folder = None

def log(captcha_type, captcha_object):
    global log_filename, session_folder

    if not ENABLE_LOGS:
        return

    # If a session folder doesn't exist, create one
    if session_folder is None:
        # Find the highest existing session number
        highest_session_number = 0
        for dirname in os.listdir('.'):
            if dirname.startswith('Session'):
                session_number = int(dirname[7:])
                highest_session_number = max(highest_session_number, session_number)

        # Create a new session folder with a number one higher than the highest existing session number
        session_folder = f'Session{highest_session_number + 1:02}'
        os.makedirs(session_folder, exist_ok=True)

        # Save the current values of all global variables to a text file in the session folder
        save_global_variables()

    # Find the highest existing log file number
    highest_log_number = 0
    if log_filename is None:
        for filename in os.listdir(session_folder):
            if filename.startswith('logs_'):
                log_number = int(filename[5:7])
                highest_log_number = max(highest_log_number, log_number)

        # Create a new log file with a number one higher than the highest existing log file number
        log_filename = os.path.join(session_folder, f'logs_{highest_log_number + 1:02}.csv')

    with open(log_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        # Get the current time and format it as a string
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([timestamp, captcha_type, captcha_object])



def save_global_variables():
    with open(os.path.join(session_folder, 'global_variables.txt'), 'w') as file:
        file.write(f'CAPTCHA_URL = {CAPTCHA_URL}\n')
        file.write(f'THRESHOLD = {THRESHOLD}\n')
        file.write(f'CLASSES = {CLASSES}\n')
        file.write(f'YOLO_CLASSES = {YOLO_CLASSES}\n')
        file.write(f'MODEL = {MODEL}\n')
        file.write(f'TYPE1 = {TYPE1}\n')
        file.write(f'TYPE2 = {TYPE2}\n')
        file.write(f'TYPE3 = {TYPE3}\n')
        file.write(f'ENABLE_LOGS = {ENABLE_LOGS}\n')
        file.write(f'ENABLE_VPN = {ENABLE_VPN}\n')
        file.write(f'ENABLE_MOUSE_MOVEMENT = {ENABLE_MOUSE_MOVEMENT}\n')
        file.write(f'ENABLE_NATURAL_MOUSE_MOVEMENT = {ENABLE_NATURAL_MOUSE_MOVEMENT}\n')
        file.write(f'ENABLE_COOKIES = {ENABLE_COOKIES}\n')
        file.write(f'USE_TOP_N_STRATEGY = {USE_TOP_N_STRATEGY}\n')
        file.write(f'N = {N}\n')   



def reset_globals():
    global log_filename
    log_filename = None



def run():
    model = getFirstModel()
    try:
        driver = open_browser_with_captcha()
    except:
        vpn.disconnect()
        return False
    while True:
        
        try:
            #check type of captcha
            if "squares" in driver.find_element(By.ID, 'rc-imageselect').text and TYPE2:
                print("found a 4x4 segmentation problem") 
                solve_type2(driver)
            elif "none" in driver.find_element(By.ID, 'rc-imageselect').text and TYPE3:
                print("found a 3x3 dynamic captcha")
                dynamic_captcha = True
                solve_classification_type(driver, model, dynamic_captcha)
            elif TYPE1:
                print("found a 3x3 one time selection captcha")
                dynamic_captcha = False
                solve_classification_type(driver, model, dynamic_captcha)
            else:
                driver.find_element(By.ID, "recaptcha-reload-button").click()
                continue

            
            #check if captcha is solved or still present
            if captcha_is_solved(driver):
                log("SOLVED", "captcha solved")
                driver.close()
                vpn.disconnect()
                break
        except Exception as e:
            print("error occurred:", e)
            traceback.print_exc()
            vpn.disconnect()
            if captcha_is_solved(driver):
                log("SOLVED", "captcha solved")
                driver.close()
                break
            driver.find_element(By.ID, "recaptcha-reload-button").click()
            continue



if __name__ == "__main__":
    run()

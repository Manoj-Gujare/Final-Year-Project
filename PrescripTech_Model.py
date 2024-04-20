import cv2
import numpy as np
import os
import yaml
import pandas as pd
import re
import pytesseract
from yaml.loader import SafeLoader

class YOLO_Pred():
    
    def __init__(self,onnx_model,data_yaml):
        # load YAML
        with open(data_yaml, mode='r') as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)

        self.labels = data_yaml['names']

        # load YOLO model
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
    def predictions(self, image):
        # Resize image to 640x640
        input_image = cv2.resize(image, (640, 640))

        # get the YOLO prediction from the image
        # step-1 convert image into square image(array)
        row, col, d = input_image.shape
        
        # step-2 get predictions from square array
        input_WH_yolo = 640
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (input_WH_yolo, input_WH_yolo), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward() # detection or prediction from YOLO

        # Non Maximum Suppression
        # step-1 filter detection based on confidence (0.5) and probability score (0.3)
        detections = preds[0]
        boxes = []
        confidences = []
        classes = []

        # width and height of the image (input_image)
        image_w, image_h = input_image.shape[:2]
        x_factor = image_w/input_WH_yolo
        y_factor = image_h/input_WH_yolo
        cropped_image = None

        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4] # confidence of detection on object
            if confidence > 0.5:
                class_score = row[5:].max() # maximum probability from rest of the objects
                class_id = row[5:].argmax() # get the index position at which max probability occurs

                if class_score > 0.3:
                    cx, cy, w, h = row[0:4]
                    # construct bounding box from four values
                    # left, top, width and height
                    left = int((cx-0.5*w)*x_factor)
                    top = int((cy-0.5*h)*y_factor)
                    width = int(w*x_factor)
                    height = int(h*y_factor)

                    box = np.array([left, top, width, height])

                    # append values into the list
                    confidences.append(confidence)
                    boxes.append(box)
                    classes.append(class_id)

        # clean
        boxes_np = np.array(boxes).tolist()
        confidences_np = np.array(confidences).tolist()

        # NMS
        index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.3, 0.5)
        
        # draw the bounding box
        for ind in index:
            # extract bounding box
            x, y, w, h = boxes_np[ind]
            bb_conf = int(confidences_np[ind]*100)
            class_id = classes[ind]
            class_name = self.labels[class_id]

            text = f'{class_name}'

            # Modify font settings for better readability
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1
            font_color = (0,0,0)  # Black color for text

            # Calculate text size to position it properly
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = x + (w - text_size[0]) // 2
            text_y = y - 5

            # Draw rectangle for text background - Red
            cv2.rectangle(input_image, (x, y - text_size[1] - 5), (x + w, y), (0, 0,255), -1)

            # Put text on the image
            cv2.putText(input_image, text, (text_x, text_y), font, font_scale, font_color, font_thickness)
            
            # Draw the bounding box with red color
            cv2.rectangle(input_image, (x, y), (x + w, y + h), (0, 0,255), 2)
            cropped_image = input_image[y:y+h, x:x+w]
            
        return input_image, cropped_image

    def remove_special_chars(self, element):
        cleaned_element = re.sub(r'[^a-zA-Z]', '', element)
        return cleaned_element

    def extract_info(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray_image)
        lines = text.split('\n')
        my_list = []
        for i in lines:
            if i:
                my_list.append(i.split()[:3])
        # print(my_list)       
        output_list = []
        for sublist in my_list:
            # Check if the length of the sublist is at least 3 and the 2nd and 3rd elements are strings
            if len(sublist) >= 3 and isinstance(sublist[1], str) and isinstance(sublist[2], str):
                # Merge the 2nd and 3rd elements into a single string
                if self.remove_special_chars(sublist[2]):
                    merged_string = sublist[1] + ' ' + self.remove_special_chars(sublist[2])
                    output_list.append([sublist[0], merged_string] + sublist[3:])
                else:
                    # Append the modified sublist to the output list
                    output_list.append([sublist[0], sublist[1]] + sublist[3:])
            
        return output_list

    def search_info(self, name,pkg):
        # Load CSV file
        data = pd.read_csv('Generic Medicine.csv')
        rows = data[data['Brand'] == name]
        if not rows.empty:
            info_list = []
            for index, row in rows.iterrows():
                generic_name = row['Generic Name']
                if row['Package']==pkg:
                    package = row['Package']
                else:
                    continue
                strength = row['Strength']
                price = row['Price']
                info_list.append((name,generic_name, package, strength, price))
            return info_list
        
    def final_predictions(self,image):
        img_pred, crop = self.predictions(image)
        data = self.extract_info(crop)
        info_list = []
        not_available = []
        for i in data:
            j,k = i
            name_to_search = k
            pack = j
            info = self.search_info(name_to_search.lower(),pack)
            if info is not None:
                info_list.append(info)
            else :
                not_available.append(k)

        # Flatten the list of lists
        flat_list = [item for sublist in info_list for item in sublist]

        # Define column names
        columns = ['Brand Name', 'Generic Name', 'Package', 'Strength', 'Price']

        # Create DataFrame
        df = pd.DataFrame(flat_list, columns=columns)

        return df, not_available

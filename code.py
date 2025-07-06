

import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

ctk.set_appearance_mode("dark")  
ctk.set_default_color_theme("blue")  

root = ctk.CTk()
root.title("Détection des Objets - YOLOv4")
root.geometry("800x750")
root.configure(bg="#FF8C00")  

image_path = None
image = None

def parcourir_image():
    global image_path, image
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if image_path:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        afficher_image(image)

def afficher_image(img):
    img = Image.fromarray(img)
    img.thumbnail((500, 500))  
    img_tk = ImageTk.PhotoImage(img)
    panel.configure(image=img_tk)
    panel.image = img_tk

def detecter_objets():
    global image, image_path
    if image_path:
        net = cv2.dnn.readNet("C:/Users/Lonovo/Desktop/Detections_Des_Objets_IMVD/Detection_Image/DetectionObjetsImages/yolov4.weights",
                              "C:/Users/Lonovo/Desktop/Detections_Des_Objets_IMVD/Detection_Image/DetectionObjetsImages/yolov4.cfg")
        with open("C:/Users/Lonovo/Desktop/Detections_Des_Objets_IMVD/Detection_Image/DetectionObjetsImages/coco.names", "r") as f:
            classes = f.read().strip().split("\n")
        
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward(net.getUnconnectedOutLayersNames())
        
        boxes, confidences, class_ids = [], [], []
        
        for output in detections:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    box = detection[0:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    startX, startY = int(centerX - (width / 2)), int(centerY - (height / 2))
                    boxes.append([startX, startY, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)
        objets_detectes = []
        
        if len(indices) > 0:
            for i in indices.flatten():
                (startX, startY, width, height) = boxes[i]
                label = classes[class_ids[i]]
                objets_detectes.append(label)
                cv2.rectangle(image, (startX, startY), (startX + width, startY + height), (255, 140, 0), 2)  # Orange
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(image, f"{label} {confidences[i]:.2f}", (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 2)
        
        afficher_image(image)
        resultat_text.set(f"Objets détectés : {', '.join(objets_detectes)}\nNombre total : {len(objets_detectes)}")




main_frame = ctk.CTkFrame(root, fg_color="#1C1C1C")  
main_frame.pack(pady=20, padx=20, fill="both", expand=True)


bouton_parcourir = ctk.CTkButton(main_frame, text="📂 Parcourir une image",
                                 command=parcourir_image, fg_color="#FFA500", hover_color="#FF4500",
                                 font=("Arial", 14, "bold"), height=40, corner_radius=10)
bouton_parcourir.pack(pady=10)


panel = ctk.CTkLabel(main_frame, text="", width=500, height=400, fg_color="#FFFFFF", corner_radius=10)
panel.pack(pady=10)


bouton_detecter = ctk.CTkButton(main_frame, text="🔍 Détecter les objets",
                                command=detecter_objets, fg_color="#FFA500", hover_color="#FF4500",
                                font=("Arial", 14, "bold"), height=40, corner_radius=10)
bouton_detecter.pack(pady=10)

resultat_text = ctk.StringVar()
resultat_label = ctk.CTkLabel(main_frame, textvariable=resultat_text, fg_color="#1C1C1C",
                              font=("Arial", 14, "bold"), width=400, height=50, corner_radius=10)
resultat_label.pack(pady=10)

root.mainloop()

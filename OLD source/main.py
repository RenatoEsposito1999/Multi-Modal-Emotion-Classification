import pyrealsense2 as rs
import numpy as np
import cv2
from cnn_model import cnn

cap = None
pipeline = None
# Crea un oggetto context per accedere ai dispositivi RealSense
context = rs.context()
# Elenco dei dispositivi collegati
devices = context.query_devices()
if len(devices) == 0: #No RealSense detected
    # Open a connection to the webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(1)
else:
    # Crea un oggetto config
    config = rs.config()
    # Abilita solo lo stream color (RGB)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Crea un pipeline
    pipeline = rs.pipeline()
    # Inizia lo streaming con la configurazione
    pipeline.start(config)


model = cnn()

while True:
    if pipeline:
        # Ottieni i frames con la realsense
        frames = pipeline.wait_for_frames()
        # Prendi solo il frame RGB
        color_frame = frames.get_color_frame()
        # Converti l'immagine in formato numpy
        color_image = np.asanyarray(color_frame.get_data())
        if not color_frame:
            continue
    else:
        # Capture frame-by-frame con la cam standard
        ret, color_image = cap.read()
        
    
    model.process(color_image)
        
    cv2.imshow('Video',color_image)


    # Premi 'q' per uscire
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Ferma lo streaming
if pipeline:
    pipeline.stop()
else:
    cap.release()
# Chiudi la finestra di OpenCV
cv2.destroyAllWindows()

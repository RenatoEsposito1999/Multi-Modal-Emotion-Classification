import pyrealsense2 as rs
import numpy as np
import cv2
import time
# Crea un oggetto config
config = rs.config()

# Abilita solo lo stream color (RGB)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Crea un pipeline
pipeline = rs.pipeline()

# Inizia lo streaming con la configurazione
pipeline.start(config)

try:
    while True:
        # Ottieni i frames
        frames = pipeline.wait_for_frames()
        
        # Prendi solo il frame RGB
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            continue

        # Converti l'immagine in formato numpy
        color_image = np.asanyarray(color_frame.get_data())

        # Salva il frame come immagine JPEG
        cv2.imwrite('frame_catturato.jpeg', color_image)
        
        print("Frame salvato come frame_catturato.jpeg")
        time.sleep(1)
        break
    '''
        # Visualizza l'immagine (puoi usare OpenCV ad esempio)
        #cv2.imshow('RGB Video', color_image)

        print(color_image)

        # Premi 'q' per uscire
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    '''
finally:
    # Ferma lo streaming
    pipeline.stop()

# Chiudi la finestra di OpenCV
cv2.destroyAllWindows()

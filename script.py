import cv2
import wave
import threading
import time
 

 
'''def record_video(output_video_path, record_time):
    cap = cv2.VideoCapture(0)  # Apre la webcam
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (640, 480))
 
    print("Registrazione video in corso...")
    start_time = time.time()
    while time.time() - start_time < record_time:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            cv2.imshow('Recording', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
 
    print("Registrazione video terminata.")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
 
if __name__ == "__main__":
    durata_minuti = 0.5 # Inserisci la durata desiderata in minuti
    durata_secondi = durata_minuti * 60
 
    video_path = 'output_video.mp4'
    audio_path = 'output_audio.wav'
 
    
    video_thread = threading.Thread(target=record_video, args=(video_path, durata_secondi))
 
    
    video_thread.start()
 
    
    video_thread.join()
 
    print("Registrazione completata.")'''
    
    
import cv2

import time
 
def record_video_with_audio(output_path, record_time):

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Usa DirectShow su Windows, lascia solo 0 su macOS/Linux
 
    # Controlla se la webcam è stata aperta correttamente

    if not cap.isOpened():

        print("Errore nell'apertura della webcam.")

        return
 
    # Imposta il codec e i parametri del video

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Usa XVID per .avi, o mp4v per .mp4

    fps = 20.0

    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
 
    print("Registrazione video con audio in corso...")

    start_time = time.time()

    while time.time() - start_time < record_time:

        ret, frame = cap.read()

        if ret:

            out.write(frame)

            cv2.imshow('Recording', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):

                break

        else:

            break
 
    print("Registrazione terminata.")

    cap.release()

    out.release()

    cv2.destroyAllWindows()
 
if __name__ == "__main__":

    durata_minuti = 0.1  # Inserisci la durata desiderata in minuti

    durata_secondi = durata_minuti * 60

    output_file = 'output_video_with_audio.avi'
 
    record_video_with_audio(output_file, durata_secondi)

 
import numpy as np
import cv2
from tqdm import tqdm
import torch
from facenet_pytorch import MTCNN

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

select_distributed = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
save_avi = True
class Video_preprocessing():
    def __init__(self, video_path):
        self.video_path = video_path
        self.MTCNN = MTCNN(image_size=(720,1280), device=device)
        self.save_frames = 15
        self.input_fps = 30
        self.save_length = 3.6 #seconds
    def process(self):
        cap = cv2.VideoCapture(self.video_path)
        #calculate length in frames
        framen = 0
        while True:
            i, _ = cap.read()
            if not i:
                break
            framen += 1
        cap = cv2.VideoCapture(self.video_path)

        if self.save_length*self.input_fps > framen:                    
            skip_begin = int((framen - (self.save_length*self.input_fps)) // 2)
            for i in range(skip_begin):
                _, im = cap.read() 
                    
        framen = int(self.save_length*self.input_fps)    
        frames_to_select = select_distributed(self.save_frames,framen)
        save_fps = self.save_frames // (framen // self.input_fps) 
        '''if save_avi:
            out = cv2.VideoWriter("Prova.avi",cv2.VideoWriter_fourcc('M','J','P','G'), save_fps, (224,224))
        '''
        numpy_video = []
        frame_ctr = 0
            
        while True: 
            ret, im = cap.read()
            if not ret:
                break
            if frame_ctr not in frames_to_select:
                frame_ctr += 1
                continue
            else:
                frames_to_select.remove(frame_ctr)
                frame_ctr += 1

            temp = im[:,:,-1]
            im_rgb = im.copy()
            im_rgb[:,:,-1] = im_rgb[:,:,0]
            im_rgb[:,:,0] = temp
            im_rgb = torch.tensor(im_rgb)
            im_rgb = im_rgb.to(device)

            bbox = self.MTCNN.detect(im_rgb)
            if bbox[0] is not None:
                bbox = bbox[0][0]
                bbox = [round(x) for x in bbox]
                x1, y1, x2, y2 = bbox
            im = im[y1:y2, x1:x2, :]
            im = cv2.resize(im, (224,224))
            '''if save_avi:
                out.write(im)'''
            numpy_video.append(im)
        if len(frames_to_select) > 0:
            for i in range(len(frames_to_select)):
                '''if save_avi:
                    out.write(np.zeros((224,224,3), dtype = np.uint8))'''
                numpy_video.append(np.zeros((224,224,3), dtype=np.uint8))
        '''if save_avi:
            out.release() '''
        return np.array(numpy_video)
        #if len(numpy_video) != 15:
            #print('Error')
        
        
    
        
        

#!/usr/bin/env python
# coding: utf-8

# In[4]:


import threading
import cv2


# In[5]:


from deepface import DeepFace


# In[ ]:





# In[6]:


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False

# Load the reference image
reference_img = cv2.imread(r"C:\Users\thien\OneDrive\Documents\jupyter\reference_image\bao.jpg") #put your face image here
reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)

while True:
    ret, frame = cap.read()
    
    if ret:
        if counter % 30 == 0:
            try:
                # Perform face verification
                if DeepFace.verify(frame, reference_img.copy(), enforce_detection=False)['verified']:
                    face_match = False
                else:
                    face_match = True
            except ValueError as e:
                print("Error:", str(e))
                face_match = False
        counter += 1
        
        if face_match:
            cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NOT MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        
        cv2.imshow("face recognition", frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:






# coding: utf-8

# In[1]:


# import essential stuff
import numpy as np
import cv2
import matplotlib.pyplot as plt


# In[2]:


# import the haar cascade classifier
face_cascades = cv2.CascadeClassifier("downloads/haarcascade_frontalface_default.xml")


# In[3]:


# load the image
image = cv2.imread ("downloads/lena.jpg")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# In[4]:


# do the detection
faces = face_cascades.detectMultiScale(image_gray, 1.3, 5)
print(faces)


# In[5]:


# draw bounding box over detected faces
for (x, y, width, height) in faces:
    cv2.rectangle(image, (x, y), (x+width, y+height), (255, 0, 0), 2)


# In[6]:


# show the final image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)


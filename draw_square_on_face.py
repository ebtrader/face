import numpy
import cv2
import matplotlib.pyplot as plt

# https://www.datacamp.com/community/tutorials/face-detection-python-opencv?utm_source=adwords_ppc&utm_campaignid=1565261270&utm_adgroupid=67750485268&utm_device=c&utm_keyword=&utm_matchtype=b&utm_network=g&utm_adpostion=&utm_creative=295208661514&utm_targetid=aud-299261629614:dsa-429603003980&utm_loc_interest_ms=&utm_loc_physical_ms=9004005&gclid=Cj0KCQjw5eX7BRDQARIsAMhYLP8w1lcWosrOLMV4vXbzuRqMGv3bprD0cF2ffWhFpzEFe66SQtZ6Ik4aAqDlEALw_wcB

test_image = cv2.imread('cartoon_faces.jpg')

#convert to grayscale
test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

#matplotlib colors
#plt.imshow(test_image)
#plt.show()

#original colors
#img_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
#plt.imshow(img_rgb)
#plt.show()

# Displaying the grayscale image
#plt.imshow(test_image_gray, cmap='gray')
#plt.show()

haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

faces_rects = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor = 1.2, minNeighbors = 5)


# Let us print the no. of faces found
print('Faces found: ', len(faces_rects))

# put a rectangle around the face
for (x,y,w,h) in faces_rects:
     cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

#convert back to original colors
def convertToRGB(test_image):
    return cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

#convert image to RGB and show image
plt.imshow(convertToRGB(test_image))
plt.show()

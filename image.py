import os
import numpy as np
from PIL import Image
import h5py
import cv2

 # Global variables
MALE = np.arange(0, 5)
FEMALE = np.arange(5, 10)

 # Folder path
PATH = "MSFDE"
FILE_FORMAT = (".tif", ".jpg")

 # Get first three digits
def getImageId(name):
	return name[:3]

 images = []
imagesResized = []
sex = []
ethnic = []
emotion = []

 for subdir, dirs, files in os.walk(PATH):
	for file in files:
		if file.endswith(FILE_FORMAT):
			name = os.path.join(subdir, file)
			im = cv2.imread(name, cv2.IMREAD_COLOR)
			im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
			im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

 			# im.show()
			images.append(np.array(im))

 			im = cv2.resize(im, (224, 224))
			imagesResized.append(np.array(im))

 			imageId = getImageId(file)
			if int(imageId[1]) in MALE:
				sex.append(1)
			else:
				sex.append(0)
			ethnic.append(int(imageId[0]) - 1)
			if (imageId[2].isdigit()):
				emotion.append(int(imageId[2]))
			else:
				emotion.append(0)


 # Concatenate
images = np.float64(np.stack(images))
print(images.shape)
imagesResized = np.float64(np.stack(imagesResized))
sex = np.stack(sex)
ethnic = np.stack(ethnic)
emotion = np.stack(emotion)



 # Normalize data
images /= 255.0
imagesResized /= 255.0
# Save to disk
f = h5py.File("images.h5", "w")
# Create dataset to store images
X_dset = f.create_dataset('data', images.shape, dtype='f')
X_dset[:] = images
X_dset = f.create_dataset('dataResized', imagesResized.shape, dtype='f')
X_dset[:] = imagesResized

 # Create dataset to store labels
y_dset = f.create_dataset('sex', sex.shape, dtype='i')
y_dset[:] = sex
y_dset = f.create_dataset('ethnic', ethnic.shape, dtype='i')
y_dset[:] = ethnic
y_dset = f.create_dataset('emotion', emotion.shape, dtype='i')
y_dset[:] = emotion

 f.close()
--------------------------------------------------------------------------------
test emotions;
import cv2
import pickle
import argparse
import numpy as np
import dlib

 IMAGE_SIZE=50
SHAPE_PREDICTOR="shape_predictor_68_face_landmarks.dat"

 EMOTION = {0: "neutral",
	1: "anger",
	2: "joy",
	3: "sadness",
	4: "fear",
	5: "disgust",
	6: "shame"}

 def predict_emotion(name):
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(SHAPE_PREDICTOR)

 	im = cv2.imread(name, cv2.IMREAD_COLOR)
	im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

 	rects = detector(im, 0)

 	for rect in rects:
		face = im[rect.top():rect.bottom(), rect.left():rect.right()].copy() # Crop from x, y, w, h -> 100, 200, 300, 400
		# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
		face = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))
		shape = predictor(face, dlib.rectangle(left=0, top=0, right=IMAGE_SIZE, bottom=IMAGE_SIZE))
		poi = []
		for i in range(17,68):
			poi.append([shape.part(i).x, shape.part(i).y])
		poi = np.array(poi)

 	poi = poi.reshape(poi.shape[0] * poi.shape[1])



 	with open('emotion_classifier.pkl', 'rb') as fid:
	    clf = pickle.load(fid)

 	return clf.predict(np.array([poi]))


 if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True,
		help="path to test image")
	args = vars(ap.parse_args())
	result = predict_emotion(args["image"])
	print(EMOTION[result[0]]) 
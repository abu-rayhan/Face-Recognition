import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime
 
path='imlist'
names=[]
images=[]
mylist=os.listdir(path)  
#print(mylist)
for nm in mylist:
	im=cv2.imread(f'{path}/{nm}')
	images.append(im)
	names.append(os.path.splitext(nm)[0])
#print(names)
 
def findcodes(images):
	enlist=[]
	for im in images:
		im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
		encode=face_recognition.face_encodings(im)[0]
		enlist.append(encode)
	return enlist
def attendance(name):
	with open('attendance.csv','r+') as f:
		datalist=f.readlines()
		#print(datalist)
		namelist=[]
		for line in datalist:
			entry=line.split(',')
			namelist.append(entry[0])
		if name not in namelist:
			now=datetime.now()
			dtstring=now.strftime('%H:%M:%S')
			f.writelines(f'\n{name},  {dtstring}')
 
 
encodelist=findcodes(images)
print('encoding completed')
 
cap=cv2.VideoCapture(0)
 
while True:
	success,img=cap.read()
	img1=cv2.resize(img,(0,0),None,0.25,0.25)
	img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
	faceloc=face_recognition.face_locations(img1)
	encode=face_recognition.face_encodings(img1,faceloc)
 
	for enface,locface in zip(encode,faceloc):
		matches=face_recognition.compare_faces(encodelist,enface)
		dist=face_recognition.face_distance(encodelist,enface)
		indx=np.argmin(dist)
		name='UNKNOWN'
		if matches[indx]:
			name=names[indx]
			attendance(name)
		x1,y1,x2,y2=locface
		x1,y1,x2,y2=x1*4,y1*4,x2*4,y2*4
		cv2.rectangle(img,(y2,x1),(y1,x2),(0,0,255),3)
		cv2.rectangle(img,(y2,x2-35),(y1,x2),(0,0,255),cv2.FILLED)
		cv2.putText(img,name,(y2+6,x2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
	cv2.imshow('live',img)
	if cv2.waitKey(1)&0xFF==ord('q'):
		break
cap.release()
cv2.destroyAllWindows()
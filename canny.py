import cv2
import numpy as np 
import math

def main():
    #initialize webcam 
	cap=cv2.VideoCapture('run.avi')
	cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml');
	

	#initialize bg subtractor
	foreground_background =cv2.createBackgroundSubtractorMOG2()
	array_size=7
	sumar=0
	a=np.zeros(array_size,dtype=int)
	count=-1
	p,q=0,0
	flag=0
	print(a)
	while True:
		ret,frame = cap.read()
		body = cascade.detectMultiScale(frame)
		#print(body)
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		kernel = np.ones((5,5),np.uint8)
		#apply bg subtractor to get foreground mask 
		foreground_mask=foreground_background.apply(frame)
		foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
		hist,bins = np.histogram(foreground_mask.ravel(),256,[0,256])
		cv2.imshow("foreground mask",foreground_mask)
		
		#print(hist[0])
		sum=0
		for i in range(10):
			sum=sum+hist[255-i]
		if(sum>200): #no. of pixels of white 
			#print('object is doing motion')

			#extracting just the moving body using threshhold
			#ret,thresh = cv2.threshold(frame_gray,100,255,0)
			ret,thresh = cv2.threshold(frame_gray,100,255,cv2.THRESH_OTSU)
			thresh=cv2.GaussianBlur(thresh,(5,5),0)
			cv2.imshow("otsu",thresh)
			#applying canny 
			
			kernel = np.ones((5,5),np.uint8)
			#thresh = cv2.dilate(thresh,kernel,iterations=1)
			cannyi=cv2.Canny(thresh,10,250)
			#cannyi = cv2.dilate(cannyi,kernel,iterations=2)
			#cannyi = cv2.erode(cannyi,kernel,iterations=3  )
			cv2.imshow('canny',cannyi)

			
			im2, contours, hierarchy = cv2.findContours(cannyi,cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE)
			#print(len(contours))			
						
			cnt = contours[0]
			maxarea=0			
			for con in contours:
				if cv2.contourArea(con)>maxarea:
					maxarea=cv2.contourArea(con)
					cnt=con
			#print(contours)
			cv2.drawContours(im2,[cnt],0,(0,255,0),1)
			cv2.imshow("contour image",im2)
     			#cv2.drawContours(img, [cnt], 0, (0,255,0), 3)
			#print(cnt)
			
			avgx=0
			avgy=0
			for i in range(len(cnt)):
				avgx=avgx+cnt[0][0][0]
				avgy=avgy+cnt[0][0][1]
				#avg=avg+int(cnt[0][0])
			avgx=int(avgx/len(cnt))
			avgy=int(avgy/len(cnt))
			if flag == 0:
				prex,prey=avgx,avgy
				flag=1
			else:
				
				sub=int(math.sqrt((prex-avgx)*(prex-avgx)+(prey-avgy)*(prey-avgy)))
				#print("sub is done")
				if sub>25:
					continue;
				else:
					if count < array_size-1:
						#print(count)				
						a[count]=sub
						print(sub)
						count=count+1
						sumar=sumar+sub
				
				
					else:
						count=(count+1)%array_size
						a[count]=sub
						#sum = sum(a)
						#print(sumar)
						if sumar>100:
							print(sumar)
						
						sumar=0
									
				
					flag=0

					
					cv2.rectangle(im2,(avgx,avgy),(avgx+7,avgy+7),(0,0,0),2)
				
			cv2.imshow('frame',im2)
		k = cv2.waitKey(0) & 0xFF
		if k == 27:
			
			break
	
	cap.release()
	cv2.closeAllWindows()
if __name__ == '__main__':
	main()

import cv2

def addLabelsToImage(screen, numberList):

     font = cv2.FONT_HERSHEY_SIMPLEX
     cv2.putText(screen,str(numberList[5]),(545,560), font, 2,(255,255,255),2,cv2.LINE_AA)
     cv2.putText(screen,str(numberList[4]),(500,560), font, 2,(255,255,255),2,cv2.LINE_AA)
     cv2.putText(screen,str(numberList[3]),(455,560), font, 2,(255,255,255),2,cv2.LINE_AA)
     cv2.putText(screen,str(numberList[2]),(350,560), font, 2,(255,255,255),2,cv2.LINE_AA)
     cv2.putText(screen,str(numberList[1]),(305,560), font, 2,(255,255,255),2,cv2.LINE_AA)
     cv2.putText(screen,str(numberList[0]),(260,560), font, 2,(255,255,255),2,cv2.LINE_AA)

def addScoreToImage(screen, score) :
     font = cv2.FONT_HERSHEY_SIMPLEX
     cv2.putText(screen,str(score),(620,560), font, 2,(255,255,255),2,cv2.LINE_AA)

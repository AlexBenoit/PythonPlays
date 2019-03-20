import imageProcessing as imgProc
from textAnalyzer import TextAnalyzer
import ImageAnnotator as imA
import cv2

#reward

KILL_OPPONENT = 1000000
DIE = -10000000
INFLICT_DAMAGE = 101
TAKE_DAMAGE = -101
ADVANCE_TOWARD_ENNEMY = 10
GO_BACKWARD_ENNEMY = -10
NEUTRAL = 0

class FrameComparator: 

    def __init__(self):
        self.previousReward = None
        self.previousP1Damage = None
        self.previousP2Damage = None
        self.opponentTransition = False
        self.myTransition = False
        self.digitAnalzer = TextAnalyzer()

    #damagaList contains 6 character, first 3 is ourcharacter, last 3 is ennemy
    def compareWithLastFrame(self, currentFrame):

        # takes the screen above and identifies the zone of the numbers into 6 images
        numberImages = imgProc.processNumber(currentFrame)

        #predict a number for the 6 images
        damageList = self.digitAnalzer.predict(numberImages)

        if(self.previousReward is None):
            self.previousReward = 0
            self.previousP1Damage = 0
            self.previousP2Damage = 0
            return NEUTRAL

        #ignore false read from grabscreen (between numbers transitions)
        reward = 0

       #transition on my character
        if(damageList[2] == ' '):
            if(self.myTransition == True):
                self.myTransition = False
                reward = 0
            else : 
                self.myTransition = False
                reward =  -1
            

          #transition on opponent 
        if(damageList[5] == ' '):
             if(self.opponentTransition == True):
                self.opponentTransition = False
                reward = 0
             else : 
                self.opponentTransition = False
                reward = 1

        ##concatenated numbers
        #previousAllyHealth = self.previousP1Damage
        #previousOpponentHealth = self.previousP2Damage

        #currentAllyHealth = damageList[0]+damageList[1]+damageList[2]
        #currentOpponentHealth = damageList[3]+damageList[4]+damageList[5]

        #currentAllyHealth = currentAllyHealth.replace(' ', '')
        #currentOpponentHealth = currentOpponentHealth.replace(' ', '')

        #allyVariation = int(currentAllyHealth) - int(previousAllyHealth)
        #opponentVariation = int(currentOpponentHealth) - int(previousOpponentHealth)


        #self.previousFrame = currentFrame
        #self.previousP1Damage = int(currentAllyHealth)
        #self.previousP2Damage = int(currentOpponentHealth)

        ## current hp is 0 and previous was greater than 0 which means ally died
        #if (int(currentAllyHealth) == 0 and int(previousAllyHealth) > 0):
        #    self.previousReward = DIE
        #    return DIE

        # # current hp is 0 and previous was greater than 0 which means opponent died
        #if (int(currentOpponentHealth) == 0 and int(previousOpponentHealth) > 0):
        #    self.previousReward = KILL_OPPONENT
        #    return KILL_OPPONENT

        #reward = allyVariation * TAKE_DAMAGE + opponentVariation * INFLICT_DAMAGE

        self.previousReward = reward

        
        #add labels/annotations to the screen image for debugging purpose
        imA.addScoreToImage(currentFrame, reward)
        imA.addLabelsToImage(currentFrame,damageList)

        return reward


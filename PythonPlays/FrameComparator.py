

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
        self.previousFrame = None
        self.previousScore = None
        self.previousPercentageNumbers = None

    #damagaList contains 6 character, first 3 is ourcharacter, last 3 is ennemy
    def compareWithLastFrame(self, currentFrame, damageList):

        #ignore false read from grabscreen (between numbers transitions)
        if(damageList[2] == ' ' or damageList[5] == ' '):
            return NEUTRAL

        if(self.previousFrame is None):
            self.previousFrame = currentFrame
            self.previousScore = 0
            self.previousPercentageNumbers = damageList
            return NEUTRAL

        #concatenated numbers
        previousAllyHealth = self.previousPercentageNumbers[0]+self.previousPercentageNumbers[1]+self.previousPercentageNumbers[2]
        previousOpponentHealth = self.previousPercentageNumbers[3]+self.previousPercentageNumbers[4]+self.previousPercentageNumbers[5]

        currentAllyHealth = damageList[0]+damageList[1]+damageList[2]
        currentOpponentHealth = damageList[3]+damageList[4]+damageList[5]

        #in case middle character is a space, remove it for int casting
        previousAllyHealth = previousAllyHealth.replace(' ', '')
        previousOpponentHealth = previousOpponentHealth.replace(' ', '')
        currentAllyHealth = currentAllyHealth.replace(' ', '')
        currentOpponentHealth = currentOpponentHealth.replace(' ', '')

        print('------------------')
        allyVariation = int(currentAllyHealth) - int(previousAllyHealth)
        opponentVariation = int(currentOpponentHealth) - int(previousOpponentHealth)

        print(allyVariation)
        print(opponentVariation)
          
        print('------------------')


        self.previousFrame = currentFrame
        self.previousPercentageNumbers = damageList

        # current hp is 0 and previous was greater than 0 which means ally died
        if (int(currentAllyHealth) == 0 and int(previousAllyHealth) > 0):
            self.previousScore = DIE
            return DIE

         # current hp is 0 and previous was greater than 0 which means opponent died
        if (int(currentOpponentHealth) == 0 and int(previousOpponentHealth) > 0):
            self.previousScore = KILL_OPPONENT
            return KILL_OPPONENT

        score = allyVariation * TAKE_DAMAGE + opponentVariation * INFLICT_DAMAGE

        self.previousScore = score

        return score


    def print(self):
        print(self.previousFrame)
        print(self.previousScore)
        print(self.previousPercentageNumbers)

        



if __name__ == '__main__':
   print(KILL_OPPONENT)

   frameComparator = FrameComparator()

   frame1 = [1,0,1,1,1,1,1,0,0,1,1,0,0,1,0,1]

   list = [' ','1','2',' ','2','3']

   score = frameComparator.compareWithLastFrame(frame1,list)
   print('SCORE')
   print(score)
   list2 = [' ','1','2',' ','3','1']
   score = frameComparator.compareWithLastFrame(frame1,list2)
   print(score)

   list2 = [' ','1','2',' ','5','1']
   score = frameComparator.compareWithLastFrame(frame1,list2)
   print(score)

   list2 = [' ','2','7',' ','6','1']
   score = frameComparator.compareWithLastFrame(frame1,list2)
   print(score)

   list2 = [' ','3','2',' ','6','6']
   score = frameComparator.compareWithLastFrame(frame1,list2)
   print(score)

   list2 = [' ','3','2',' ','9','3']
   score = frameComparator.compareWithLastFrame(frame1,list2)
   print(score)

   list2 = [' ','3','2',' ','8','7']
   score = frameComparator.compareWithLastFrame(frame1,list2)
   print(score)

   list2 = [' ','3','2',' ',' ','0']
   score = frameComparator.compareWithLastFrame(frame1,list2)
   print(score)

   list2 = [' ','3','2',' ',' ','5']
   score = frameComparator.compareWithLastFrame(frame1,list2)
   print(score)

   list2 = [' ','5','8',' ',' ','5']
   score = frameComparator.compareWithLastFrame(frame1,list2)
   print(score)

   list2 = [' ','5','3',' ',' ','5']
   score = frameComparator.compareWithLastFrame(frame1,list2)
   print(score)

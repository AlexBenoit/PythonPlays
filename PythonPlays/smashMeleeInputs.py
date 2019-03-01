from keyInputs import pressKey as KIpressKey, releaseKey as KIreleaseKey
from keyInputs import X, Z, C, S, RETURN, ARROW_UP, ARROW_DOWN, ARROW_LEFT, ARROW_RIGHT, I, K, J, L, Q, W, T, G, F, H

inputArray = [X, Z, C, S, ARROW_UP, ARROW_DOWN, ARROW_LEFT, ARROW_RIGHT, I, K, J, L, Q, W, T, G, F, H]

def getSmashMeleeInputs():
    #a,b,x,y,z, start, up, down, left, right, cup, cdown, cleft, cright, L, R, dup, ddown, dleft, dright
    #x,z,c,s,d,return,up,down,left,right,i,k,j,l,q,w,t,g,f,h
    return len(inputArray)

def pressKey(index):
    KIpressKey(inputArray[index])

def releaseKey(index):
    KIreleaseKey(inputArray[index])
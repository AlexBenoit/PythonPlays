from keyInputs import pressKey as KIpressKey, releaseKey as KIreleaseKey
from keyInputs import X, Z, C, S, D, RETURN, ARROW_UP, ARROW_DOWN, ARROW_LEFT, ARROW_RIGHT, I, K, J, L, Q, W, T, G, F, H

#a, b, x, y, z, up, down, left, right, cup, cdown, cleft, cright, L
#x, z, c, s, d, up, down, left, right, i,   k,     j,     l,      q
inputArray = [X, Z, C, S, D, ARROW_UP, ARROW_DOWN, ARROW_LEFT, ARROW_RIGHT, I, K, J, L, Q]

def getSmashMeleeInputs():
    return inputArray

def pressKey(index):
    KIpressKey(inputArray[index])

def releaseKey(index):
    KIreleaseKey(inputArray[index])
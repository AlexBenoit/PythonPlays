from keyInputs import pressKey as KIpressKey, releaseKey as KIreleaseKey
from keyInputs import X, Z, C, S, RETURN, ARROW_UP, ARROW_DOWN, ARROW_LEFT, ARROW_RIGHT, I, K, J, L, Q, W, T, G, F, H

inputArray = [X, Z, C, S, ARROW_UP, ARROW_DOWN, ARROW_LEFT, ARROW_RIGHT, I, K, J, L, Q, W, T, G, F, H]

def getSmashMeleeInputs():
    #a, b, x, y, z, up, down, left, right, cup, cdown, cleft, cright, L, R
    #x, z, c, s, d, up, down, left, right, i,   k,     j,     l,      q, w
    return inputArray

def pressKey(index):
    KIpressKey(inputArray[index])

def releaseKey(index):
    KIreleaseKey(inputArray[index])
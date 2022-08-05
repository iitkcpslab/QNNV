from math import ceil, log2, floor
from numpy import clip


class FixedPointData:
    '''
    A class for converting floating point numbers to fixed point numbers with
    given number of total bits.
    '''
    def __init__(self, totalBits=64):
        self.totalBits = totalBits

    def findDataType(self, mn, mx):
        '''
        Find the number of bits required for integer and float part.
        There is a better way to handle this for numbers with really
        small magnitude on both sides. Look into it later. This method
        is wasting bits.
        '''
        if abs(mn) < 1 and abs(mx) < 1:
            return (self.totalBits, self.totalBits - 1)

        intPart = ceil(log2(max(abs(mn), abs(mx)) + 1)) + 1

        if intPart > self.totalBits:
            print("WARNING: Cannot fit IntPart with {} bits into"
                  " total {} bits".format(intPart, self.totalBits))
            intPart = self.totalBits

        floatPart = self.totalBits - intPart
        return (self.totalBits, floatPart)

    def convert(self, num, floatBits):
        '''
        If the number is overflowing, saturate it
        '''
        resNum = int(num * (1 << floatBits))
        lb = -2**(self.totalBits - 1)
        ub = 2**(self.totalBits - 1) - 1
        return clip(resNum, lb, ub)

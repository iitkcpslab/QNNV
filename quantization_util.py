import numpy as np

input_min_value = -8
input_max_value = 7.9375

def quantize(real,num_bits=4):
    max_range = 7.0 + 1.0*(2**num_bits-1)/(2**num_bits)
    min_range = -8
    real = np.clip(real,min_range,max_range)

    if(type(real) is np.ndarray):
        quant = np.empty(real.shape,dtype=np.int16)
        for i in range(real.shape[0]):
            # Recursion
            quant[i] = quantize(real[i],num_bits)
    else:
        scaled = real * 2**num_bits
        quant = int(scaled)
        # if(scaled-quant >= 0.5):
        #     quant += 1
        # if(scaled - quant <= -0.5):
        #     quant -= 1

    quant = np.int16(quant)
    return quant


def de_quantize(quant,num_bits=4):
    real = np.float32(quant)
    real = real * 2.0**(-num_bits)
    return real

def binary_str_to_int(binary_str):
    value = 0
    twos_complement = False
    if(binary_str[0] == '1'):
        twos_complement = True

    for i in range(1,len(binary_str)):
        value *= 2
        if((not twos_complement and binary_str[i] == '1') or 
            (twos_complement and binary_str[i] == '0')):
            value += 1
    
    if(twos_complement):
        value += 1
        value = -value
    return value

def binary_str_to_uint(binary_str):
    value = 0

    for i in range(0,len(binary_str)):
        value *= 2
        if( binary_str[i] == '1'):
            value += 1
    
    return value

if(__name__ == '__main__'):
    print('Quantize 1  : ',str(quantize(1.0)))
    print('De_quant 100: ',str(de_quantize(100)))
    print('De_quant 112: ',str(de_quantize(112)))

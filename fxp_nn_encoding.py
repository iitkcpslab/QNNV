import numpy as np
import os
from relu7_model import Relu7Model
from pysmt.shortcuts import SFXPAdd, SFXPMul, SFXPLT, SFXPGT, SFXPLE, SFXPGE, Equals, SFXP, ST, RD, RU, BV, Equals, Symbol, get_model, is_sat, And, Implies, Or, Not, SFXPDiv
from pysmt.typing import SFXPType
from pysmt.rewritings import get_fp_bv_converter
from pysmt.logics import QF_BV
from quantization_util import quantize, de_quantize
from bitstring import BitArray
import time


class FXPencoding:
    def __init__(self, nn, var_prefix="", verbose=False):
        init_ts = time.time()
        self.nn = nn
        self._verbose = verbose
        self._var_prefix = var_prefix
        self.conjunctions = []
        self.input_symbols = []
        self.all_symbols = []

        self._fractional_digits = 8
        self._quantization_frac_digits = 4
        self._total_digits = 24
        for i in range(self.nn.get_input_size()):
            sym = self.create_symbol("input_{:d}".format(i))
            self.input_symbols.append(sym)

        self.output_symbols = self.encode_network()
        self._fx_build_time = time.time() - init_ts

    def create_symbol(self, name):
        sym = Symbol(self._var_prefix + name,
                     SFXPType(self._total_digits, self._fractional_digits))
        self.all_symbols.append(sym)
        return sym

    def multiply(self, left, right):
        return SFXPMul(ST, RD, left, right)

    def add(self, left, right):
        return SFXPAdd(ST, left, right)

    def twos_complement(self, val, bits):
        b = BitArray("uint:{}={}".format(bits, val))
        if (self._verbose):
            print("B: ", str(b.int))
            print("B: ", str(b.bin))
        b.invert()
        b.prepend("0b0")
        integer = b.int
        if (self._verbose):
            print("B: ", str(b.bin))
            print("B: ", str(b.int))
        return integer + 1

    def const(self, float_value):
        # Quantize the weights by the actual quanization resolution
        quanitzed_value = int(
            quantize(float_value, self._quantization_frac_digits))
        # Fill the additional digits with zeros
        quanitzed_value *= 2**(self._fractional_digits -
                               self._quantization_frac_digits)
        if (self._verbose):
            print("Quantize {} to {}".format(float_value, quanitzed_value))

        if (quanitzed_value < 0):
            quanitzed_value = self.twos_complement(-quanitzed_value,
                                                   self._total_digits)
            if (self._verbose):
                print("q: ", str(quanitzed_value))
                print("two's complement: ", str(quanitzed_value))
        return SFXP(BV(quanitzed_value, self._total_digits),
                    self._fractional_digits)

    def full_const(self, float_value):
        quanitzed_value = int(quantize(float_value, self._fractional_digits))
        if (self._verbose):
            print("Quantize {} to {}".format(float_value, quanitzed_value))

        if (quanitzed_value < 0):
            quanitzed_value = self.twos_complement(-quanitzed_value,
                                                   self._total_digits)
            if (self._verbose):
                print("q: ", str(quanitzed_value))
                print("two's complement: ", str(quanitzed_value))
        return SFXP(BV(quanitzed_value, self._total_digits),
                    self._fractional_digits)

    def assertion(self, predicate):
        if (self._verbose):
            print("asserttion added: ", str(predicate))
        self.conjunctions.append(predicate)

    def recursive_add(self, input_vars, w, layer_no, node_no, balanced=True):
        if (len(input_vars) == 1):
            if (int(quantize(w[0], self._fractional_digits)) == 0):
                return None
            # mult = self.create_symbol("res_{}_{}_{}".format(
            #     layer_no, node_no, input_vars[0].symbol_name()))
            # self.assertion(
            #     Equals(mult, self.multiply(input_vars[0], self.const(w[0]))))
            # return mult
            return self.multiply(input_vars[0], self.const(w[0]))
        if (balanced):
            mid = len(input_vars) // 2
        else:
            mid = 1
        left = self.recursive_add(input_vars[:mid], w[:mid], layer_no, node_no,
                                  balanced)
        right = self.recursive_add(input_vars[mid:], w[mid:], layer_no,
                                   node_no, balanced)
        # sumAll = self.create_symbol("sum_{}_{}_{}".format(
        #     layer_no, node_no, 784 - len(input_vars)))
        if (left is None):
            # self.assertion(Equals(sumAll, right))
            return right
        elif (right is None):
            # self.assertion(Equals(sumAll, left))
            return left
        else:
            # self.assertion(Equals(sumAll, self.add(left, right)))
            return self.add(left, right)

        # return sumAll

    def encode_relu(self, pre_act, post_act):
        clip_ub = 2**(self._quantization_frac_digits - 1) - 1 + (
            (2**(self._quantization_frac_digits) - 1) /
            2**self._quantization_frac_digits)

        clip_lb = -8
        if (self.nn.clip_by_0):
            clip_lb = 0

        # Quantize and convert to constant
        lb = self.const(clip_lb)
        ub = self.const(clip_ub)

        # The activaiton funciton has 3 cases
        case_1 = Implies(SFXPLE(pre_act, lb), Equals(post_act, lb))
        case_2 = Implies(SFXPGE(pre_act, ub), Equals(post_act, ub))
        case_3 = Implies(And(SFXPLT(pre_act, ub), SFXPGT(pre_act, lb)),
                         Equals(post_act, pre_act))

        self.assertion(And(case_1, case_2, case_3))
        # case_1 = Implies(SFXPLE(pre_act, self.const(0)),
        #                  Equals(post_act, self.const(0)))
        # case_2 = Implies(SFXPGT(pre_act, self.const(0)),
        #                  Equals(post_act, pre_act))
        # self.assertion(And(case_1, case_2))

        # Disable activation function
        # self.assertion(Equals(post_act,pre_act))

    def round(self, x):
        shift = SFXP(BV(2**self._quantization_frac_digits, self._total_digits),
                     self._fractional_digits)
        shifted_x = SFXPMul(ST, RD, x, shift)
        shifted_x_var = self.create_symbol("shifted_{}".format(
            x.symbol_name()))
        self.assertion(Equals(shifted_x_var, shifted_x))
        unshifted_x = SFXPDiv(ST, RD, shifted_x, shift)
        unshifted_x_var = self.create_symbol("unshifted_{}".format(
            x.symbol_name()))
        self.assertion(Equals(unshifted_x_var, unshifted_x))
        return unshifted_x

    def encode_neuron(self, input_vars, w, b, layer_no, neuron_no):
        if (self._verbose):
            print("encoding neuron with w {} b {}".format(str(w), str(b)))
        accumulator = self.recursive_add(input_vars, w, layer_no, neuron_no)
        if (accumulator is None):
            # happens if all weights are zero
            accumulator = self.const(0)
        if (b != 0):
            accumulator = self.add(accumulator, self.full_const(b))
        return accumulator

    def encode_layer(self, input_vars, w, b, name, layer_no):
        output_vars = []
        for neuron in range(b.shape[0]):
            pre_act_var = self.create_symbol("pre_{}_{:d}".format(
                name, neuron))
            post_act_var = self.create_symbol("post_{}_{:d}".format(
                name, neuron))
            weight_row = w[:, neuron]
            neuron = self.encode_neuron(input_vars, weight_row, b[neuron],
                                        layer_no, neuron)
            self.assertion(Equals(pre_act_var, neuron))
            rounded_var = self.round(pre_act_var)
            self.encode_relu(rounded_var, post_act_var)

            output_vars.append(post_act_var)

        return output_vars

    def encode_network(self):
        # current_vars = self.input_symbols
        self.clipped_symbols = [
            self.create_symbol("clip_{}".format(x.symbol_name()))
            for x in self.input_symbols
        ]
        current_vars = self.clipped_symbols
        for c, i in zip(current_vars, self.input_symbols):
            self.assertion(Equals(c, self.round(i)))

        for i in range(self.nn.count_layers()):
            w = self.nn.weights[i]
            b = self.nn.biases[i]
            self.nn.clip_by_0 = (i != self.nn.count_layers() - 1)
            current_vars = self.encode_layer(current_vars, w, b,
                                             "layer_{}".format(i), i)

        return current_vars

    def inference_with_sat(self, x):
        conv_start = time.time()
        for i in range(len(self.input_symbols)):
            input_const = self.const(x[i])
            self.assertion(self.input_symbols[i].Equals(input_const))

        conv = get_fp_bv_converter()
        bv_cons = conv.convert(And(self.conjunctions))

        sat_start = time.time()
        model = get_model(bv_cons, solver_name="btor", logic=QF_BV)
        end_time = time.time()
        if model:
            outputs = []
            for i in range(len(self.output_symbols)):
                bv_value = model.get_value(
                    conv.symbol_map[self.output_symbols[i]])
                us_value = int(bv_value.bv_signed_value())
                float_value = de_quantize(us_value,
                                          num_bits=self._fractional_digits)
                outputs.append(float_value)
            outputs = np.array(outputs)
            if (self._verbose):
                print("SAT!")
                print("sat pred: ", str(outputs))
                print("\nall symbols:")
                for s in self.all_symbols:
                    bv_value = model.get_value(conv.symbol_map[s])
                    us_value = int(bv_value.bv_signed_value())
                    float_value = de_quantize(us_value,
                                              num_bits=self._fractional_digits)
                    print("BV value {}".format(bv_value))
                    print("Dequantize {} to {}".format(us_value, float_value))
                    print("{}: {}".format(str(s), float_value))
            print("Encode  time {:0.2f} seconds".format(self._fx_build_time))
            print("Convert time {:0.2f} seconds".format(sat_start -
                                                        conv_start))
            print("SAT     time {:0.2f} seconds".format(end_time - sat_start))
            return outputs
        else:
            raise ValueError("This should never happen!")


def unit_test_model(relu_model, name):
    print("Running {} ... ".format(name), end="")
    env = FXPencoding(relu_model)

    x = np.array([0.5, 0.25])
    float_prediction = relu_model.forward(x)
    fp_prediction = env.inference_with_sat(x)

    passed = np.max(np.abs(float_prediction - fp_prediction)) < 0.001
    if (passed):
        print("[PASS]")
    else:
        print("[FAIL]")
        raise ValueError("Failed test!")


def run_unit_tests():
    print("Running unit test of NN fixed-point encodings")
    test0 = Relu7Model()
    test0.create_debug_mlp_0()
    unit_test_model(test0, "test 0")

    test1 = Relu7Model()
    test1.create_debug_mlp_1()
    unit_test_model(test1, "test 1")

    test2 = Relu7Model()
    test2.create_debug_mlp_2()
    unit_test_model(test2, "test 2")

    test3 = Relu7Model()
    test3.create_debug_mlp_3()
    unit_test_model(test3, "test 3")

    print("All unit tests passed!\n\n")


if __name__ == "__main__":

    run_unit_tests()

    relu_model = Relu7Model()
    relu_model.create_debug_mlp_3()
    env = FXPencoding(relu_model)

    x = np.array([0.5, 0.25])
    print("relu forward: \n" + str(relu_model.forward(x)))
    print("fp forward: \n" + str(env.inference_with_sat(x)))

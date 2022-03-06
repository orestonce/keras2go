"""make_test_suite.py
This file is part of keras2go
Copyright 2020 Rory Conlin
Licensed under MIT License
https://github.com/f0uriest/keras2c

Generates automatic test suite for converted code
"""

# imports
import numpy as np
from keras2go.io_parsing import get_model_io_names
from keras2go.weights2go import Weights2C
import tensorflow as tf
import subprocess
tf.compat.v1.disable_eager_execution()

__author__ = "Rory Conlin"
__copyright__ = "Copyright 2020, Rory Conlin"
__license__ = "MIT"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
__email__ = "wconlin@princeton.edu"


def make_test_suite(model, function_name, package_name, num_tests=10, stateful=False, verbose=True, tol=1e-3):
    """Generates code to test the generated C function.

    Generates random inputs to the model, and gets the corresponding predictions for them.
    Writes input/output pairs to a C file, along with code to call the generated C function
    and compare the true outputs with the outputs from the generated code.

    Writes the test function to a file `<function_name>_test_suite.c`

    Args:
        model (keras Model): model being converted to C
        function_name (str): name of the neural net function being generated
        num_tests (int): number of tests to generate
        stateful (bool): whether the model contains layers that maintain state between calls.
        verbose (bool): whether to print output
        tol (float): tolerance for passing tests. Tests pass if the maximum error over
            all elements between the true output and generated code output is less than tol.

    Returns:
        None
    """

    if verbose:
        print('Writing tests')
    input_shape = []
    # output_shape = []
    model_inputs, model_outputs = get_model_io_names(model)
    num_inputs = len(model_inputs)
    num_outputs = len(model_outputs)
    for i in range(num_inputs):
        temp_input_shape = np.array(model.inputs[i].shape)
        temp_input_shape = np.where(
            temp_input_shape == None, 1, temp_input_shape)
        if stateful:
            temp_input_shape = temp_input_shape[:]
        else:
            temp_input_shape = temp_input_shape[1:]
        input_shape.insert(i, temp_input_shape)
  #  for i in range(num_outputs):
  #      output_shape.insert(i, model.outputs[i].shape[1:])

    file = open(function_name + '_test.go', "x+")
    s = 'package '+package_name+'\n\n'
    s += 'import "github.com/orestonce/keras2go"\n'
    s += 'import "testing"\n'
    s += 'import "time"\n'
    s += 'import "fmt"\n'
    s += 'import "strconv"\n'
    s += 'import "math"\n'
    file.write(s)

    s = 'func TestFn_'+function_name+'(t *testing.T){\n'
    file.write(s)

    s = """maxabs := func(tensor1, tensor2 *keras2go.K2c_tensor) float64{ \n
    var x float64
    for i:=0; i<tensor1.Numel; i++{\n
        y := math.Abs(tensor1.Array[i]-tensor2.Array[i])
        if y>x {
            x=y
        }
    }
    return x
}\n\n"""
    file.write(s)

    for i in range(num_tests):
        if i == num_tests//2 and stateful:
            model.reset_states()
        # generate random input and write to file
        ct = 0
        while True:
            rand_inputs = []
            for j, _ in enumerate(model_inputs):
                rand_input = 4*np.random.random(input_shape[j]) - 2
                if not stateful:
                    rand_input = rand_input[np.newaxis, ...]
                rand_inputs.insert(j, rand_input)
                # make predictions
            outputs = model.predict(rand_inputs)
            if np.isfinite(outputs).all():
                break
            else:
                ct += 1
            if ct > 20:
                raise Exception('Cannot find inputs to the \
                network that result in a finite output')
        for j, _ in enumerate(model_inputs):
            file.write(Weights2C.array2c((rand_inputs[j][0, :]), 'test' + str(i+1) +
                                         '_' + model_inputs[j] + '_input'))

            # write predictions
        if not isinstance(outputs, list):
            outputs = [outputs]
        for j, _ in enumerate(model_outputs):
            output = outputs[j][0, :]
            file.write(Weights2C.array2c(output, 'keras_' +
                                         model_outputs[j] + '_test' + str(i+1)))
            file.write(Weights2C.array2c(np.zeros(output.shape), 'c_' +
                                         model_outputs[j] + '_test' + str(i+1)))
    
    s = 'var errors[' + str(num_tests*num_outputs) + ']float64\n'
    s += 'var num_tests = ' + str(num_tests) + ' \n'
    s += 'var num_outputs = ' + str(num_outputs) + '\n'

    if stateful:
        reset_sig = function_name + '_reset_states()'
        s += reset_sig
    s += 'var t0 = time.Now()\n'
    file.write(s)

    for i in range(num_tests):
        if i == num_tests//2 and stateful:
            file.write(reset_sig)
        s = function_name + '('
        model_in = ['&test' + str(i+1) + '_' + inp +
                    '_input' for inp in model_inputs]
        model_out = ['&c_' + outp + '_test' +
                     str(i+1) for outp in model_outputs]
        s += ','.join(model_in + model_out)
        s += ')\n'
        file.write(s)
    file.write('\n')
    s = 'var t1 = time.Now()\n'
    s += 'fmt.Println("Average time over ' + str(num_tests) + \
        ' tests: ",strconv.FormatFloat(t1.Sub(t0).Seconds(), \'f\', 5, 64), "s"' + ')\n'
    file.write(s)

    for i in range(num_tests):
        for j, _ in enumerate(model_outputs):
            s = 'errors[' + str(i*num_outputs+j) + '] = maxabs(&keras_' + model_outputs[j] + '_test' + \
                str(i+1) + ',&c_' + \
                model_outputs[j] + '_test' + str(i+1) + ')\n'
            file.write(s)
    s = 'var maxerror = errors[0]\n'
    s += 'for i:=1; i< num_tests*num_outputs;i++{ \n'
    s += 'if errors[i] > maxerror { \n'
    s += 'maxerror = errors[i]\n'
    s += '}\n}\n'
    s += 'fmt.Println("Max absolute error for ' + \
        str(num_tests) + ' tests:", maxerror)\n'
    file.write(s)

    s = 'if (maxerror > ' + str(tol) + ') { \n'
    s += 't.Fatal(maxerror)\n'
    s += '}\n'
    s += '\n}\n\n'
    file.write(s)
    file.close()

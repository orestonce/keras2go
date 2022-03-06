"""keras2go_main.py
This file is part of keras2go
Copyright 2020 Rory Conlin
Licensed under MIT License
https://github.com/f0uriest/keras2c

Converts keras model to C code
"""

# imports
from keras2go.layer2c import Layers2C
from keras2go.weights2go import Weights2C
from keras2go.io_parsing import layer_type, get_all_io_names, get_layer_io_names, \
    get_model_io_names, flatten
from keras2go.check_model import check_model
from keras2go.make_test_suite import make_test_suite
import numpy as np
import subprocess
import tensorflow.keras as keras
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


__author__ = "Rory Conlin"
__copyright__ = "Copyright 2020, Rory Conlin"
__license__ = "MIT"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2go"
__email__ = "wconlin@princeton.edu"


def model2c(model, function_name, package_name, verbose=True):
    """Generates C code for model

    Writes main function definition to "function_name.c" and a public header 

    Args:
        model (keras Model): model to convert
        function_name (str): name of C function
        verbose (bool): whether to print info to stdout

    Returns:
        stateful (bool): whether the model must maintain state between calls
    """

    model_inputs, model_outputs = get_model_io_names(model)

    if verbose:
        print('Gathering Weights')
    stack_vars, static_vars = Weights2C(
        model, function_name).write_weights(verbose)
    stateful = len(static_vars) > 0
    layers = Layers2C(model).write_layers(verbose)

    function_signature = 'func ' + function_name + '('
    function_signature += ', '.join(['' +
                                     in_nm + '_input *keras2go.K2c_tensor' for in_nm in model_inputs]) + ', '
    function_signature += ', '.join(['' +
                                     out_nm + '_output *keras2go.K2c_tensor' for out_nm in model_outputs])
    function_signature += ')'

    reset_sig, reset_fun = gen_function_reset(function_name)

    with open(function_name + '.go', 'x+') as source:
        source.write('package '+package_name+'\n\n')
        source.write('import "github.com/orestonce/keras2go"\n')
        source.write('import "math"\n')
        source.write('\n')
        source.write('var _ = math.MaxInt8\n')
        source.write(static_vars + '\n\n')
        source.write(function_signature)
        source.write(' { \n\n')
        source.write(stack_vars)
        source.write(layers)
        source.write('\n } \n\n')
        if stateful:
            source.write(reset_fun)

    return stateful


def gen_function_reset(function_name):
    """Writes a reset function for stateful models

    Reset function is used to clear internal state of the model

    Args:
        function_name (str): name of main function

    Returns:
       signature (str): delcaration of the reset function
       function (str): definition of the reset function
    """

    reset_sig = 'func ' + function_name + '_reset_states()'

    reset_fun = reset_sig
    reset_fun += ' { \n\n'
    reset_fun += 'memset(&' + function_name + \
                 '_states,0,sizeof(' + function_name + '_states)); \n'
    reset_fun += "} \n\n"
    return reset_sig, reset_fun


def k2c(model, function_name, package_name, num_tests=10, verbose=True):
    """Converts keras model to C code and generates test suite

    Args:
        model (keras Model or str): model to convert or path to saved .h5 file
        function_name (str): name of main function
        malloc (bool): whether to allocate variables on the stack or heap
        num_tests (int): how many tests to generate in the test suite
        verbose (bool): whether to print progress

    Raises:
        ValueError: if model is not instance of keras.models.Model 

    Returns:
        None
    """

    function_name = str(function_name)
    filename = function_name + '.c'
    if isinstance(model, str):
        model = keras.models.load_model(model, compile=False)
    elif not isinstance(model, keras.models.Model):

        raise ValueError('Unknown model type. Model should ' +
                         'either be an instance of keras.models.Model, ' +
                         'or a filepath to a saved .h5 model')

    # check that the model can be converted
    check_model(model, function_name)
    if verbose:
        print('All checks passed')

    stateful = model2c(
        model, function_name, package_name, verbose)

    s = 'Done \n'
    s += "Go code is in '" + function_name + ".go'\n"
    if num_tests > 0:
        make_test_suite(model, function_name, package_name,
                        num_tests, stateful, verbose)
        s += "Tests are in '" + function_name + "_test.go' \n"
    if verbose:
        print(s)

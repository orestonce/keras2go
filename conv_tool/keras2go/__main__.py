"""__main__.py
This file is part of keras2go
Copyright 2020 Rory Conlin
Licensed under MIT License
https://github.com/f0uriest/keras2c

Runs keras2go
"""
import argparse
import sys
from keras2go.keras2go_main import k2c


__author__ = "Rory Conlin"
__copyright__ = "Copyright 2020, Rory Conlin"
__license__ = "MIT"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
__email__ = "wconlin@princeton.edu"


def parse_args(args):
    """Parses command line arguments
    """

    parser = argparse.ArgumentParser(prog='keras2go',
                                     description="""A library for converting the forward pass (inference) part of a keras model to a C function""")
    parser.add_argument("-m", "--model_path", type=str,
        help="File path to saved keras .h5 model file")
    parser.add_argument("-f", "--function_name", type=str,
        help="What to name the resulting Go function")
    parser.add_argument("-p", "--package_name", help="What to name the resulting Go package")
    parser.add_argument("-t", "--num_tests", type=int,
                        help="""Number of tests to generate. Default is 10""")

    return parser.parse_args(args)


def main(args=sys.argv[1:]):

    args = parse_args(args)
    if args.num_tests:
        num_tests = args.num_tests
    else:
        num_tests = 10

    k2c(args.model_path, args.function_name, args.package_name, num_tests)


if __name__ == '__main__':
    main(sys.argv[1:])

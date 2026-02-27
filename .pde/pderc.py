#!/usr/bin/env python3


def parameters_and_defaults():
    param_dict = {
            "infile" : "pde.in.xml", # String
            "outfile" : "pde.out" # String
            }
    return param_dict


if __name__ == '__main__':
    print(parameters_and_defaults())

#!/usr/bin/env python3

import argparse as ap
import os
import sys
from datetime import datetime

from lxml import etree
from dataclasses import dataclass
import numpy as np
import h5py as h5
import math as m
import random

# Ensure companion modules in the same directory are importable both when
# running directly from src/scripts/ and when installed to $PDE_DIR/bin/.
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)


# Define the main class that holds script data structures and settings.
class ScriptSettings():
    """The instance variables of this object are the user settings that
       control the program. The variable values are pulled from a list
       that is created within a resource control file and that are then
       reconciled with command line parameters."""


    def __init__(self):
        """Define default values for the graph parameters by pulling them
        from the resource control file in the default location:
        $PDE_RC/pderc.py or from the current working directory if a local
        copy of pderc.py is present."""

        # Read default variables from the resource control file.
        sys.path.insert(1, os.getenv('PDE_RC'))
        from pderc import parameters_and_defaults
        default_rc = parameters_and_defaults()

        # Assign values to the settings from the rc defaults file.
        self.assign_rc_defaults(default_rc)

        # Parse the command line.
        args = self.parse_command_line()

        # Reconcile the command line arguments with the rc file.
        self.reconcile(args)

        # At this point, the command line parameters are set and accepted.
        #   When this initialization subroutine returns the script will
        #   start running. So, we use this as a good spot to record the
        #   command line parameters that were used.
        self.recordCLP()


    def assign_rc_defaults(self, default_rc):

        # Default filename variables.
        self.infile = default_rc["infile"]
        self.outfile = default_rc["outfile"]


    def parse_command_line(self):
    
        # Create the parser tool.
        prog_name = "pde"

        description_text = """
Version 0.1
The purpose of this program is to allow definition of the phase energy curves
and visualize the phase diagrams that they produce.
"""

        epilog_text = """
Please contact Paul Rulis (rulisp@umkc.edu) regarding questions.
Defaults are given in ./pderc.py or $PDE_RC/pderc.py.
"""

        parser = ap.ArgumentParser(prog = prog_name,
                formatter_class=ap.RawDescriptionHelpFormatter,
                description = description_text,
                epilog = epilog_text)
    
        # Add arguments to the parser.
        self.add_parser_arguments(parser)

        # Parse the arguments and return the results.
        return parser.parse_args()


    def add_parser_arguments(self, parser):
    
        # Define the input file.
        parser.add_argument('-i', '--infile', dest='infile', type=ascii,
                            default=self.infile, help='Input file name. ' +
                            f'Default: {self.infile}')
    
        # Define the output file prefix.
        parser.add_argument('-o', '--outfile', dest='outfile', type=ascii,
                            default=self.outfile, help='Output file name ' +
                            f'prefix for hdf5 and xdmf. Default: {self.outfile}')


    def reconcile(self, args):
        self.infile = args.infile.strip("'")
        self.outfile = args.outfile.strip("'")


    def recordCLP(self):
        with open("command", "a") as cmd:
            now = datetime.now()
            formatted_dt = now.strftime("%b. %d, %Y: %H:%M:%S")
            cmd.write(f"Date: {formatted_dt}\n")
            cmd.write(f"Cmnd:")
            for argument in sys.argv:
                cmd.write(f" {argument}")
            cmd.write("\n\n")


    def read_input_file(self):
        from pde_input import parse_system
        self.system = parse_system(self.infile)


def start_program(settings):
    # Launch the interactive visualization UI.
    from pde_viz import launch_ui
    launch_ui(settings.system)


def main():

    # Get script settings from a combination of the resource control file
    #   and parameters given by the user on the command line.
    settings = ScriptSettings()

    infile = settings.infile
    if os.path.isfile(infile):
        # Normal path: input file present → parse and visualize.
        settings.read_input_file()
        start_program(settings)
    else:
        if infile != 'pde.in.xml':
            # User explicitly specified -i <file> but it doesn't exist → error.
            print(f'Error: input file not found: {infile}', file=sys.stderr)
            sys.exit(1)
        # Default input file absent → open empty UI with builder.
        from pde_viz import launch_ui_empty
        launch_ui_empty()

    # Finalize the program activities and quit.


if __name__ == '__main__':
    # Everything before this point was a subroutine definition or a request
    #   to import information from external modules. Only now do we actually
    #   start running the program. The purpose of this is to allow another
    #   python program to import *this* script and call its functions
    #   internally.
    main()

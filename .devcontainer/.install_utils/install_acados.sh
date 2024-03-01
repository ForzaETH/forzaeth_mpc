#!/bin/bash

# Define the acados installation path
ACADOS_ROOT=~/catkin_ws/src/race_stack/f110_utils/libs/acados_pbl/acados

# Clone acados into the target folder
git clone https://github.com/acados/acados.git $ACADOS_ROOT
cd $ACADOS_ROOT
git submodule update --init --recursive

# Build and install acados
mkdir -p build
cd build
cmake -DACADOS_WITH_QPOASES=ON .. # Add more options as needed
make install -j4

# Update shared library path
make shared_library
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ACADOS_ROOT/lib

# Compile and run examples (optional)
make examples_c
make run_examples_c

# Install Python interface
pip install -e $ACADOS_ROOT/interfaces/acados_template

# Add environment variables to .bashrc and/or .zshrc
echo "export ACADOS_SOURCE_DIR=$ACADOS_ROOT" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$ACADOS_ROOT/lib" >> ~/.bashrc


echo "Installation completed."
echo "Run 'source ~/.bashrc' or 'source ~/.zshrc' depending on your shell."
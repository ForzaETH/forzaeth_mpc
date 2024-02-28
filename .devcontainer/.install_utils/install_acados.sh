#!/bin/bash

# Define acados root directory
acados_root=$(pwd)/src/race_stack/f110_utils/libs/acados_pbl

# Create bin directory inside acados_root
mkdir -p $acados_root/bin

# Ensure the acados_root directory exists
if [ ! -d "$acados_root" ]; then
  echo "The acados root directory does not exist: $acados_root"
  exit 1
fi

# Navigate into acados directory
cd $acados_root

# Create and navigate into build directory
mkdir -p build
cd build

# Configure the build with cmake, setting the install prefix to the acados_root directory
cmake -DACADOS_WITH_QPOASES=ON -DACADOS_WITH_OSQP=OFF/ON -DCMAKE_INSTALL_PREFIX=$acados_root/install ..

# Build and install acados
make install -j4

# Update LD_LIBRARY_PATH environment variable to include acados install lib and qpOASES lib directory
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$acados_root/install/lib:$acados_root/lib

# Install Python interfaces
if [ -d "$acados_root/interfaces/acados_template" ]; then
  pip install -e $acados_root/interfaces/acados_template
else
  echo "The specified Python interface directory does not exist: $acados_root/interfaces/acados_template"
fi

# Download and set up t_renderer
t_renderer_url="https://github.com/acados/tera_renderer/releases/download/v0.0.34/t_renderer-v0.0.34-linux"
t_renderer_path="$acados_root/bin/t_renderer"
wget -O "$t_renderer_path" "$t_renderer_url"
chmod +x "$t_renderer_path"

# Inform the user about t_renderer setup
echo "t_renderer downloaded and set up in $t_renderer_path"

# Determine the owner of the current directory
dir_owner=$(stat -c '%U' .)

# Get the home directory of the owner
owner_home=$(getent passwd "$dir_owner" | cut -d: -f6)

# Append environment variable settings to the .bashrc of the directory owner
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$acados_root/install/lib:$acados_root/lib" >> "$owner_home/.bashrc"
echo "export ACADOS_SOURCE_DIR=$acados_root" >> "$owner_home/.bashrc"

# Inform the user
echo "Appended environment variables to $dir_owner's .bashrc"

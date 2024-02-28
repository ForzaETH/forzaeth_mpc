# MPC
This is how to visualize the MPC with plot_juggler

## Acados Installation

```
#clone acados
git clone https://github.com/acados/acados.git
cd acados
git submodule update --recursive --init

#Build and install
mkdir -p build
cd build
cmake -DACADOS_WITH_QPOASES=ON ..
# add more optional arguments e.g. -DACADOS_WITH_OSQP=OFF/ON -DACADOS_INSTALL_DIR=<path_to_acados_installation_folder> above
make install -j4

#make
make shared_library
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path_to_acados_folder>/lib
make examples_c
make run_examples_c


#python interfaces
pip install -e <acados_root>/interfaces/acados_template

sudo gedit ~/.bashrc

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"<acados_root>/lib"
export ACADOS_SOURCE_DIR="<acados_root>/acados"

source ~/.bashrc
```



## Visualization-Test

Install plotjuggler:

```
sudo snap install plotjuggler-ros
```

Start-test:

```
plotjuggler-ros
```



## MPC Visualization

Step 1: start the simulator

```
# define the parameter of the simulator:stack_master/config/NUC5/NUC5_pacejka.yaml
roslaunch stack_master base_system.launch map_name:=f racecar_version:=SIM sim:=true tire_model:=pacejka
```

step 2: start the mpc controller

```
roslaunch controller mpc_controller.launch
```

step 3: open plotjuggler 

```
plotjuggler-ros
# load xml file to visualization from:
./config/mpc_plot.xml
```

step 3: follow the command

## Sector-MPC

If you already selected sector, just skip step 1.

step 1 start simulator:

```
roslaunch stack_master mapping.launch map_name:=JFR_racingv6 racecar_version:=NUC5 sim:=true
```

step 2 select sector:

```
roslaunch stack_master base_system.launch map_name:=JFR_racingv6 sim:=True racecar_version:=NUC5 tire_model:=pacejka
```

step 3 start controller:

```
roslaunch controller sec_mpc_controller.launch
```

step 4 open rqt to turn the parameters online

```
rqt
```


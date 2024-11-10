#!/usr/bin/env python3
import os
import rospkg
import numpy as np
import rospy
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from MAP2.Sparse_GP_function import sparse_gp_model_train, sparse_gp_model_predict, sparse_gp_plot
from MAP2.Sparse_GP_data_selection import data_selection, filter_by_range
from datetime import datetime
import warnings
from linear_operator.utils.warnings import NumericalWarning
class GPModel:
    def __init__(self):
        """Initialize the GP model and related parameters."""
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('gp_mpc_node')
        self.model_file = os.path.join(package_path, '/save_model/sparse_gp_model.pth')
        # onliner turner parameters
        self.GP_model_load = False
        self.GP_model_update = True
        self.GP_model_save = False
        self.y_min = -0.1
        self.y_max = 0.1

        self.model = None
        self.likelihood = None
        self.sampling_rate = 40
        self.Initial_data_size = 800 #TODO these parameters initial values needed to be obtained from the yaml file
        self.Max_data_size = 1200
        self.data_duration = self.Initial_data_size / self.sampling_rate
        self.timesteps = self.data_duration * self.sampling_rate
        self.pbar = None
        self.X_train = np.array([])
        self.y_train = np.array([])
        self.new_X = np.array([])
        self.new_Y = np.array([])
        self.inducing_points_number = 50
        self.sgp_training_iterations = 30
        self.center_car_s = None # get from the controller
        self.beta_error = None # get from the controller
        self.new_data_available = False # get from the controller
        self.lap_counter = 0
        self.counter = 0
        self.Initial_gp = False
        self.update_gp = False
        self.lap_flag = False
        self.gp_ready_preidct = False
        self.GP_predict_viz = False
        self.GP_predict_viz_Initial = False
        self.GP_predict_viz_update = False
        #save model
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        # clean the controller command terminal
        warnings.filterwarnings("ignore", category=UserWarning, module="gpytorch")
        warnings.filterwarnings("ignore", category=NumericalWarning)

        #GPpytorch warmup
        self.gpytorch_warmup()
        #rosinfo
        rospy.loginfo("The data collection for GP will be started after one lap.")

    def gpytorch_warmup(self):
        warmup_train_x = np.linspace(0, 1, 100).reshape(-1, 1)
        warmup_train_y = np.sin(12 * warmup_train_x) + 0.66 * np.cos(25 * warmup_train_x)
        warmup_train_y = (warmup_train_y - warmup_train_y.min()) / (warmup_train_y.max() - warmup_train_y.min())
        _, _ =sparse_gp_model_train(warmup_train_x, warmup_train_y, self.inducing_points_number, self.sgp_training_iterations, predic_warmup=True, train_viz_disable=True)
        rospy.loginfo("GPpytorch warmup is completed.")

    def gp_callback(self, event):
        """Callback for GP model computation."""

        self.loop_logistic()

        # Create the initial GP model
        if self.Initial_gp and not self.update_gp:
            self.Initial_GP_model_train()
            self.gp_ready_preidct = True
            # save the model
            self.save_model()

        # Update the GP model with new data
        if self.lap_flag and self.update_gp:
            if self.GP_model_update:
                self.gp_ready_preidct = False
                self.online_GP_model_update()
                self.gp_ready_preidct = True
                # save the model
                self.save_model()
            else:
                print("The GP model is not updated.")

            # Visualize the GP model predictions (only viz when GP_predict_viz is True)
            plt.close('all')
            self.GP_predict_viz_update = True




    def loop_logistic(self):
        """Manage the loop and data collection logistics for the GP model update."""
        if self.lap_counter > 1 and not self.update_gp:
            self.Initial_gp = True

        # Collect data to update the GP model
        self.collect_lap_data()

    def collect_lap_data(self):
        """Collect data points for GP model training."""
        self.new_X = np.append(self.new_X, self.center_car_s)
        self.new_Y = np.append(self.new_Y, self.beta_error)

        if self.lap_flag:
            # Store the current lap data
            self.new_x = self.new_X.copy()  # Use .copy() to prevent reference issues
            self.new_y = self.new_Y.copy()

            # Reset the data for the next lap
            self.new_X = np.array([])
            self.new_Y = np.array([])

    def collect_Initial_GP_data(self):
        """Collects initial training data for the GP model."""
        if self.pbar is None:
            self.pbar = tqdm(total=self.timesteps, desc='Collecting data', ascii=True)
        if self.new_data_available:
            if self.center_car_s is not None and self.beta_error is not None:
                self.X_train = np.append(self.X_train, self.center_car_s)
                self.y_train = np.append(self.y_train, self.beta_error)
                self.counter += 1
                self.pbar.update(1)
                if self.counter == self.timesteps + 1:
                    self.pbar.close()
                    rospy.loginfo("Initial Data collection for GP is completed.")
                self.new_data_available = False

    def Initial_GP_model_train(self):
        """Train the initial Gaussian Process (GP) model."""
        self.collect_Initial_GP_data()

        if self.counter == self.timesteps + 1:
            start_time = rospy.get_time()
            self.X_train = self.X_train.reshape(-1, 1)
            self.y_train = self.y_train.reshape(-1, 1)
            self.X_train, self.y_train, _ = filter_by_range(self.X_train, self.y_train, self.y_min, self.y_max)
            self.model, self.likelihood = sparse_gp_model_train(self.X_train, self.y_train, self.inducing_points_number,
                                                                self.sgp_training_iterations, predic_warmup=True, train_viz_disable=True)
            end_time = rospy.get_time() - start_time
            print(f"GP model Initial training time: {end_time}")
            print("The GP model is initialized.")
            self.Initial_gp = False
            self.update_gp = True
            self.GP_predict_viz_Initial = True

    def online_GP_model_update(self):
        """Update the Gaussian Process (GP) model with new training data."""
        print("Data selection is implementing. The size of new data: ", len(self.new_x))
        start_time = rospy.get_time()
        self.X_train = self.X_train.reshape(-1, 1)
        self.y_train = self.y_train.reshape(-1, 1)
        self.X_train, self.y_train = data_selection(self.model, self.likelihood, self.X_train, self.y_train,
                                                    self.new_x, self.new_y, self.Max_data_size, self.y_min, self.y_max,
                                                    selection_mode="inducing_cluster")
        print("Data selection is completed. The current data size: ", len(self.X_train))
        self.model, self.likelihood = sparse_gp_model_train(self.X_train, self.y_train, self.inducing_points_number,
                                                            self.sgp_training_iterations, predic_warmup=True, train_viz_disable=True)
        print("The time for updating the GP model: ", rospy.get_time() - start_time)


    def gp_model_viz(self):
        """Visualize the GP model predictions."""
        if self.GP_predict_viz_Initial or (self.GP_predict_viz_update and self.GP_predict_viz):
            if self.model is not None:
                xtest = np.linspace(min(self.X_train), max(self.X_train), len(self.X_train)).reshape(-1, 1)
                gp_mean, gp_var, gp_inducing = sparse_gp_model_predict(self.model, self.likelihood, xtest)
                plt.close('all')
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                sparse_gp_plot(self.X_train, self.y_train, xtest, gp_mean, gp_var, gp_inducing, ax)
                plt.tight_layout()
                plt.pause(0.01)  # Pause to update the plot without blocking the code
                self.GP_predict_viz_Initial = False
                self.GP_predict_viz_update = False

    def save_model(self):
        # save the model when the GP model created or updated
        if self.model is not None:
            if self.GP_model_save == True:
                    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.historical_model_file = os.path.join(
                    self.script_dir,
                    f'history_model/lap_{self.lap_counter}_gp_model_{current_time_str}.pth')
                    torch.save({'model': self.model,'likelihood': self.likelihood}, self.historical_model_file)






# %%
import time
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import random
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import math
from sklearn.metrics.pairwise import haversine_distances
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error, max_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import RandomizedSearchCV
from gurobi_ml import add_predictor_constr
warnings.filterwarnings('ignore', category=ConvergenceWarning)
pd.options.mode.chained_assignment = None  # Disables SettingWithCopyWarning
import os
from pathlib import Path


# %%
# %%
class synthetic_experiments:
    def __init__(self, seed, year_range, numCustomers, numStations, start_date_str, end_date_str):
        self.year_range = year_range
        self.numCustomers = numCustomers
        self.numStations = numStations
        self.trainSizeData = numStations + 1

        # Convert string dates to datetime objects and store them as instance variables
        self.start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

        self.seed = seed
        np.random.seed(self.seed)
        self.noise_sd = 3

        # coefficients for decision-dependent regression model
        self.dep_alpha1 = 4
        self.dep_beta_distance = -0.05
        self.dep_beta_education = 300
        self.dep_beta_gdp = 3000
        self.dep_intercept = 1500

        self.sample_size = 300

        self.seed_value = 16

        self.test_sizeData = 5000

        self.num_exp_Facilities = 50
        self.num_exp_Customers = 5

    def assign_random_dates(self, seed):
        # Calculate the difference between the start and end date
        delta = self.end_date - self.start_date

        # Assign random dates to each EV charging station
        random.seed(seed)
        random_dates = [
            self.start_date + timedelta(days=random.randint(0, delta.days))
            for _ in range(self.numStations)
        ]

        # Sort the random dates in ascending order
        sorted_dates = sorted(random_dates)
        sorted_dates_df = pd.DataFrame(sorted_dates).reset_index()

        # Modify the DataFrame in place and assign it directly
        sorted_dates_df.rename(columns={'index': 'station_index', 0: 'Open Date'}, inplace=True)
        self.sorted_dates_df = sorted_dates_df

        return self.sorted_dates_df

    def generate_random_point(self, center, radius, seed):
        # Convert radius from miles to degrees (approximately)
        radius_degrees = radius / 69.0

        random.seed(seed)

        # Generate random angle (0 to 360 degrees)
        random_angle = random.uniform(0, 2 * math.pi)

        # Generate random distance (0 to radius_degrees)
        random_distance = random.uniform(0, radius_degrees)

        # Calculate the new latitude and longitude
        new_latitude = center[0] + (random_distance * math.cos(random_angle))
        new_longitude = center[1] + (random_distance * math.sin(random_angle))

        return new_latitude, new_longitude

    def data_generation(self, seed):
        np.random.seed(seed)
        # Generate distance data
        central_latitude = 40
        central_longitude = -73
        radius_miles = 45

        # Generate random points within the circle for selected_coordinates_df
        random_coordinates_selected = []
        for seed in range(1, self.numCustomers + 1):  # Use a different range of seeds
            random_point = self.generate_random_point((central_latitude, central_longitude), radius_miles, seed)
            random_coordinates_selected.append(random_point)

        selected_coordinates_df = pd.DataFrame(random_coordinates_selected,
                                               columns=['Latitude', 'Longitude']).reset_index(drop=True)
        self.customers_locations = selected_coordinates_df

        # Generate random points within the circle for random_points_df
        random_coordinates_random = []
        for seed in range(self.numCustomers + 1,
                          self.numCustomers + self.numStations + 1):  # Use a different range of seeds
            random_point = self.generate_random_point((central_latitude, central_longitude), radius_miles, seed)
            random_coordinates_random.append(random_point)

        random_points_df = pd.DataFrame(random_coordinates_random, columns=['Latitude', 'Longitude']).reset_index(
            drop=True)
        self.stations_locations = random_points_df

        c1 = np.radians(selected_coordinates_df[['Latitude', 'Longitude']].to_numpy())
        c2 = np.radians(random_points_df[['Latitude', 'Longitude']].to_numpy())
        dist = haversine_distances(c2, c1) * 6371 * 0.621371  # Converting distances to miles

        experiments_distance = pd.DataFrame(dist, index=range(self.numStations),
                                            columns=range(self.numCustomers)).reset_index()
        experiments_distance.rename(columns={'index': 'station_index'}, inplace=True)
        experiments_distance_df = experiments_distance.melt(id_vars=['station_index'], var_name='customer_index',
                                                            value_name='distance')
        self.distance_df = experiments_distance_df

        filterd_dist_df = experiments_distance_df[experiments_distance_df['distance'] <= 30]
        merged_dist_date_df = filterd_dist_df.merge(self.sorted_dates_df, on=['station_index'], how='inner')
        merged_dist_date_df = merged_dist_date_df.sort_values(by=['customer_index', 'Open Date']).reset_index(drop=True)
        merged_dist_date_df['Year'] = merged_dist_date_df['Open Date'].dt.year

        gdp_values = {}
        # base_gdp = np.random.uniform(-0.1, 0.1, (self.numCustomers, 1))  # Baseline variation
        base_gdp = np.random.uniform(6000, 14000, (self.numCustomers, 1))  # Baseline variation （1000, 5000)
        error_term_gdp = np.random.normal(0, 12000, size=(self.numCustomers, len(self.year_range)))  # Larger noise
        for i, year in enumerate(self.year_range):
            # predicted_gdp = 0.15 * year - 302.4  # Example trend line
            predicted_gdp = 16000 * (year - 2016)  # Example trend line
            # predicted_gdp = 16000 * (year - 2016)
            customer_gdp_for_year = predicted_gdp + base_gdp.flatten() + error_term_gdp[:, i]

            # Clip values to ensure they're within range, and scale to [0, 1]
            # customer_gdp_for_year = np.clip(customer_gdp_for_year, 1000, 5000)

            gdp_values[year] = customer_gdp_for_year

        # gdp_df = pd.DataFrame(gdp_values).reset_index()
        # gdp_df.rename(columns={'index': 'customer_index'}, inplace=True)
        # gdp_melted = gdp_df.melt(id_vars=['customer_index'], var_name='Year', value_name='GDP')
        # self.gdp_df = gdp_melted

        gdp_df = pd.DataFrame(gdp_values).reset_index()
        gdp_df.rename(columns={'index': 'customer_index'}, inplace=True)
        gdp_melted = gdp_df.melt(id_vars=['customer_index'], var_name='Year', value_name='GDP')
        self.gdp_df = gdp_melted

        # Generate education data
        education_values = {}
        base_education = np.random.uniform(4, 16, (self.numCustomers, 1))  # (10, 50)
        error_term_education = np.random.normal(0, 0.2, size=(self.numCustomers, len(self.year_range)))
        for i, year in enumerate(self.year_range):
            # predicted_education = 0.1 * year - 201.4
            # predicted_education = 4 * year - 8000
            predicted_education = 4 * (year - 2016)
            customer_education_for_year = predicted_education + base_education.flatten() + error_term_education[:, i]
            education_values[year] = customer_education_for_year

        education_df = pd.DataFrame(education_values).reset_index()
        education_df.rename(columns={'index': 'customer_index'}, inplace=True)
        education_melted = education_df.melt(id_vars=['customer_index'], var_name='Year', value_name='Education')
        self.education_df = education_melted

        # Generate capacity and initial demand data
        capacity_data = pd.DataFrame(np.random.randint(1, 11, (self.numStations, 1))).reset_index()
        capacity_data.rename(columns={'index': 'station_index', 0: 'capacity'}, inplace=True)
        initial_data = pd.DataFrame(np.random.randint(10000, 180000, (self.numCustomers, 1))).reset_index()
        initial_data.rename(columns={'index': 'customer_index', 0: 'initial_demand'}, inplace=True)
        self.capacity_data = capacity_data
        self.initial_data = initial_data

        # Merge data
        merged_data = merged_dist_date_df.merge(capacity_data, on=['station_index'], how='left')
        merged_data = merged_data.merge(initial_data, on=['customer_index'], how='left')
        merged_data = merged_data.merge(gdp_melted, on=['customer_index', 'Year'], how='left')
        merged_data = merged_data.merge(education_melted, on=['customer_index', 'Year'], how='left')

        columns_to_normalize = ['Education', 'GDP']
        # Create a StandardScaler object
        scaler = MinMaxScaler(feature_range=(0, 1))
        merged_data[['normalized_education_level', 'normalized_gdp']] = scaler.fit_transform(
            merged_data[columns_to_normalize])
        self.orig_data = merged_data

        # Initialize variables
        z_county = {}
        z_dist = {}
        z_capacity = {}

        regression_data_list = []

        for customer in merged_data['customer_index'].unique():
            data = merged_data[merged_data['customer_index'] == customer]
            z_stations = len(data)
            z_rows = z_stations + 1

            # Generate the lower triangular matrix for z_county
            z_county[customer] = np.tril(np.ones((z_rows, z_stations)), -1)
            z_dist[customer] = data['distance'].values
            z_capacity[customer] = data['capacity'].values

            # Prepare matrices
            z_matrix = z_county[customer]
            z_dist_temp = z_dist[customer]
            z_capacity_temp = z_capacity[customer]

            gdp = data['normalized_gdp'].values
            education = data['normalized_education_level'].values
            gdp_2017 = gdp[0]
            education_2017 = education[0]

            # Add noise
            np.random.seed(customer)
            # noise = np.random.normal(0, 50, (z_rows,))
            noise = np.random.normal(0, self.noise_sd, (z_rows,))

            # Capacity matrix
            capacity_matrix = z_matrix @ np.diag(z_capacity_temp)

            # Exponential demand term
            exp_term = np.exp(self.dep_alpha1 + self.dep_beta_distance * z_dist_temp)

            # Calculate exp_demand
            exp_demand = (capacity_matrix @ exp_term.reshape(-1, 1)).flatten()
            # Add 2017 row
            new_row = pd.DataFrame({
                'customer_index': [customer],
                'station_index': [None],
                'distance': [0],
                'Open Date': [self.start_date],
                'Year': [2017],
                'normalized_gdp': [gdp_2017],
                'normalized_education_level': [education_2017],
                'capacity': [0],
                'initial_demand': [data['initial_demand'].values[0]]
            })
            data = pd.concat([new_row, data]).reset_index(drop=True)

            # Demand calculation
            init_demand = (
                    data['initial_demand'] +
                    data['normalized_education_level'] * self.dep_beta_education +
                    data['normalized_gdp'] * self.dep_beta_gdp +
                    self.dep_intercept +
                    noise
            )
            #  # print(init_demand)

            # Final demand
            demand = init_demand + exp_demand
            data['demand'] = demand
            #  # print(demand)

            # Add to regression list
            regression_data_list.append(data)

            # Combine all customer data into one DataFrame
        self.regression_data = pd.concat(regression_data_list, ignore_index=True)

    def step1_regression_data(self, data):
        regression_data = data
        grouped_data = regression_data.groupby(['customer_index'])

        # Initialize lists to store the processed data
        processed_data_list = []
        step1_data = []

        # Iterate over each group (customer_index and year)
        for customer, group in grouped_data:
            # Work on a copy of the group to avoid modifying the original data
            group_copy = group.copy()
             # print(group_copy)

            group_year = group_copy.groupby('Year')

            for year, year_group in group_year:
                year_group['increase_demand'] = year_group['demand'].diff()
                year_group = year_group.dropna()
                step1_data.append(year_group)

            # # Compute the difference for the 'demand' column within each group
            # group_copy['increase_demand'] = group_copy['demand'].diff()
            #
            # # Append the group with the 'increase_demand' column to step1_data
            # step1_data.append(group_copy)

            # Drop rows with NaN (the first row after .diff()) and store cleaned data
            group_cleaned = group_copy.dropna()

             # print(group_cleaned)

            # Append cleaned data to processed_data_list
            processed_data_list.append(group_cleaned)

        # Combine all the processed groups back into DataFrames
        processed_data = pd.concat(processed_data_list, ignore_index=True)
        processed_data = processed_data.drop_duplicates(subset=['Open Date', 'customer_index'], keep=False)
        reg_data = processed_data
        step1_data = pd.concat(step1_data, ignore_index=True)
        self.remove_step1_data = step1_data[step1_data['increase_demand'] <= 0]
        step1_data = step1_data[step1_data['increase_demand'] > 0]
        step1_data['exp(Y)'] = step1_data['increase_demand'] / step1_data['capacity']

        return reg_data, step1_data

    def step1_regression(self, data):
        reg_data = data
        X = reg_data['distance'].values.reshape(-1, 1)
        Y = np.log(reg_data['exp(Y)'])

        # Fit a linear regression model
        model = LinearRegression()
        model.fit(X, Y)

        # Predict the y values
        y_pred = model.predict(X)

        # Coefficients
        self.step1_intercept = model.intercept_
        self.step1_coef = model.coef_

        # Compute R2 score
        self.step1_r2 = r2_score(Y, y_pred)
        self.step1_mae = median_absolute_error(Y, y_pred)
        self.step1_me = max_error(Y, y_pred)

        # Compute mean squared error (MSE)
        self.step1_mse = mean_squared_error(Y, y_pred)
        self.step1_residuals = Y - y_pred

        # Create a DataFrame to return the results
        results = pd.DataFrame({
            'Data Size': [len(data)],
            'Coefficient': [self.step1_coef],
            'Intercept': [self.step1_intercept],
            'MSE': [self.step1_mse],
            'R^2': [self.step1_r2],
            'MAE': [self.step1_mae],
            'ME': [self.step1_me]
        })

        self.step1_results_df = results

    # def step2_regression(self, data):
    #     grouped_data = data.groupby(['customer_index', 'seed'])
    #
    #     z_county = {}
    #     z_dist = {}
    #     z_capacity = {}
    #
    #     step2_data_list = []
    #
    #     for (customer, seed), group in grouped_data:
    #         # Filter data for the current customer
    #         data = group
    #
    #         # Number of stations and rows for the current customer
    #         z_rows = len(data)  # z_rows is stations plus 1
    #         z_stations = z_rows - 1
    #         z_dist[seed] = {}
    #         z_capacity[seed] = {}
    #         z_county[seed] = {}
    #
    #         # Generate the lower triangular matrix for z_county
    #         z_county[seed][customer] = np.tril(np.ones((z_rows, z_stations)), -1)
    #
    #         # Distance matrix for z_dist (just the distance column for this customer)
    #         z_dist[seed][customer] = data['distance'][1:].values
    #         z_capacity[seed][customer] = data['capacity'][1:].values
    #
    #         z_matrix = z_county[seed][customer]
    #         z_dist_temp = z_dist[seed][customer]
    #         z_capacity_temp = z_capacity[seed][customer]
    #
    #         #  # print(z_matrix, z_dist_temp, z_time_temp, z_capacity_temp)
    #
    #         capacity_matrix = z_matrix @ np.diag(z_capacity_temp)
    #         #  # print(capacity_matrix)
    #         exp_term = np.exp(self.step1_intercept + self.step1_coef * z_dist_temp)
    #         #  # print(exp_term)
    #
    #         exp_demand = capacity_matrix @ exp_term.reshape(-1, 1)
    #         exp_demand = exp_demand.flatten()
    #         #  # print(exp_demand)
    #
    #         # Create a DataFrame for the demand and add it as a new column
    #         demand_df = pd.DataFrame(exp_demand, columns=['ested_exp_demand'])
    #         #  # print(demand_df)
    #         data_copy = data.copy()
    #         data_copy['ested_exp_demand'] = demand_df['ested_exp_demand'].values
    #         #  # print(data_copy)
    #
    #         # add H1, H2, H3, H4 columns
    #         # H1: exp_term = 1 (sum_capacity)
    #         exp_term_h1 = np.ones_like(z_dist_temp)
    #         exp_demand_h1 = capacity_matrix @ exp_term_h1.reshape(-1, 1)
    #
    #         # H2: exp_term = exp(-distance)
    #         exp_term_h2 = np.exp(-z_dist_temp)
    #         exp_demand_h2 = capacity_matrix @ exp_term_h2.reshape(-1, 1)
    #
    #         # H3: exp_term = exp(-distance/25)
    #         exp_term_h3 = np.exp(-z_dist_temp / 25)
    #         exp_demand_h3 = capacity_matrix @ exp_term_h3.reshape(-1, 1)
    #
    #         # H4: exp_term = exp(step1_coef * distance + step1_intercept)
    #         exp_term_h4 = np.exp(self.step1_intercept + self.step1_coef * z_dist_temp)
    #         exp_demand_h4 = capacity_matrix @ exp_term_h4.reshape(-1, 1)
    #
    #         data_copy['sum_capacity'] = exp_demand_h1.flatten()
    #         data_copy['exp_distance_capacity'] = exp_demand_h2.flatten()
    #         data_copy['assumption_exp_distance_capacity'] = exp_demand_h3.flatten()
    #         data_copy['step1_exp_distance_capacity'] = exp_demand_h4.flatten()
    #
    #         step2_data_list.append(data_copy)
    #
    #         # Combine all customer data into one DataFrame
    #     step2_data = pd.concat(step2_data_list, ignore_index=True)
    #     self.step2_data = step2_data
    #
    #     # decision-dependent regression
    #     training_data = self.step2_data[self.step2_data['Year'] <= 2020]
    #     self.training_data = training_data
    #     self.step2_regression_data = training_data
    #     test_data = self.step2_data[self.step2_data['Year'] >= 2021]  # Combine test1_data (2021) and test2_data (2022)
    #     self.testing_data = test_data
    #
    #     self.step2_Y_train_dep = training_data['demand'] - training_data['ested_exp_demand'] - training_data[
    #         'initial_demand']
    #     self.step2_Y_train_ind = training_data['demand'] - training_data['initial_demand']
    #     self.step2_X_train = training_data[['normalized_education_level', 'normalized_gdp']]
    #
    #     self.test_Y_dep = test_data['demand'] - test_data['ested_exp_demand'] - test_data['initial_demand']
    #     self.test_Y_ind = test_data['demand'] - test_data['initial_demand']
    #     self.test_X = test_data[['normalized_education_level', 'normalized_gdp']]
    #
    #     ### Decision-Dependent Model
    #     # Fit Training dataset for decision-dependent
    #     dep_model = LinearRegression()
    #     dep_model.fit(self.step2_X_train, self.step2_Y_train_dep)
    #     step2_dep_pred = dep_model.predict(self.step2_X_train)
    #     self.step2_dep_intercept = dep_model.intercept_
    #     self.step2_dep_coef = dep_model.coef_
    #     self.step2_dep_mse = mean_squared_error(self.step2_Y_train_dep, step2_dep_pred)
    #     self.step2_dep_median_ae = median_absolute_error(self.step2_Y_train_dep, step2_dep_pred)
    #     self.step2_dep_max_error = max_error(self.step2_Y_train_dep, step2_dep_pred)
    #     self.step2_dep_residuals = self.step2_Y_train_dep - step2_dep_pred
    #     self.step2_dep_r2 = r2_score(self.step2_Y_train_dep, step2_dep_pred)
    #
    #     # Test results for decision-dependent
    #     step2_test_dep_pred = dep_model.predict(self.test_X)
    #     self.step2_dep_mse_test = mean_squared_error(self.test_Y_dep, step2_test_dep_pred)
    #     self.step2_dep_median_ae_test = median_absolute_error(self.test_Y_dep, step2_test_dep_pred)
    #     self.step2_dep_max_error_test = max_error(self.test_Y_dep, step2_test_dep_pred)
    #     self.step2_dep_residuals_test = self.test_Y_dep - step2_test_dep_pred
    #     self.step2_dep_r2_test = r2_score(self.test_Y_dep, step2_test_dep_pred)
    #
    #     ### Decision-Independent Model
    #     # Fit Training dataset for decision-independent
    #     ind_model = LinearRegression()
    #     ind_model.fit(self.step2_X_train, self.step2_Y_train_ind)
    #     step2_ind_pred = ind_model.predict(self.step2_X_train)
    #     self.step2_ind_intercept = ind_model.intercept_
    #     self.step2_ind_coef = ind_model.coef_
    #     self.step2_ind_mse = mean_squared_error(self.step2_Y_train_ind, step2_ind_pred)
    #     self.step2_ind_median_ae = median_absolute_error(self.step2_Y_train_ind, step2_ind_pred)
    #     self.step2_ind_max_error = max_error(self.step2_Y_train_ind, step2_ind_pred)
    #     self.step2_ind_residuals = self.step2_Y_train_ind - step2_ind_pred
    #     self.step2_ind_r2 = r2_score(self.step2_Y_train_ind, step2_ind_pred)
    #
    #     # Test results for decision-independent
    #     step2_test_ind_pred = ind_model.predict(self.test_X)
    #     self.step2_ind_mse_test = mean_squared_error(self.test_Y_ind, step2_test_ind_pred)
    #     self.step2_ind_median_ae_test = median_absolute_error(self.test_Y_ind, step2_test_ind_pred)
    #     self.step2_ind_max_error_test = max_error(self.test_Y_ind, step2_test_ind_pred)
    #     self.step2_ind_residuals_test = self.test_Y_ind - step2_test_ind_pred
    #     self.step2_ind_r2_test = r2_score(self.test_Y_ind, step2_test_ind_pred)
    #
    #     coefficient_df = pd.DataFrame({
    #         'Model': ['Decision-Dependent', 'Decision-Independent'],
    #         'Intercept': [self.step2_dep_intercept, self.step2_ind_intercept],
    #         'Coef_Education': [self.step2_dep_coef[0], self.step2_ind_coef[0]],
    #         'Coef_GDP': [self.step2_dep_coef[1], self.step2_ind_coef[1]],
    #         'MSE_Test': [self.step2_dep_mse_test, self.step2_ind_mse_test],
    #         'Median_AE_Test': [self.step2_dep_median_ae_test, self.step2_ind_median_ae_test],
    #         'Max_Error_Test': [self.step2_dep_max_error_test, self.step2_ind_max_error_test],
    #         'R_2_Train': [self.step2_dep_r2, self.step2_ind_r2],
    #         'R_2_Test': [self.step2_dep_r2_test, self.step2_ind_r2_test]
    #     })
    #
    #     ### Store results in DataFrame
    #     # regression_results = pd.DataFrame({
    #     #     'Model': ['Decision-Dependent', 'Decision-Independent'],
    #     #     'MSE_Train': [self.step2_dep_mse, self.step2_ind_mse],
    #     #     'MSE_Test': [self.step2_dep_mse_test, self.step2_ind_mse_test]
    #     # })
    #
    #     regression_results = pd.DataFrame({
    #         'Model': ['Decision-Dependent', 'Decision-Independent'],
    #         'MSE_Train': [self.step2_dep_mse, self.step2_ind_mse],
    #         'Median_AE_Train': [self.step2_dep_median_ae, self.step2_ind_median_ae],
    #         'Max_Error_Train': [self.step2_dep_max_error, self.step2_ind_max_error],
    #         'R_2_Train': [self.step2_dep_r2, self.step2_ind_r2]
    #     })
    #
    #     # Create a DataFrame with the coefficients and intercepts in two rows
    #     coefficient_df = pd.DataFrame({
    #         'Model': ['Dependent', 'Independent'],
    #         'Intercept': [self.step2_dep_intercept, self.step2_ind_intercept],
    #         'Coef 1': [self.step2_dep_coef[0], self.step2_ind_coef[0]],
    #         'Coef 2': [self.step2_dep_coef[1], self.step2_ind_coef[1]]
    #     })
    #
    #     # self.plot_step2_regression()
    #
    #     return regression_results, coefficient_df

    def step2_regression(self, data):
        original_data = data
        grouped_data = original_data.groupby(['customer_index', 'seed'])

        z_county = {}
        z_dist = {}
        z_capacity = {}

        step2_data_list = []

        for (customer, seed), group in grouped_data:
            # Filter data for the current customer
            data = group[(group['customer_index'] == customer) & (group['seed'] == seed)]

            # Number of stations and rows for the current customer
            z_rows = len(data)  # z_rows is stations plus 1
            z_stations = z_rows - 1
            z_dist[seed] = {}
            z_capacity[seed] = {}
            z_county[seed] = {}

            # Generate the lower triangular matrix for z_county
            z_county[seed][customer] = np.tril(np.ones((z_rows, z_stations)), -1)

            # Distance matrix for z_dist (just the distance column for this customer)
            z_dist[seed][customer] = data['distance'][1:].values
            z_capacity[seed][customer] = data['capacity'][1:].values

            z_matrix = z_county[seed][customer]
            z_dist_temp = z_dist[seed][customer]
            z_capacity_temp = z_capacity[seed][customer]

            #  # print(z_matrix, z_dist_temp, z_time_temp, z_capacity_temp)

            capacity_matrix = z_matrix @ np.diag(z_capacity_temp)
            #  # print(capacity_matrix)
            exp_term = np.exp(self.step1_intercept + self.step1_coef * z_dist_temp)
            #  # print(exp_term)

            exp_demand = capacity_matrix @ exp_term.reshape(-1, 1)
            exp_demand = exp_demand.flatten()
            #  # print(exp_demand)

            # Create a DataFrame for the demand and add it as a new column
            demand_df = pd.DataFrame(exp_demand, columns=['ested_exp_demand'])
            #  # print(demand_df)
            data_copy = data.copy()
            data_copy['ested_exp_demand'] = demand_df['ested_exp_demand'].values
            #  # print(data_copy)

            # add H1, H2, H3, H4 columns
            # H1: exp_term = 1 (sum_capacity)
            exp_term_h1 = np.ones_like(z_dist_temp)
            exp_demand_h1 = capacity_matrix @ exp_term_h1.reshape(-1, 1)

            # H2: exp_term = exp(-distance)
            exp_term_h2 = np.exp(-z_dist_temp)
            exp_demand_h2 = capacity_matrix @ exp_term_h2.reshape(-1, 1)

            # H3: exp_term = exp(-distance/25)
            exp_term_h3 = np.exp(-z_dist_temp / 25)
            exp_demand_h3 = capacity_matrix @ exp_term_h3.reshape(-1, 1)

            # H4: exp_term = exp(step1_coef * distance + step1_intercept)
            exp_term_h4 = np.exp(self.step1_intercept + self.step1_coef * z_dist_temp)
            exp_demand_h4 = capacity_matrix @ exp_term_h4.reshape(-1, 1)

            data_copy['sum_capacity'] = exp_demand_h1.flatten()
            data_copy['exp_distance_capacity'] = exp_demand_h2.flatten()
            data_copy['assumption_exp_distance_capacity'] = exp_demand_h3.flatten()
            data_copy['step1_exp_distance_capacity'] = exp_demand_h4.flatten()

            step2_data_list.append(data_copy)

            # Combine all customer data into one DataFrame
        step2_data = pd.concat(step2_data_list, ignore_index=True)
        self.step2_data = step2_data

        # decision-dependent regression
        training_data = self.step2_data[self.step2_data['Year'] <= 2020]
        self.training_data = training_data
        self.step2_regression_data = training_data
        test_data = self.step2_data[self.step2_data['Year'] >= 2021]  # Combine test1_data (2021) and test2_data (2022)
        self.testing_data = test_data

        self.step2_Y_train_dep = training_data['demand'] - training_data['ested_exp_demand'] - training_data[
            'initial_demand']
        self.step2_Y_train_ind = training_data['demand'] - training_data['initial_demand']
        self.step2_X_train = training_data[['normalized_education_level', 'normalized_gdp']]

        self.test_Y_dep = test_data['demand'] - test_data['ested_exp_demand'] - test_data['initial_demand']
        self.test_Y_ind = test_data['demand'] - test_data['initial_demand']
        self.test_X = test_data[['normalized_education_level', 'normalized_gdp']]

        ### Decision-Dependent Model
        # Fit Training dataset for decision-dependent
        dep_model = LinearRegression()
        dep_model.fit(self.step2_X_train, self.step2_Y_train_dep)
        step2_dep_pred = dep_model.predict(self.step2_X_train)
        self.step2_dep_intercept = dep_model.intercept_
        self.step2_dep_coef = dep_model.coef_
        self.step2_dep_mse = mean_squared_error(self.step2_Y_train_dep, step2_dep_pred)
        self.step2_dep_median_ae = median_absolute_error(self.step2_Y_train_dep, step2_dep_pred)
        self.step2_dep_max_error = max_error(self.step2_Y_train_dep, step2_dep_pred)
        self.step2_dep_residuals = self.step2_Y_train_dep - step2_dep_pred
        self.step2_dep_r2 = r2_score(self.step2_Y_train_dep, step2_dep_pred)

        # Test results for decision-dependent
        step2_test_dep_pred = dep_model.predict(self.test_X)
        self.step2_dep_mse_test = mean_squared_error(self.test_Y_dep, step2_test_dep_pred)
        self.step2_dep_median_ae_test = median_absolute_error(self.test_Y_dep, step2_test_dep_pred)
        self.step2_dep_max_error_test = max_error(self.test_Y_dep, step2_test_dep_pred)
        self.step2_dep_residuals_test = self.test_Y_dep - step2_test_dep_pred
        self.step2_dep_r2_test = r2_score(self.test_Y_dep, step2_test_dep_pred)

        ### Decision-Independent Model
        # Fit Training dataset for decision-independent
        ind_model = LinearRegression()
        ind_model.fit(self.step2_X_train, self.step2_Y_train_ind)
        step2_ind_pred = ind_model.predict(self.step2_X_train)
        self.step2_ind_intercept = ind_model.intercept_
        self.step2_ind_coef = ind_model.coef_
        self.step2_ind_mse = mean_squared_error(self.step2_Y_train_ind, step2_ind_pred)
        self.step2_ind_median_ae = median_absolute_error(self.step2_Y_train_ind, step2_ind_pred)
        self.step2_ind_max_error = max_error(self.step2_Y_train_ind, step2_ind_pred)
        self.step2_ind_residuals = self.step2_Y_train_ind - step2_ind_pred
        self.step2_ind_r2 = r2_score(self.step2_Y_train_ind, step2_ind_pred)

        # Test results for decision-independent
        step2_test_ind_pred = ind_model.predict(self.test_X)
        self.step2_ind_mse_test = mean_squared_error(self.test_Y_ind, step2_test_ind_pred)
        self.step2_ind_median_ae_test = median_absolute_error(self.test_Y_ind, step2_test_ind_pred)
        self.step2_ind_max_error_test = max_error(self.test_Y_ind, step2_test_ind_pred)
        self.step2_ind_residuals_test = self.test_Y_ind - step2_test_ind_pred
        self.step2_ind_r2_test = r2_score(self.test_Y_ind, step2_test_ind_pred)

        coefficient_df = pd.DataFrame({
            'Model': ['Decision-Dependent', 'Decision-Independent'],
            'Intercept': [self.step2_dep_intercept, self.step2_ind_intercept],
            'Coef_Education': [self.step2_dep_coef[0], self.step2_ind_coef[0]],
            'Coef_GDP': [self.step2_dep_coef[1], self.step2_ind_coef[1]],
            'MSE_Test': [self.step2_dep_mse_test, self.step2_ind_mse_test],
            'Median_AE_Test': [self.step2_dep_median_ae_test, self.step2_ind_median_ae_test],
            'Max_Error_Test': [self.step2_dep_max_error_test, self.step2_ind_max_error_test],
            'R_2_Train': [self.step2_dep_r2, self.step2_ind_r2],
            'R_2_Test': [self.step2_dep_r2_test, self.step2_ind_r2_test]
        })

        ### Store results in DataFrame
        # regression_results = pd.DataFrame({
        #     'Model': ['Decision-Dependent', 'Decision-Independent'],
        #     'MSE_Train': [self.step2_dep_mse, self.step2_ind_mse],
        #     'MSE_Test': [self.step2_dep_mse_test, self.step2_ind_mse_test]
        # })

        regression_results = pd.DataFrame({
            'Model': ['Decision-Dependent', 'Decision-Independent'],
            'MSE_Train': [self.step2_dep_mse, self.step2_ind_mse],
            'Median_AE_Train': [self.step2_dep_median_ae, self.step2_ind_median_ae],
            'Max_Error_Train': [self.step2_dep_max_error, self.step2_ind_max_error],
            'R_2_Train': [self.step2_dep_r2, self.step2_ind_r2]
        })

        # Create a DataFrame with the coefficients and intercepts in two rows
        coefficient_df = pd.DataFrame({
            'Model': ['Dependent', 'Independent'],
            'Intercept': [self.step2_dep_intercept, self.step2_ind_intercept],
            'Coef 1': [self.step2_dep_coef[0], self.step2_ind_coef[0]],
            'Coef 2': [self.step2_dep_coef[1], self.step2_ind_coef[1]]
        })

        # self.plot_step2_regression()

        return regression_results, coefficient_df

    def plot_step2_regression(self):
        # Extract features and responses for decision-dependent and decision-independent models
        education = self.step2_X_train['normalized_education_level']
        gdp = self.step2_X_train['normalized_gdp']
        dep_response = self.step2_Y_train_dep
        ind_response = self.step2_Y_train_ind

        # Predicted values
        dep_pred = self.step2_dep_intercept + self.step2_dep_coef[0] * education + self.step2_dep_coef[1] * gdp
        ind_pred = self.step2_ind_intercept + self.step2_ind_coef[0] * education + self.step2_ind_coef[1] * gdp

        # Plot Education vs Response
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(education, dep_response, color='blue', label='Decision-Dependent Actual')
        plt.scatter(education, dep_pred, color='red', label='Decision-Dependent Predicted')
        plt.xlabel('Education')
        plt.ylabel('Response (Decision-Dependent)')
        plt.title('Step 2 Regression: Education vs. Decision-Dependent Response')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.scatter(education, ind_response, color='green', label='Decision-Independent Actual')
        plt.scatter(education, ind_pred, color='orange', label='Decision-Independent Predicted')
        plt.xlabel('Education')
        plt.ylabel('Response (Decision-Independent)')
        plt.title('Step 2 Regression: Education vs. Decision-Independent Response')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        # Plot GDP vs Response
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(gdp, dep_response, color='blue', label='Decision-Dependent Actual')
        plt.scatter(gdp, dep_pred, color='red', label='Decision-Dependent Predicted')
        plt.xlabel('GDP')
        plt.ylabel('Response (Decision-Dependent)')
        plt.title('Step 2 Regression: GDP vs. Decision-Dependent Response')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.scatter(gdp, ind_response, color='green', label='Decision-Independent Actual')
        plt.scatter(gdp, ind_pred, color='orange', label='Decision-Independent Predicted')
        plt.xlabel('GDP')
        plt.ylabel('Response (Decision-Independent)')
        plt.title('Step 2 Regression: GDP vs. Decision-Independent Response')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def one_step_neuralnetworks_data(self, data):
        regression_data = data.copy()
        regression_data = regression_data.sort_values(by=['customer_index', 'Open Date']).reset_index(drop=True)
        grouped = regression_data.groupby(['customer_index'])

        for index, group in grouped:
            group_data = group.copy()

            # ✅ 直接在cumsum时减去第一行的值（一步到位）
            first_capacity = group_data['capacity'].iloc[0]
            first_distance = group_data['distance'].iloc[0]

            regression_data.loc[group_data.index, 'sum_capacity'] = \
                group_data['capacity'].cumsum() - first_capacity

            regression_data.loc[group_data.index, 'exp_distance_capacity'] = \
                (np.exp(-group_data['distance']) * group_data['capacity']).cumsum() - \
                (np.exp(-first_distance) * first_capacity)

            regression_data.loc[group_data.index, 'assumption_exp_distance_capacity'] = \
                (np.exp(- group_data['distance']/25) * group_data[
                    'capacity']).cumsum() - \
                (np.exp(-first_distance/25) * first_capacity)

            regression_data.loc[group_data.index, 'step1_exp_distance_capacity'] = \
                (np.exp(self.step1_intercept + self.step1_coef * group_data['distance']) * group_data[
                    'capacity']).cumsum() - \
                (np.exp(self.step1_intercept + self.step1_coef * first_distance) * first_capacity)

            # regression_data.loc[group_data.index, 'max_distance_capacity'] = \
            #     (np.maximum(1 - group_data['distance'] / 30, 0) * group_data['capacity']).cumsum() - \
            #     (np.maximum(1 - first_distance / 30, 0) * first_capacity)

        return regression_data

    def nonparametric_regression_tuning(self, train_data, test_data, features, feature_type):
        # Prepare features and response
        x = train_data[features].values
        y = train_data['demand'] - train_data['initial_demand'].values

        # Define models including Neural Network
        models = {
            'Decision Tree': DecisionTreeRegressor(),
            'Random Forest': RandomForestRegressor(),
            'Gradient Boosting': GradientBoostingRegressor(),
            'Neural Network': MLPRegressor(
                max_iter=5000,
                early_stopping=True,
                n_iter_no_change=50,
                validation_fraction=0.1,
                random_state=42
            )
        }

        # Parameter grids for hyperparameter tuning
        param_grids = {
            'Decision Tree': {
                'max_depth': range(3, 10),
                'min_samples_split': range(2, 10),
                'random_state': range(40, 50)
            },
            'Random Forest': {
                'random_state': range(40, 50),
                'n_estimators': [50, 100, 150],
                'max_depth': range(3, 10),
                'min_samples_split': range(2, 10)
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 150],
                'max_depth': range(3, 10),
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'Neural Network': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu'],
                'solver': ['adam', 'lbfgs'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['adaptive'],
                'learning_rate_init': [0.001, 0.01],
            }
        }

        # Initialize dictionaries to store best models and results
        if not hasattr(self, 'best_models_dep'):
            self.best_models_dep = {}
        if not hasattr(self, 'nonparametric_regression_results'):
            self.nonparametric_regression_results = {'decision_dependent': {}}
        if not hasattr(self, 'residuals_dep'):
            self.residuals_dep = {}
        if not hasattr(self, 'residuals_test_dep'):
            self.residuals_test_dep = {}
        print("=" * 80)
        print("NONPARAMETRIC MODEL TRAINING")
        print("=" * 80)

        train_x = x
        train_y = y

        test_x = test_data[features].values
        test_y = (test_data['demand'] - test_data['initial_demand']).values

        # Tune for decision-dependent
        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")

            # ✅ Start timing for EACH individual model
            model_start_time = time.time()

            # ✅ 对Neural Network使用RandomizedSearchCV以加速训练
            if model_name == 'Neural Network':
                grid_search_dep = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grids[model_name],
                    n_iter=50,
                    scoring='neg_mean_squared_error',
                    cv=5,
                    verbose=1,
                    random_state=42
                )
            else:
                grid_search_dep = GridSearchCV(
                    estimator=model,
                    param_grid=param_grids[model_name],
                    scoring='neg_mean_squared_error',
                    cv=5,
                    verbose=1
                )

            grid_search_dep.fit(x, y)
            dep_best_model = grid_search_dep.best_estimator_

            # ✅ End timing for individual model
            model_runtime = time.time() - model_start_time

            print(f"Best parameters for {model_name}:")
            for param, value in grid_search_dep.best_params_.items():
                print(f"  {param}: {value}")
            print(f"Runtime: {model_runtime:.2f} seconds")

            # Store best model and predictions
            model_key = f"{model_name}_{feature_type}"
            self.best_models_dep[model_key] = dep_best_model

            train_dep_pred = dep_best_model.predict(train_x)
            test_y_pred_dep = dep_best_model.predict(test_x)

            # Compute metrics for training set
            train_metrics_dep = {
                'mse': mean_squared_error(train_y, train_dep_pred),
                'r2': r2_score(train_y, train_dep_pred),
                'mae': mean_absolute_error(train_y, train_dep_pred),
                'medae': median_absolute_error(train_y, train_dep_pred),
                'max_error': max_error(train_y, train_dep_pred)
            }

            # Compute metrics for test set
            test_metrics_dep = {
                'mse': mean_squared_error(test_y, test_y_pred_dep),
                'r2': r2_score(test_y, test_y_pred_dep),
                'mae': mean_absolute_error(test_y, test_y_pred_dep),
                'medae': median_absolute_error(test_y, test_y_pred_dep),
                'max_error': max_error(test_y, test_y_pred_dep)
            }

            # ✅ Store runtime for EACH model
            self.nonparametric_regression_results['decision_dependent'][model_name] = {
                'train_metrics': train_metrics_dep,
                'test_metrics': test_metrics_dep,
                'best_params': grid_search_dep.best_params_,
                'runtime': model_runtime  # ✅ Individual model runtime
            }

            # Residuals: actual - predicted
            self.residuals_dep[model_name] = train_y - train_dep_pred
            self.residuals_test_dep[model_name] = test_y - test_y_pred_dep

            print(f"{model_name} - Train MSE: {train_metrics_dep['mse']:.2f}, "
                  f"Test MSE: {test_metrics_dep['mse']:.2f}, "
                  f"Train R²: {train_metrics_dep['r2']:.4f}, "
                  f"Test R²: {test_metrics_dep['r2']:.4f}")

        # Prepare the data for DataFrame
        data_list = []
        for model_name in self.nonparametric_regression_results['decision_dependent'].keys():
            dep_results = self.nonparametric_regression_results['decision_dependent'][model_name]

            data_list.append([
                model_name,
                dep_results['train_metrics']['mse'],
                dep_results['test_metrics']['mse'],
                dep_results['train_metrics']['mae'],
                dep_results['test_metrics']['mae'],
                dep_results['train_metrics']['medae'],
                dep_results['test_metrics']['medae'],
                dep_results['train_metrics']['max_error'],
                dep_results['test_metrics']['max_error'],
                dep_results['train_metrics']['r2'],
                dep_results['test_metrics']['r2'],
                dep_results['runtime']  # ✅ Add runtime column
            ])

        df_columns = [
            'Model Name',
            'Train_MSE', 'Test_MSE',
            'Train_MAE', 'Test_MAE',
            'Train_MedAE', 'Test_MedAE',
            'Train_MaxError', 'Test_MaxError',
            'Train_R2', 'Test_R2',
            'Runtime'  # ✅ Runtime column
        ]

        self.nonparametric_df = pd.DataFrame(data_list, columns=df_columns)

        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print(self.nonparametric_df.to_string())

        return self.nonparametric_df

    def nonparametric_verify(self):
        self.two_step_results = pd.DataFrame()
        self.nonparametric_results_all = pd.DataFrame()
        """
        Run nonparametric regression models and compare results
        """
        # Initialize lists to store results for each model
        model_names = ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'Neural Network']
        alpha1_dep_list_two_step = []
        beta_dep_distance_list_two_step = []
        beta_dep_education_list_two_step = []
        beta_dep_gdp_list_two_step = []
        intercept_dep_list_two_step = []
        num_data_list_step1 = []
        num_data_list_step2 = []
        mse_list_two_step = []
        mse_test_two_step = []
        runtime_two_step = []

        # Dictionary to store all results
        train_mse_results = {model: [] for model in model_names}
        test_mse_results = {model: [] for model in model_names}
        train_mae_results = {model: [] for model in model_names}
        test_mae_results = {model: [] for model in model_names}
        train_r2_results = {model: [] for model in model_names}
        test_r2_results = {model: [] for model in model_names}
        runtime_results = {model: [] for model in model_names}  # ✅ Individual model runtimes
        best_params_results = {model: [] for model in model_names}

        num_data_list_train = []
        num_data_list_test = []

        seed_list = range(16, 26)
        step2_orig_data = pd.DataFrame()
        step1_reg_data = pd.DataFrame()
        nonparametric_results_all = pd.DataFrame()

        # create output_dir
        date_time = datetime.now().strftime("%Y%m%d")

        base_dir = "/Users/huangrongsun/Library/CloudStorage/OneDrive-TheOhioStateUniversity/EVChargingLocationWin"
        output_dir = os.path.join(base_dir, "nonparametric_results", date_time)

        os.makedirs(output_dir, exist_ok=True)

        scenario_labels = []

        # --- 2. 第一阶段：生成并缓存数据 (Seeds 16-28) ---
        all_seeds = range(16, 29)  # 16 to 28
        data_storage = {}

        print(">>> Starting Data Generation for seeds 16-28...")
        for seed in all_seeds:
            # 生成数据
            self.assign_random_dates(seed)
            experiment.data_generation(seed)
            temp_data = experiment.regression_data
            temp_data['seed'] = seed

            # 分割并缓存
            step1_data_part, step1_reg_data_part = self.step1_regression_data(temp_data)
            step1_data_part = step1_data_part.dropna()

            data_storage[seed] = {
                'step1_reg': step1_reg_data_part,
                'step2_orig': step1_data_part
            }

        self.data_storage = data_storage

        # --- 3. 第二阶段：定义场景并运行回归 ---

        # 定义场景
        seeds_scenario_1 = list(range(16, 24)) + [25]  # 16-23 + 25
        seeds_scenario_2 = list(range(16, 27))  # 16-26
        seeds_scenario_3 = list(range(16, 28))  # 16-27
        # seeds_scenario_4 = list(range(16, 24)) + [25, 26]

        scenarios = [
            ("Seeds 16-23 + 25", seeds_scenario_1),
            ("Seeds 16-26", seeds_scenario_2),
            ("Seeds 16-27", seeds_scenario_3)
        ]
        # scenarios = [
        #     ("seeds 16-23 + 25 + 26", seeds_scenario_4)
        # ]

        one_step_results = pd.DataFrame()

        for label, current_seeds in scenarios:
            print(f"\n--- Running Scenario: {label} ---")
            scenario_labels.append(label)

            # 3.1 拼接数据
            current_step1_reg_data = pd.DataFrame()
            current_step2_orig_data = pd.DataFrame()

            for s in current_seeds:
                current_step1_reg_data = pd.concat([current_step1_reg_data, data_storage[s]['step1_reg']],
                                                   ignore_index=True)
                current_step2_orig_data = pd.concat([current_step2_orig_data, data_storage[s]['step2_orig']],
                                                    ignore_index=True)

            print("Scenario Data Prepared:"
                  " Step 1 Reg Data Points =", len(current_step1_reg_data),
                  ", Step 2 Orig Data Points =", len(current_step2_orig_data))

        #     self.step2_orig_data = current_step2_orig_data
        #
        #     num_data_list_step2.append(len(current_step2_orig_data))
        #     num_data_list_step1.append(len(current_step1_reg_data))
        #
        #     # ---------------------------
        #     # 3.2 Two-Step Regression
        #     # ---------------------------
        #     start_time = time.time()
        #     self.step1_regression(current_step1_reg_data)
        #     self.step2_regression(current_step2_orig_data)
        #     runtime_two_step.append(time.time() - start_time)
        #
        #     # 获取并存储 Two-Step 结果
        #     alpha1_dep_list_two_step.append(self.step1_intercept)
        #     beta_dep_distance_list_two_step.append(self.step1_coef[0])
        #     beta_dep_education_list_two_step.append(self.step2_dep_coef[0])
        #     beta_dep_gdp_list_two_step.append(self.step2_dep_coef[1])
        #     intercept_dep_list_two_step.append(self.step2_dep_intercept)
        #     mse_list_two_step.append(self.step2_dep_mse)
        #     mse_test_two_step.append(self.step2_dep_mse_test)
        #
        #     print(f"Data points: {len(current_step2_orig_data)}")
        #
        # # for seed in seed_list:
        # #     print(f"\n{'=' * 80}")
        # #     print(f"Processing Seed: {seed}")
        # #     print(f"{'=' * 80}")
        # #
        # #     random_dates = self.assign_random_dates(seed)
        # #     experiment.data_generation(seed)
        # #     temp_data = experiment.regression_data
        # #     temp_data['seed'] = seed
        # #
        # #     step1_data, step1_regression_data = self.step1_regression_data(temp_data)
        # #     step1_data = step1_data.dropna()
        # #     step1_reg_data = pd.concat([step1_reg_data, step1_regression_data], ignore_index=True)
        # #     self.step1_reg_data = step1_reg_data
        # #     step2_orig_data = pd.concat([step2_orig_data, step1_data], ignore_index=True)
        # #     self.step2_orig_data = step2_orig_data
        # #     num_data_list_step2.append(len(step2_orig_data))
        # #     num_data_list_step1.append(len(step1_reg_data))
        # #
        # #     # Start time for two_step
        # #     start_time = time.time()
        # #     self.step1_regression(step1_reg_data)
        # #     reg_results, coef_df = self.step2_regression(step2_orig_data)
        # #     runtime_two_step.append(time.time() - start_time)
        # #
        # #     reg_results['seed'] = seed
        # #     coef_df['seed'] = seed
        # #     reg_results = pd.concat([reg_results, reg_results], ignore_index=True)
        # #     coef_df = pd.concat([coef_df, coef_df], ignore_index=True)
        # #
        # #     # Get estimated results
        # #     alpha1_dep = self.step1_intercept
        # #     beta_distance_dep = self.step1_coef[0]
        # #     beta_education_dep = self.step2_dep_coef[0]
        # #     beta_gdp_dep = self.step2_dep_coef[1]
        # #     intercept_dep = self.step2_dep_intercept
        # #     mse_dep = self.step2_dep_mse
        # #     mse_test = self.step2_dep_mse_test
        # #
        # #     # Append results
        # #     alpha1_dep_list_two_step.append(alpha1_dep)
        # #     beta_dep_distance_list_two_step.append(beta_distance_dep)
        # #     beta_dep_education_list_two_step.append(beta_education_dep)
        # #     beta_dep_gdp_list_two_step.append(beta_gdp_dep)
        # #     intercept_dep_list_two_step.append(intercept_dep)
        # #     mse_list_two_step.append(mse_dep)
        # #     mse_test_two_step.append(mse_test)
        # #
        # #     print(f"Two-Step Results:")
        # #     print(f"  Train MSE: {mse_dep:.2f}")
        # #     print(f"  Test MSE: {mse_test:.2f}")
        # #     print(f"  Train R²: {self.step2_dep_r2:.4f}")
        # #     print(f"  Test R²: {self.step2_dep_r2_test:.4f}")
        #
        #     # Generate neural_networks data
        #     # data_nn = self.one_step_neuralnetworks_data(step2_orig_data)
        #     # self.data_nn = data_nn
        #     #
        #     # # Train dataset before year <= 2020
        #     # train_data = data_nn[data_nn['Year'] <= 2020]
        #     # test_data = data_nn[data_nn['Year'] >= 2021]
        #
        #     train_data = self.training_data
        #     test_data = self.testing_data
        #
        #     # self.train_ml = train_data
        #     # self.test_ml = test_data
        #
        #     num_data_list_train.append(len(train_data))
        #     num_data_list_test.append(len(test_data))
        #
        #     print(f"\nTraining data size: {len(train_data)}")
        #     print(f"Test data size: {len(test_data)}")
        #     print(f"Number of data points: {len(step2_orig_data)}")
        #
        #     # ✅ Define feature_list as a LIST (not a set) to maintain order
        #     # feature_list = [
        #     #     ['normalized_education_level', 'normalized_gdp', 'sum_capacity'],
        #     #     ['normalized_education_level', 'normalized_gdp', 'exp_distance_capacity'],
        #     #     ['normalized_education_level', 'normalized_gdp', 'step1_exp_distance_capacity'],
        #     #     ['normalized_education_level', 'normalized_gdp', 'distance', 'capacity']
        #     # ]
        #
        #     feature_list = [
        #         ['normalized_education_level', 'normalized_gdp', 'sum_capacity'],
        #         ['normalized_education_level', 'normalized_gdp', 'exp_distance_capacity'],
        #         ['normalized_education_level', 'normalized_gdp', 'assumption_exp_distance_capacity'],
        #         ['normalized_education_level', 'normalized_gdp', 'step1_exp_distance_capacity']
        #     ]
        #
        #     # Map feature_list to 4 feature type H1, H2, H3, H4
        #     feature_type_map = ['H1', 'H2', 'H3', 'H4']
        #
        #     print(f"\n{'=' * 80}")
        #     print("Starting Nonparametric Regression Training...")
        #     print(f"{'=' * 80}")
        #
        #     # ✅ Loop through all 4 feature versions
        #     for idx, features in enumerate(feature_list):
        #         feature_type = feature_type_map[idx]
        #         print(f"\n{'=' * 80}")
        #         print(f"Training with Feature Set {feature_type}: {features}")
        #         print(f"{'=' * 80}")
        #
        #         nonparametric_df = self.nonparametric_regression_tuning(train_data, test_data, features, feature_type)
        #         # ✅ Add Feature_Type column to identify which feature set
        #         nonparametric_df['Feature_Type'] = feature_type
        #         nonparametric_df['scenario'] = label
        #         # record data size
        #         nonparametric_df['Num_Train_Data'] = len(train_data)
        #         nonparametric_df['Num_Test_Data'] = len(test_data)
        #
        #         # Store results for each model with this feature set
        #         for model_name in model_names:
        #             if model_name in self.nonparametric_regression_results['decision_dependent']:
        #                 results = self.nonparametric_regression_results['decision_dependent'][model_name]
        #
        #                 train_mse_results[model_name].append(results['train_metrics']['mse'])
        #                 test_mse_results[model_name].append(results['test_metrics']['mse'])
        #                 train_mae_results[model_name].append(results['train_metrics']['mae'])
        #                 test_mae_results[model_name].append(results['test_metrics']['mae'])
        #                 train_r2_results[model_name].append(results['train_metrics']['r2'])
        #                 test_r2_results[model_name].append(results['test_metrics']['r2'])
        #                 runtime_results[model_name].append(results['runtime'])  # ✅ Individual runtime
        #                 best_params_results[model_name].append(str(results['best_params']))
        #             else:
        #                 # If model not found, append NaN
        #                 train_mse_results[model_name].append(np.nan)
        #                 test_mse_results[model_name].append(np.nan)
        #                 train_mae_results[model_name].append(np.nan)
        #                 test_mae_results[model_name].append(np.nan)
        #                 train_r2_results[model_name].append(np.nan)
        #                 test_r2_results[model_name].append(np.nan)
        #                 runtime_results[model_name].append(np.nan)
        #                 best_params_results[model_name].append(None)
        #
        #         # ✅ Concatenate results for all feature versions
        #         nonparametric_results_all = pd.concat([nonparametric_results_all, nonparametric_df], ignore_index=True)
        #         # save nonparametric_df to csv
        #         output_file = os.path.join(output_dir, f"nonparametric_results_seed{seed}_featureset{feature_type}.csv")
        #         nonparametric_df.to_csv(output_file, index=False)
        #         print(f"Saved nonparametric results to {output_file}")
        #
        #
        # # Create two-step results DataFrame
        # results_df = pd.DataFrame({
        #     'Num Data Points(Step1)': num_data_list_step1,
        #     'Num Data Points(Step2)': num_data_list_step2,
        #     'Alpha1(Two Step)': alpha1_dep_list_two_step,
        #     'Beta Distance(Two Step)': beta_dep_distance_list_two_step,
        #     'Beta Education(Two Step)': beta_dep_education_list_two_step,
        #     'Beta GDP(Two Step)': beta_dep_gdp_list_two_step,
        #     'Intercept(Two Step)': intercept_dep_list_two_step,
        #     'MSE(Two Step)': mse_list_two_step,
        #     'MSE Test(Two Step)': mse_test_two_step,
        #     'R2(Two Step)': self.step2_dep_r2,
        #     'R2 Test(Two Step)': self.step2_dep_r2_test,
        #     'Runtime(Two Step)': runtime_two_step,
        # })
        # self.two_step_results = results_df
        #
        # # Store all results with feature types
        # self.nonparametric_results_all = nonparametric_results_all
        #
        # print(f"\n{'=' * 80}")
        # print("TWO-STEP REGRESSION RESULTS")
        # print(f"{'=' * 80}")
        # print(self.two_step_results.to_string())
        #
        # print(f"\n{'=' * 80}")
        # print("NONPARAMETRIC REGRESSION RESULTS (All Feature Sets)")
        # print(f"{'=' * 80}")
        # print(self.nonparametric_results_all.to_string())

        return self.two_step_results, self.nonparametric_results_all

    @staticmethod
    def generate_random_point(center, radius, seed):
        # Convert radius from miles to degrees (approximately)
        radius_degrees = radius / 69.0

        random.seed(seed)

        # Generate random angle (0 to 360 degrees)
        random_angle = random.uniform(0, 2 * math.pi)

        # Generate random distance (0 to radius_degrees)
        random_distance = math.sqrt(random.uniform(0, radius_degrees ** 2))

        # Calculate the new latitude and longitude
        new_latitude = center[0] + (random_distance * math.cos(random_angle))
        new_longitude = center[1] + (random_distance * math.sin(random_angle))

        return new_latitude, new_longitude

    @staticmethod
    def generate_grid_points(center, radius, num_points, seed):
        random.seed(seed)
        np.random.seed(seed)

        radius_degrees = radius / 69.0
        num_points_per_side = int(np.sqrt(num_points)) + 1

        latitudes = np.linspace(center[0] - radius_degrees, center[0] + radius_degrees, num_points_per_side)
        longitudes = np.linspace(center[1] - radius_degrees, center[1] + radius_degrees, num_points_per_side)

        grid = [(lat, lon) for lat in latitudes for lon in longitudes]
        np.random.shuffle(grid)  # Shuffle to randomize
        grid = grid[:num_points]

        # Perturb slightly to add randomness
        perturbed_grid = [
            (
                lat + np.random.uniform(-0.01, 0.01),
                lon + np.random.uniform(-0.01, 0.01)
            )
            for lat, lon in grid
        ]

        return perturbed_grid

    def input_data(self):
        # Define the central point (latitude and longitude)
        central_latitude = 40  # Example latitude
        central_longitude = -73  # Example longitude

        # Define the radius in miles
        radius_miles = 15

        # Generate random points for selected_coordinates_df (unchanged)
        random_coordinates_selected = []
        for seed in range(1, self.num_exp_Customers + 1):  # Use a different range of seeds
            random_point = self.generate_random_point((central_latitude, central_longitude), radius_miles, seed)
            random_coordinates_selected.append(random_point)

        selected_coordinates_df = pd.DataFrame(random_coordinates_selected,
                                               columns=['Latitude', 'Longitude']).reset_index(drop=True)
        selected_coordinates_df.iloc[1] = [40.16, -73.08]  # Update row 2 (index 1) to (-73.08, 40.16)
        selected_coordinates_df.iloc[3] = [39.91, -72.92]  # Update row 4 (index 3) to (-72.92, 39.91)
        self.selected_coordinates_df = selected_coordinates_df

        # Generate evenly distributed points for random_points_df (grid-based with perturbation)
        random_coordinates_random = self.generate_grid_points(
            center=(central_latitude, central_longitude),
            radius=radius_miles,
            num_points=self.num_exp_Facilities,
            seed=42  # Different seed for randomization
        )

        random_points_df = pd.DataFrame(random_coordinates_random, columns=['Latitude', 'Longitude']).reset_index(
            drop=True)
        self.random_points_df = random_points_df

        c1 = np.radians(selected_coordinates_df[['Latitude', 'Longitude']].to_numpy())
        c2 = np.radians(random_points_df[['Latitude', 'Longitude']].to_numpy())

        # Compute distances in miles.
        dist = haversine_distances(c2, c1) * 6371 * 0.621371
        self.experiments_distance = pd.DataFrame(dist, index=range(self.num_exp_Facilities),
                                                 columns=range(self.num_exp_Customers))

    def generateSyntheticData(self):
        self.revenue = [10] * self.num_exp_Customers
        self.fixedcost = [8000] * self.num_exp_Facilities  # Fixed cost
        self.chargercost = [400] * self.num_exp_Facilities
        self.transportationcost = 0.005  # Transportation cost (matrix)
        self.penaltycost = [5] * self.num_exp_Customers  # Penalty cost
        self.capacity = [500] * self.num_exp_Facilities
        self.gdp = [0.3] * self.num_exp_Customers
        self.education = [0.2] * self.num_exp_Customers
        # Original GDP: 37923405.8
        # Original Education: 22.68
        # np.random.seed(1216)
        # self.gdp = np.random.uniform(0.5, 1, self.num_exp_Customers).tolist()

        # Generate random education values between 0.15 and 0.25 for each customer
        # self.education = np.random.uniform(0, 1, self.num_exp_Customers).tolist()

    def initialize_for_optimization(self):
        self.input_data()

        self.generateSyntheticData()

        # 生成测试噪声
        np.random.seed(16)
        self.test_noise = np.random.normal(0, self.noise_sd,
                                           (self.test_sizeData, self.num_exp_Customers))

        np.random.seed(16)
        noise = np.random.normal(0, self.noise_sd, (self.test_sizeData, self.num_exp_Customers))
        self.test_noise = noise

    def insample_two_step(self, residuals_dep):
        # Generate random values
        r = self.revenue
        f = self.fixedcost
        c = self.experiments_distance * self.transportationcost
        chargercost = self.chargercost
        C = self.capacity
        p = self.penaltycost
        num_potential = self.num_exp_Facilities
        num_zip = self.num_exp_Customers
        education_array = np.array(self.education)  # Convert education list to array
        gdp_array = np.array(self.gdp)  # Convert GDP list to array
        X = np.vstack((education_array, gdp_array))
        predicted_demands = np.dot(self.step2_dep_coef,
                                   X) + self.step2_dep_intercept  # This gives an array of predictions, one for each customer
        residuals = list(residuals_dep)
        sce = len(residuals)
        prob = [1 / sce] * sce  # Equal probability for each scenario
        # Create initial demand array with base demand + predicted effects
        initial_demand = 15000 + predicted_demands
        initial_demand = initial_demand.tolist()
        orig_demand = pd.DataFrame(np.exp(self.step1_coef[0] * self.experiments_distance + self.step1_intercept))
        exp_demand = {(i, j, k): orig_demand.iloc[i, j] for i in range(num_potential) for j in range(num_zip) for k in
                      range(sce)}
        m = gp.Model('nonparametric_dep')
        m.setParam('MIPGap', 0.01)

        # Variables
        charger = m.addVars(num_potential, vtype=GRB.INTEGER, ub=10, lb=0, name='Charger')
        location = m.addVars(num_potential, vtype=GRB.BINARY, name='Location')
        assign = m.addVars(num_potential, num_zip, sce, lb=0.0, vtype=GRB.CONTINUOUS, name='Assign')
        unmetdemand = m.addVars(num_zip, sce, lb=0.0, vtype=GRB.CONTINUOUS, name='Unmet')
        real_demand = m.addVars(num_zip, sce, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='real_demand')

        # Constraints
        m.addConstrs((charger[i] <= 10 * location[i] for i in range(num_potential)), name='Location')
        m.addConstrs(
            (gp.quicksum(assign[(i, j, k)] for i in range(num_potential)) + unmetdemand[(j, k)] == real_demand[
                (j, k)]
             for j in range(num_zip) for k in range(sce)), name='Demand')
        m.addConstrs(
            (gp.quicksum(assign[(i, j, k)] for j in range(num_zip)) <= C[i] * charger[i] for i in
             range(num_potential)
             for k in range(sce)), name='Capacity')
        m.addConstrs((real_demand[(j, k)] == initial_demand[j] + gp.quicksum(
            exp_demand[(i, j, k)] * charger[i] for i in range(num_potential))
                      + residuals[k] for j in range(num_zip) for k in range(sce)), name='Real_Demand')

        # Objective function components
        fixed_cost = location.prod(f)
        charger_cost = charger.prod(chargercost)
        transportation_cost = gp.quicksum(assign[(i, j, k)] * c.iloc[i, j] * prob[k] for i in range(num_potential)
                                          for j in range(num_zip) for k in range(sce))
        penalty_cost = gp.quicksum(unmetdemand[j, k] * p[j] * prob[k] for j in range(num_zip) for k in range(sce))
        revenue = gp.quicksum(r[j] * real_demand[(j, k)] * prob[k] for j in range(num_zip) for k in range(sce))

        # Set the objective function
        m.setObjective(fixed_cost + charger_cost + transportation_cost + penalty_cost - revenue, GRB.MINIMIZE)
        m.optimize()
        m.write("insample_dep.lp")

        if m.status == GRB.OPTIMAL:
            # Extract decision variable values
            location_values = [location[i].X for i in range(num_potential)]
            charger_values = [charger[i].X for i in range(num_potential)]
            optimal_value = m.objVal
            revenue_value = sum(r[j] * real_demand[(j, k)].X * prob[k] for j in range(num_zip) for k in range(sce))
            real_demand_values = []
            for j in range(self.num_exp_Customers):
                real_demand = initial_demand[j] + sum(
                    orig_demand.iloc[i, j] * location_values[i] for i in range(self.num_exp_Facilities))
                real_demand_values.append(real_demand)
            self.dep_decision = charger_values
            print('location', [location[i].X for i in range(num_potential)])
            print('charger', [charger[i].X for i in range(num_potential)])
            print('fixed cost insample dep', sum(location[i].X * f[i] for i in range(num_potential)))
            print('charger cost insample dep', sum(charger[i].X * chargercost[i] for i in range(num_potential)))
            print('penalty cost insample dep', penalty_cost.getValue())
            print('transportation cost insample dep', transportation_cost.getValue())
            print('revenue insample dep', revenue.getValue())
            return charger_values, optimal_value, revenue_value, real_demand_values

    def insample_ml(self, model_name, train_data, feature_type, features):
        """
        Use trained ML model to make decisions via optimization
        """
        # Get the trained model

        # 准备训练数据
        response = train_data['demand'] - train_data['initial_demand']
        x = train_data[features].values
        y = response.values

        # 训练模型
        model_key = f"{model_name}_{feature_type}"
        if model_key not in self.best_models_dep:
            raise ValueError(f"Model {model_name} not found in trained models")

        trained_model = self.best_models_dep[model_key]
        trained_model.fit(x, y)

        # 计算残差
        residuals = y - trained_model.predict(x)
        sce = len(residuals)
        prob = [1 / sce] * sce

        # 优化参数
        r = self.revenue
        f = self.fixedcost
        c = self.experiments_distance * self.transportationcost
        chargercost = self.chargercost
        C = self.capacity
        p = self.penaltycost
        num_potential = self.num_exp_Facilities
        num_zip = self.num_exp_Customers
        initial_demand = 15000

        # get maximum{0, 1 - distance/30} for all distances
        max_distance_factor = np.maximum(1 - self.experiments_distance / 30, 0)

        # 创建模型
        m = gp.Model('nonparametric_dep')
        m.setParam('MIPGap', 0.01)
        m.setParam('OutputFlag', 0)
        # set time limit to 1 hrs
        m.setParam('TimeLimit', 7200)
        # set dual reductions to 0
        m.setParam('DualReductions', 0)


        # 决策变量
        charger = m.addVars(num_potential, vtype=GRB.INTEGER, ub=10, lb=0, name='Charger')
        location = m.addVars(num_potential, vtype=GRB.BINARY, name='Location')
        assign = m.addVars(num_potential, num_zip, sce, lb=0.0, vtype=GRB.CONTINUOUS, name='Assign')
        unmetdemand = m.addVars(num_zip, sce, lb=0.0, vtype=GRB.CONTINUOUS, name='Unmet')
        real_demand = m.addVars(num_zip, sce, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='real_demand')
        projected_demand = m.addVars(num_zip, sce, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='demand')
        estimated_demand = m.addVars(num_zip, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS,
                                    name='increase_demand')
        h_term = m.addVars(num_zip, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='h_term')

        # 基础约束
        m.addConstrs((charger[i] <= 10 * location[i] for i in range(num_potential)), name='Location')
        m.addConstrs(
            (gp.quicksum(assign[(i, j, k)] for i in range(num_potential)) + unmetdemand[(j, k)] == real_demand[(j, k)]
             for j in range(num_zip) for k in range(sce)), name='Demand')
        m.addConstrs(
            (gp.quicksum(assign[(i, j, k)] for j in range(num_zip)) <= C[i] * charger[i]
             for i in range(num_potential) for k in range(sce)), name='Capacity')

        # regression_data.loc[group_data.index, 'sum_capacity'] = \
        #     group_data['capacity'].cumsum() - first_capacity
        #
        # regression_data.loc[group_data.index, 'exp_distance_capacity'] = \
        #     (np.exp(-group_data['distance']) * group_data['capacity']).cumsum() - \
        #     (np.exp(-first_distance) * first_capacity)
        #
        # regression_data.loc[group_data.index, 'step1_exp_distance_capacity'] = \
        #     (np.exp(self.step1_intercept + self.step1_coef * group_data['distance']) * group_data[
        #         'capacity']).cumsum() - \
        #     (np.exp(self.step1_intercept + self.step1_coef * first_distance) * first_capacity)
        #
        # regression_data.loc[group_data.index, 'max_distance_capacity']

        if feature_type == 'H1':
            # H1: sum_capacity
            m.addConstrs(
                (h_term[j] == gp.quicksum(charger[i] for i in range(num_potential)) for j in range(num_zip)),
                name='H1_Term'
            )
            for j in range(num_zip):
                # ✅ 正确的 DataFrame 创建方式
                feats = pd.DataFrame({
                    'normalized_education_level': [self.education[j]],
                    'normalized_gdp': [self.gdp[j]],
                    'sum_capacity': [h_term[j]]
                })

                # 添加预测约束
                pred_constr = add_predictor_constr(
                    gp_model=m,
                    predictor=trained_model,
                    input_vars=feats,  # DataFrame 格式
                    output_vars=estimated_demand[j],
                    epsilon=1e-5
                )
        elif feature_type == 'H2':
            # H2: exp_distance_capacity
            m.addConstrs(
                (h_term[j] == gp.quicksum(
                    np.exp(-self.experiments_distance.iloc[i, j]) * charger[i]
                    for i in range(num_potential)) for j in range(num_zip)
                 ),
                name='H2_Term'
            )
            for j in range(num_zip):
                # ✅ 正确的 DataFrame 创建方式
                feats = pd.DataFrame({
                    'normalized_education_level': [self.education[j]],
                    'normalized_gdp': [self.gdp[j]],
                    'exp_distance_capacity': [h_term[j]]
                })

                # 添加预测约束
                pred_constr = add_predictor_constr(
                    gp_model=m,
                    predictor=trained_model,
                    input_vars=feats,  # DataFrame 格式
                    output_vars=estimated_demand[j],
                    epsilon=1e-5
                )
        elif feature_type == 'H4':
            # H3: step1_exp_distance_capacity
            m.addConstrs(
                (h_term[j] == gp.quicksum(
                    np.exp(- self.experiments_distance.iloc[i, j]/25) * charger[i]
                    for i in range(num_potential)) for j in range(num_zip)
                 ),
                name='H3_Term'
            )
            for j in range(num_zip):
                # ✅ 正确的 DataFrame 创建方式
                feats = pd.DataFrame({
                    'normalized_education_level': [self.education[j]],
                    'normalized_gdp': [self.gdp[j]],
                    'assumption_exp_distance_capacity': [h_term[j]]
                })

                # 添加预测约束
                pred_constr = add_predictor_constr(
                    gp_model=m,
                    predictor=trained_model,
                    input_vars=feats,  # DataFrame 格式
                    output_vars=estimated_demand[j],
                    epsilon=1e-5
                )
        elif feature_type == 'H4':
            # H3: step1_exp_distance_capacity
            m.addConstrs(
                (h_term[j] == gp.quicksum(
                    np.exp(self.step1_coef[0] * self.experiments_distance.iloc[i, j] + self.step1_intercept) * charger[i]
                    for i in range(num_potential)) for j in range(num_zip)
                 ),
                name='H3_Term'
            )
            for j in range(num_zip):
                # ✅ 正确的 DataFrame 创建方式
                feats = pd.DataFrame({
                    'normalized_education_level': [self.education[j]],
                    'normalized_gdp': [self.gdp[j]],
                    'step1_exp_distance_capacity': [h_term[j]]
                })

                # 添加预测约束
                pred_constr = add_predictor_constr(
                    gp_model=m,
                    predictor=trained_model,
                    input_vars=feats,  # DataFrame 格式
                    output_vars=estimated_demand[j],
                    epsilon=1e-5
                )
        # elif feature_type == 'H4':
        #     # H4: max_distance_capacity
        #     m.addConstrs(
        #         (h_term[j] == gp.quicksum(
        #             max_distance_factor.iloc[i, j] * charger[i]
        #             for i in range(num_potential))
        #             for j in range(num_zip)
        #          ),
        #         name='H4_Term'
        #     )
        #     for j in range(num_zip):
        #         # ✅ 正确的 DataFrame 创建方式
        #         feats = pd.DataFrame({
        #             'normalized_education_level': [self.education[j]],
        #             'normalized_gdp': [self.gdp[j]],
        #             'max_distance_capacity': [h_term[j]]
        #         })
        #
        #         # 添加预测约束
        #         pred_constr = add_predictor_constr(
        #             gp_model=m,
        #             predictor=trained_model,
        #             input_vars=feats,  # DataFrame 格式
        #             output_vars=estimated_demand[j],
        #             epsilon=1e-5
        #         )
        else:
            pass


        # 需求约束
        m.addConstrs(
            (real_demand[(j, k)] == initial_demand +
             estimated_demand[j] +
             residuals[k]
             for j in range(num_zip) for k in range(sce)),
            name='Real_Demand')

        # max(real_demand, 0)
        m.addConstrs(
            (projected_demand[(j, k)] == gp.max_([real_demand[(j, k)], 0.0]) for j in range(num_zip) for k in range(sce)),
            name='Projected_Demand_Positive'
        )

        # 目标函数
        fixed_cost = location.prod(f)
        charger_cost = charger.prod(chargercost)
        transportation_cost = gp.quicksum(
            assign[(i, j, k)] * c.iloc[i, j] * prob[k]
            for i in range(num_potential) for j in range(num_zip) for k in range(sce))
        penalty_cost = gp.quicksum(
            unmetdemand[j, k] * p[j] * prob[k]
            for j in range(num_zip) for k in range(sce))
        revenue = gp.quicksum(
            r[j] * projected_demand[(j, k)] * prob[k]
            for j in range(num_zip) for k in range(sce))

        m.setObjective(fixed_cost + charger_cost + transportation_cost + penalty_cost - revenue, GRB.MINIMIZE)
        m.optimize()
        m.write("insample_dep.lp")

        # grb.optimal or time limit reached with feasible solution
        if m.status == GRB.OPTIMAL or (m.status == GRB.TIME_LIMIT and m.SolCount > 0):
            # Extract decision variable values
            location_values = [location[i].X for i in range(num_potential)]
            charger_values = [charger[i].X for i in range(num_potential)]
            optimal_value = m.objVal
            revenue_value = sum(
                r[j] * projected_demand[(j, k)].X * prob[k]
                for j in range(num_zip) for k in range(sce))

            # 计算实际需求值
            real_demand_values = []
            for j in range(num_zip):
                avg_demand = sum(real_demand[(j, k)].X for k in range(sce)) / sce
                real_demand_values.append(avg_demand)

            self.dep_decision = charger_values

            print(f'  Location: {location_values}')
            print(f'  Charger: {charger_values}')
            print(f'  Fixed cost: {sum(location[i].X * f[i] for i in range(num_potential)):.2f}')
            print(f'  Charger cost: {sum(charger[i].X * chargercost[i] for i in range(num_potential)):.2f}')
            print(f'  Penalty cost: {penalty_cost.getValue():.2f}')
            print(f'  Transportation cost: {transportation_cost.getValue():.2f}')
            print(f'  Revenue: {revenue.getValue():.2f}')
            print(f'  Objective: {optimal_value:.2f}')

            return charger_values, optimal_value, revenue_value, real_demand_values
        else:
            print(f'  ⚠️ Optimization status: {m.status}')
            return None, None, None, None

    def insample_ml_sample(self, model_name, train_data, feature_type, features, sample_size=1000):
        """
        Use trained ML model to make decisions via optimization
        """
        # Get the trained model

        # 准备训练数据
        response = train_data['demand'] - train_data['initial_demand']
        x = train_data[features].values
        y = response.values

        # 训练模型
        model_key = f"{model_name}_{feature_type}"
        if model_key not in self.best_models_dep:
            raise ValueError(f"Model {model_name} not found in trained models")

        trained_model = self.best_models_dep[model_key]
        trained_model.fit(x, y)

        # 计算残差
        residuals = y - trained_model.predict(x)
        if len(residuals) > sample_size:
            rng = np.random.default_rng(seed=42)
            sampled_indices = rng.choice(len(residuals), size=sample_size, replace=False)
            residuals = residuals[sampled_indices]
        sce = len(residuals)
        prob = [1 / sce] * sce

        # 优化参数
        r = self.revenue
        f = self.fixedcost
        c = self.experiments_distance * self.transportationcost
        chargercost = self.chargercost
        C = self.capacity
        p = self.penaltycost
        num_potential = self.num_exp_Facilities
        num_zip = self.num_exp_Customers
        initial_demand = 15000

        # get maximum{0, 1 - distance/30} for all distances
        max_distance_factor = np.maximum(1 - self.experiments_distance / 30, 0)

        # 创建模型
        m = gp.Model('nonparametric_dep')
        m.setParam('MIPGap', 0.01)
        m.setParam('OutputFlag', 0)
        # set time limit to 1 hrs
        m.setParam('TimeLimit', 3600)
        # set dual reductions to 0
        m.setParam('DualReductions', 0)


        # 决策变量
        charger = m.addVars(num_potential, vtype=GRB.INTEGER, ub=10, lb=0, name='Charger')
        location = m.addVars(num_potential, vtype=GRB.BINARY, name='Location')
        assign = m.addVars(num_potential, num_zip, sce, lb=0.0, vtype=GRB.CONTINUOUS, name='Assign')
        unmetdemand = m.addVars(num_zip, sce, lb=0.0, vtype=GRB.CONTINUOUS, name='Unmet')
        real_demand = m.addVars(num_zip, sce, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='real_demand')
        projected_demand = m.addVars(num_zip, sce, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='demand')
        estimated_demand = m.addVars(num_zip, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS,
                                    name='increase_demand')
        h_term = m.addVars(num_zip, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='h_term')

        # 基础约束
        m.addConstrs((charger[i] <= 10 * location[i] for i in range(num_potential)), name='Location')
        m.addConstrs(
            (gp.quicksum(assign[(i, j, k)] for i in range(num_potential)) + unmetdemand[(j, k)] == real_demand[(j, k)]
             for j in range(num_zip) for k in range(sce)), name='Demand')
        m.addConstrs(
            (gp.quicksum(assign[(i, j, k)] for j in range(num_zip)) <= C[i] * charger[i]
             for i in range(num_potential) for k in range(sce)), name='Capacity')

        # regression_data.loc[group_data.index, 'sum_capacity'] = \
        #     group_data['capacity'].cumsum() - first_capacity
        #
        # regression_data.loc[group_data.index, 'exp_distance_capacity'] = \
        #     (np.exp(-group_data['distance']) * group_data['capacity']).cumsum() - \
        #     (np.exp(-first_distance) * first_capacity)
        #
        # regression_data.loc[group_data.index, 'step1_exp_distance_capacity'] = \
        #     (np.exp(self.step1_intercept + self.step1_coef * group_data['distance']) * group_data[
        #         'capacity']).cumsum() - \
        #     (np.exp(self.step1_intercept + self.step1_coef * first_distance) * first_capacity)
        #
        # regression_data.loc[group_data.index, 'max_distance_capacity']

        if feature_type == 'H1':
            # H1: sum_capacity
            m.addConstrs(
                (h_term[j] == gp.quicksum(charger[i] for i in range(num_potential)) for j in range(num_zip)),
                name='H1_Term'
            )
            for j in range(num_zip):
                # ✅ 正确的 DataFrame 创建方式
                feats = pd.DataFrame({
                    'normalized_education_level': [self.education[j]],
                    'normalized_gdp': [self.gdp[j]],
                    'sum_capacity': [h_term[j]]
                })

                # 添加预测约束
                pred_constr = add_predictor_constr(
                    gp_model=m,
                    predictor=trained_model,
                    input_vars=feats,  # DataFrame 格式
                    output_vars=estimated_demand[j],
                    epsilon=1e-5
                )
        elif feature_type == 'H2':
            # H2: exp_distance_capacity
            m.addConstrs(
                (h_term[j] == gp.quicksum(
                    np.exp(-self.experiments_distance.iloc[i, j]) * charger[i]
                    for i in range(num_potential)) for j in range(num_zip)
                 ),
                name='H2_Term'
            )
            for j in range(num_zip):
                # ✅ 正确的 DataFrame 创建方式
                feats = pd.DataFrame({
                    'normalized_education_level': [self.education[j]],
                    'normalized_gdp': [self.gdp[j]],
                    'exp_distance_capacity': [h_term[j]]
                })

                # 添加预测约束
                pred_constr = add_predictor_constr(
                    gp_model=m,
                    predictor=trained_model,
                    input_vars=feats,  # DataFrame 格式
                    output_vars=estimated_demand[j],
                    epsilon=1e-5
                )
        elif feature_type == 'H4':
            # H3: step1_exp_distance_capacity
            m.addConstrs(
                (h_term[j] == gp.quicksum(
                    np.exp(- self.experiments_distance.iloc[i, j]/25) * charger[i]
                    for i in range(num_potential)) for j in range(num_zip)
                 ),
                name='H3_Term'
            )
            for j in range(num_zip):
                # ✅ 正确的 DataFrame 创建方式
                feats = pd.DataFrame({
                    'normalized_education_level': [self.education[j]],
                    'normalized_gdp': [self.gdp[j]],
                    'assumption_exp_distance_capacity': [h_term[j]]
                })

                # 添加预测约束
                pred_constr = add_predictor_constr(
                    gp_model=m,
                    predictor=trained_model,
                    input_vars=feats,  # DataFrame 格式
                    output_vars=estimated_demand[j],
                    epsilon=1e-5
                )
        elif feature_type == 'H4':
            # H3: step1_exp_distance_capacity
            m.addConstrs(
                (h_term[j] == gp.quicksum(
                    np.exp(self.step1_coef[0] * self.experiments_distance.iloc[i, j] + self.step1_intercept) * charger[i]
                    for i in range(num_potential)) for j in range(num_zip)
                 ),
                name='H3_Term'
            )
            for j in range(num_zip):
                # ✅ 正确的 DataFrame 创建方式
                feats = pd.DataFrame({
                    'normalized_education_level': [self.education[j]],
                    'normalized_gdp': [self.gdp[j]],
                    'step1_exp_distance_capacity': [h_term[j]]
                })

                # 添加预测约束
                pred_constr = add_predictor_constr(
                    gp_model=m,
                    predictor=trained_model,
                    input_vars=feats,  # DataFrame 格式
                    output_vars=estimated_demand[j],
                    epsilon=1e-5
                )
        # elif feature_type == 'H4':
        #     # H4: max_distance_capacity
        #     m.addConstrs(
        #         (h_term[j] == gp.quicksum(
        #             max_distance_factor.iloc[i, j] * charger[i]
        #             for i in range(num_potential))
        #             for j in range(num_zip)
        #          ),
        #         name='H4_Term'
        #     )
        #     for j in range(num_zip):
        #         # ✅ 正确的 DataFrame 创建方式
        #         feats = pd.DataFrame({
        #             'normalized_education_level': [self.education[j]],
        #             'normalized_gdp': [self.gdp[j]],
        #             'max_distance_capacity': [h_term[j]]
        #         })
        #
        #         # 添加预测约束
        #         pred_constr = add_predictor_constr(
        #             gp_model=m,
        #             predictor=trained_model,
        #             input_vars=feats,  # DataFrame 格式
        #             output_vars=estimated_demand[j],
        #             epsilon=1e-5
        #         )
        else:
            pass


        # 需求约束
        m.addConstrs(
            (real_demand[(j, k)] == initial_demand +
             estimated_demand[j] +
             residuals[k]
             for j in range(num_zip) for k in range(sce)),
            name='Real_Demand')

        # max(real_demand, 0)
        m.addConstrs(
            (projected_demand[(j, k)] == gp.max_([real_demand[(j, k)], 0.0]) for j in range(num_zip) for k in range(sce)),
            name='Projected_Demand_Positive'
        )

        # 目标函数
        fixed_cost = location.prod(f)
        charger_cost = charger.prod(chargercost)
        transportation_cost = gp.quicksum(
            assign[(i, j, k)] * c.iloc[i, j] * prob[k]
            for i in range(num_potential) for j in range(num_zip) for k in range(sce))
        penalty_cost = gp.quicksum(
            unmetdemand[j, k] * p[j] * prob[k]
            for j in range(num_zip) for k in range(sce))
        revenue = gp.quicksum(
            r[j] * projected_demand[(j, k)] * prob[k]
            for j in range(num_zip) for k in range(sce))

        m.setObjective(fixed_cost + charger_cost + transportation_cost + penalty_cost - revenue, GRB.MINIMIZE)
        m.optimize()
        m.write("insample_dep.lp")

        # grb.optimal or time limit reached with feasible solution
        if m.status == GRB.OPTIMAL or (m.status == GRB.TIME_LIMIT and m.SolCount > 0):
            # Extract decision variable values
            location_values = [location[i].X for i in range(num_potential)]
            charger_values = [charger[i].X for i in range(num_potential)]
            optimal_value = m.objVal
            revenue_value = sum(
                r[j] * projected_demand[(j, k)].X * prob[k]
                for j in range(num_zip) for k in range(sce))

            # 计算实际需求值
            real_demand_values = []
            for j in range(num_zip):
                avg_demand = sum(real_demand[(j, k)].X for k in range(sce)) / sce
                real_demand_values.append(avg_demand)

            self.dep_decision = charger_values

            print(f'  Location: {location_values}')
            print(f'  Charger: {charger_values}')
            print(f'  Fixed cost: {sum(location[i].X * f[i] for i in range(num_potential)):.2f}')
            print(f'  Charger cost: {sum(charger[i].X * chargercost[i] for i in range(num_potential)):.2f}')
            print(f'  Penalty cost: {penalty_cost.getValue():.2f}')
            print(f'  Transportation cost: {transportation_cost.getValue():.2f}')
            print(f'  Revenue: {revenue.getValue():.2f}')
            print(f'  Objective: {optimal_value:.2f}')
            # get mipgap
            mipgap = m.MIPGap


            return charger_values, optimal_value, revenue_value, real_demand_values, mipgap, m.status
        else:
            print(f'  ⚠️ Optimization status: {m.status}')
            return None, None, None, None, None, m.status

    def oos_cost(self, decision):
        """
        Evaluate out-of-sample cost given a decision
        """
        # Get necessary parameters from class attributes
        r = self.revenue
        f = self.fixedcost
        c = self.experiments_distance * self.transportationcost
        chargercost = self.chargercost
        C = self.capacity
        p = self.penaltycost
        num_potential = self.num_exp_Facilities
        num_zip = self.num_exp_Customers

        m = gp.Model('oos_evaluation')
        m.setParam('MIPGap', 0.01)
        m.setParam('OutputFlag', 0)

        noise = self.test_noise

        # Reuse exponential term from step1
        exponential_term = np.exp(
            self.dep_alpha1 + self.experiments_distance.to_numpy() * self.dep_beta_distance)

        # Repeat the decision list for test_sizeData rows
        decision_df = pd.DataFrame([decision] * self.test_sizeData)

        print(f"decision_df shape: {decision_df.to_numpy().shape}")
        print(f"exponential_term shape: {exponential_term.shape}")
        print(f"experiments_distance shape: {self.experiments_distance.shape}")

        demand_contrib = decision_df.to_numpy() @ exponential_term

        testDemand = np.zeros((self.test_sizeData, num_zip))
        for j in range(num_zip):
            testDemand[:, j] = (
                    demand_contrib[:, j] + 15000 +
                    self.education[j] * self.dep_beta_education +
                    self.gdp[j] * self.dep_beta_gdp +
                    self.dep_intercept + noise[:, j]
            )

        self.dep_testDemand_deterministic = testDemand

        sce = self.test_sizeData
        prob = np.array([1.0 / sce] * sce)

        # Variables
        charger = m.addVars(num_potential, vtype=GRB.INTEGER, ub=10, name='Charger')
        location = m.addVars(num_potential, vtype=GRB.BINARY, name='Location')
        assign = m.addVars(num_potential, num_zip, sce, lb=0.0, vtype=GRB.CONTINUOUS, name='Assign')
        unmetdemand = m.addVars(num_zip, sce, lb=0.0, vtype=GRB.CONTINUOUS, name='Unmet')

        # Constraints
        m.addConstrs((10 * location[i] >= charger[i] for i in range(num_potential)), name='Decision')
        m.addConstrs((charger[i] == decision[i] for i in range(num_potential)), name='Location')
        m.addConstrs(
            (gp.quicksum(assign[(i, j, k)] for i in range(num_potential)) + unmetdemand[(j, k)] == testDemand[k, j]
             for j in range(num_zip) for k in range(sce)), name='Demand')
        m.addConstrs(
            (gp.quicksum(assign[(i, j, k)] for j in range(num_zip)) <= C[i] * charger[i]
             for i in range(num_potential) for k in range(sce)), name='Capacity')

        # Objective function components
        fixed_cost = location.prod(f)
        charger_cost = charger.prod(chargercost)
        transportation_cost = gp.quicksum(
            assign[(i, j, k)] * c.iloc[i, j] * prob[k]
            for i in range(num_potential) for j in range(num_zip) for k in range(sce))
        penalty_cost = gp.quicksum(
            unmetdemand[j, k] * p[j] * prob[k]
            for j in range(num_zip) for k in range(sce))
        revenue = gp.quicksum(
            r[j] * testDemand[k, j] * prob[k]
            for j in range(num_zip) for k in range(sce))

        # Set the objective function
        m.setObjective(
            fixed_cost + charger_cost + transportation_cost + penalty_cost - revenue,
            GRB.MINIMIZE)

        m.optimize()

        if m.status == GRB.OPTIMAL:
            location_values = [charger[i].X for i in range(num_potential)]
            fixed_cost_val = sum(location[i].X * f[i] for i in range(num_potential))
            charger_cost_val = sum(charger[i].X * chargercost[i] for i in range(num_potential))
            optimal_value = m.objVal
            revenue_value = sum(
                r[j] * testDemand[k, j] * prob[k]
                for j in range(num_zip) for k in range(sce))
            penalty_cost_val = sum(
                unmetdemand[j, k].X * p[j] * prob[k]
                for j in range(num_zip) for k in range(sce))
            transportation_cost_val = sum(
                assign[(i, j, k)].X * c.iloc[i, j] * prob[k]
                for i in range(num_potential) for j in range(num_zip) for k in range(sce))

            print(f'Fixed cost (OOS): {fixed_cost_val:.2f}')
            print(f'Charger cost (OOS): {charger_cost_val:.2f}')
            print(f'Penalty cost (OOS): {penalty_cost_val:.2f}')
            print(f'Transportation cost (OOS): {transportation_cost_val:.2f}')

            return location_values, optimal_value, revenue_value, testDemand
        else:
            print(f"OOS optimization failed with status: {m.status}")
            return None, None, None, None

    # def insample_experiments_sample_size(self, sample_size=1000):
    #     # sample 1000 residuals
    #     rng = np.random.default_rng(seed=42)
    #     sampled_indices = rng.choice(len(self.step2_dep_residuals), size=sample_size, replace=False)
    #     sampled_step2_dep_residuals = [self.step2_dep_residuals[i] for i in sampled_indices]
    #
    #     # Initialize insample_results for two-step and different models
    #     self.initialize_for_optimization()
    #     start_time = time.time()
    #     twostep_location_values, twostep_optimal_value, twostep_revenue_value, twostep_real_demand_values = self.insample_two_step(
    #         self.step2_dep_residuals)
    #     twostep_runtime = time.time() - start_time
    #     twostep_location_values_oos, twostep_optimal_value_oos, twostep_revenue_value_oos, twostep_real_demand_values_oos = self.oos_cost(
    #         decision=twostep_location_values)
    #
    #     twostep_insample_results = {
    #         'model': 'Two-Step',
    #         'feature_type': 'N/A',
    #         'location_values': twostep_location_values,
    #         'optimal_value': twostep_optimal_value,
    #         'revenue_value': twostep_revenue_value,
    #         'real_demand_values': twostep_real_demand_values,
    #         'runtime': twostep_runtime
    #     }
    #
    #     twostep_oos_results = {
    #         'model': 'Two-Step',
    #         'feature_type': 'N/A',
    #         'location_values': twostep_location_values_oos,
    #         'optimal_value': twostep_optimal_value_oos,
    #         'revenue_value': twostep_revenue_value_oos,
    #         'real_demand_values': twostep_real_demand_values_oos
    #     }
    #
    #     # ✅ 创建 two-step 的 DataFrame
    #     twostep_df = pd.DataFrame([twostep_insample_results])
    #     twostep_oos_df = pd.DataFrame([twostep_oos_results])
    #
    #     regression_models = ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'Neural Network']
    #     # train_ml_data = self.train_ml
    #     # test_ml_data = self.test_ml
    #     train_ml_data = self.training_data
    #     test_ml_data = self.testing_data
    #
    #     # ✅ 初始化空的 DataFrame
    #     ml_df = pd.DataFrame()
    #     ml_df_oos = pd.DataFrame()
    #
    #     # feature_list = [
    #     #     ['normalized_education_level', 'normalized_gdp', 'sum_capacity'],
    #     #     ['normalized_education_level', 'normalized_gdp', 'exp_distance_capacity'],
    #     #     ['normalized_education_level', 'normalized_gdp', 'step1_exp_distance_capacity'],
    #     #     ['normalized_education_level', 'normalized_gdp', 'max_distance_capacity']
    #     # ]
    #
    #     feature_list = [
    #         ['normalized_education_level', 'normalized_gdp', 'sum_capacity'],
    #         ['normalized_education_level', 'normalized_gdp', 'exp_distance_capacity'],
    #         ['normalized_education_level', 'normalized_gdp', 'assumption_exp_distance_capacity'],
    #         ['normalized_education_level', 'normalized_gdp', 'step1_exp_distance_capacity']
    #     ]
    #
    #     # Map feature_list to 4 feature type H1, H2, H3, H4
    #     feature_types = ['H1', 'H2', 'H3', 'H4']
    #
    #     print(f"\n{'=' * 80}")
    #     print("RUNNING INSAMPLE EXPERIMENTS FOR ALL MODEL-FEATURE COMBINATIONS")
    #     print(f"{'=' * 80}")
    #
    #     # ✅ Loop through all 4 feature versions
    #     for model_name in regression_models:
    #         for idx, features_col in enumerate(feature_list):
    #             f = feature_types[idx]
    #             print(f"\nRunning {model_name} with {f} features: {features_col}")
    #
    #             # Time each model-feature combination
    #             model_start_time = time.time()
    #             ml_location_values, ml_optimal_value, ml_revenue_value, ml_real_demand_values = self.insample_ml(
    #                 model_name, train_ml_data, feature_type=f, features=features_col)
    #             model_runtime = time.time() - model_start_time
    #
    #             # ✅ 直接创建当前结果的 DataFrame
    #             current_result = pd.DataFrame([{
    #                 'model': model_name,
    #                 'feature_type': f,
    #                 'location_values': ml_location_values,
    #                 'optimal_value': ml_optimal_value,
    #                 'revenue_value': ml_revenue_value,
    #                 'real_demand_values': ml_real_demand_values,
    #                 'runtime': model_runtime
    #             }])
    #
    #             # ✅ 直接 concat 到 ml_df
    #             ml_df = pd.concat([ml_df, current_result], ignore_index=True)
    #
    #             # if ml_location_values exists then go with oos
    #             if ml_location_values is None:
    #                 print("  ⚠️ Skipping OOS evaluation due to failed in-sample optimization.")
    #                 current_oos_result = pd.DataFrame([{
    #                     'model': model_name,
    #                     'feature_type': f,
    #                     'location_values': None,
    #                     'optimal_value': None,
    #                     'revenue_value': None,
    #                     'real_demand_values': None
    #                 }])
    #             else:
    #                 # Calculate OOS results
    #                 ml_location_values_oos, ml_optimal_value_oos, ml_revenue_value_oos, ml_real_demand_values_oos = self.oos_cost(
    #                     decision=ml_location_values)
    #
    #                 # ✅ 直接创建当前 OOS 结果的 DataFrame
    #                 current_oos_result = pd.DataFrame([{
    #                     'model': model_name,
    #                     'feature_type': f,
    #                     'location_values': ml_location_values_oos,
    #                     'optimal_value': ml_optimal_value_oos,
    #                     'revenue_value': ml_revenue_value_oos,
    #                     'real_demand_values': ml_real_demand_values_oos
    #                 }])
    #
    #                 # ✅ 直接 concat 到 ml_df_oos
    #                 ml_df_oos = pd.concat([ml_df_oos, current_oos_result], ignore_index=True)
    #
    #                 print(f"  Runtime: {model_runtime:.2f}s")
    #                 print(f"  In-sample Optimal Value: {ml_optimal_value:.2f}")
    #                 print(f"  Out-of-sample Optimal Value: {ml_optimal_value_oos:.2f}")
    #
    #     print(f"\n{'=' * 80}")
    #     print("INSAMPLE EXPERIMENTS SUMMARY")
    #     print(f"{'=' * 80}")
    #     print(f"Two-Step Model:")
    #     print(f"  Runtime: {twostep_runtime:.2f}s")
    #     print(f"  In-sample Optimal Value: {twostep_optimal_value:.2f}")
    #     print(f"  Out-of-sample Optimal Value: {twostep_optimal_value_oos:.2f}")
    #     print(f"\nML Models:")
    #     print(f"  Total model-feature combinations: {len(ml_df)}")
    #     print(f"  In-sample results shape: {ml_df.shape}")
    #     print(f"  Out-of-sample results shape: {ml_df_oos.shape}")
    #
    #     # Show best performing model-feature combination
    #     if len(ml_df) > 0:
    #         best_insample_idx = ml_df['optimal_value'].idxmin()
    #         best_oos_idx = ml_df_oos['optimal_value'].idxmin()
    #         print(
    #             f"\nBest In-sample: {ml_df.loc[best_insample_idx, 'model']} with {ml_df.loc[best_insample_idx, 'feature_type']} (Value: {ml_df.loc[best_insample_idx, 'optimal_value']:.2f})")
    #         print(
    #             f"Best Out-of-sample: {ml_df_oos.loc[best_oos_idx, 'model']} with {ml_df_oos.loc[best_oos_idx, 'feature_type']} (Value: {ml_df_oos.loc[best_oos_idx, 'optimal_value']:.2f})")
    #
    #     return twostep_df, twostep_oos_df, ml_df, ml_df_oos

    def insample_experiments_sample_size(self, sample_size=1000):
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        base_dir = "/Users/huangrongsun/Library/CloudStorage/OneDrive-TheOhioStateUniversity/EVChargingLocationWin"
        output_dir = os.path.join(base_dir, "nonparametric_results", date_time)

        os.makedirs(output_dir, exist_ok=True)

        # sample 1000 residuals
        rng = np.random.default_rng(seed=42)
        sampled_indices = rng.choice(len(self.step2_dep_residuals), size=sample_size, replace=False)
        sampled_step2_dep_residuals = [self.step2_dep_residuals[i] for i in sampled_indices]

        # Initialize insample_results for two-step and different models
        self.initialize_for_optimization()
        start_time = time.time()
        twostep_location_values, twostep_optimal_value, twostep_revenue_value, twostep_real_demand_values, mipgap, status = self.insample_two_step(
            sampled_step2_dep_residuals)
        twostep_runtime = time.time() - start_time
        twostep_location_values_oos, twostep_optimal_value_oos, twostep_revenue_value_oos, twostep_real_demand_values_oos = self.oos_cost(
            decision=twostep_location_values)

        twostep_insample_results = {
            'model': 'Two-Step',
            'feature_type': 'N/A',
            'location_values': twostep_location_values,
            'optimal_value': twostep_optimal_value,
            'revenue_value': twostep_revenue_value,
            'real_demand_values': twostep_real_demand_values,
            'runtime': twostep_runtime
        }

        twostep_oos_results = {
            'model': 'Two-Step',
            'feature_type': 'N/A',
            'location_values': twostep_location_values_oos,
            'optimal_value': twostep_optimal_value_oos,
            'revenue_value': twostep_revenue_value_oos,
            'real_demand_values': twostep_real_demand_values_oos
        }

        # ✅ 创建 two-step 的 DataFrame
        twostep_df = pd.DataFrame([twostep_insample_results])
        twostep_oos_df = pd.DataFrame([twostep_oos_results])

        regression_models = ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'Neural Network']
        # train_ml_data = self.train_ml
        # test_ml_data = self.test_ml
        train_ml_data = self.training_data
        test_ml_data = self.testing_data

        # ✅ 初始化空的 DataFrame
        ml_df = pd.DataFrame()
        ml_df_oos = pd.DataFrame()

        # feature_list = [
        #     ['normalized_education_level', 'normalized_gdp', 'sum_capacity'],
        #     ['normalized_education_level', 'normalized_gdp', 'exp_distance_capacity'],
        #     ['normalized_education_level', 'normalized_gdp', 'step1_exp_distance_capacity'],
        #     ['normalized_education_level', 'normalized_gdp', 'max_distance_capacity']
        # ]

        feature_list = [
            ['normalized_education_level', 'normalized_gdp', 'sum_capacity'],
            ['normalized_education_level', 'normalized_gdp', 'exp_distance_capacity'],
            ['normalized_education_level', 'normalized_gdp', 'assumption_exp_distance_capacity'],
            ['normalized_education_level', 'normalized_gdp', 'step1_exp_distance_capacity']
        ]

        # Map feature_list to 4 feature type H1, H2, H3, H4
        feature_types = ['H1', 'H2', 'H3', 'H4']

        print(f"\n{'=' * 80}")
        print("RUNNING INSAMPLE EXPERIMENTS FOR ALL MODEL-FEATURE COMBINATIONS")
        print(f"{'=' * 80}")

        # ✅ Loop through all 4 feature versions
        for model_name in regression_models:
            for idx, features_col in enumerate(feature_list):
                f = feature_types[idx]
                print(f"\nRunning {model_name} with {f} features: {features_col}")

                # Time each model-feature combination
                model_start_time = time.time()
                ml_location_values, ml_optimal_value, ml_revenue_value, ml_real_demand_values = self.insample_ml_sample(
                    model_name, train_ml_data, feature_type=f, features=features_col)
                model_runtime = time.time() - model_start_time

                # ✅ 直接创建当前结果的 DataFrame
                current_result = pd.DataFrame([{
                    'model': model_name,
                    'feature_type': f,
                    'location_values': ml_location_values,
                    'optimal_value': ml_optimal_value,
                    'revenue_value': ml_revenue_value,
                    'real_demand_values': ml_real_demand_values,
                    'runtime': model_runtime
                }])

                # ✅ 直接 concat 到 ml_df
                ml_df = pd.concat([ml_df, current_result], ignore_index=True)

                # if ml_location_values exists then go with oos
                if ml_location_values is None:
                    print("  ⚠️ Skipping OOS evaluation due to failed in-sample optimization.")
                    current_oos_result = pd.DataFrame([{
                        'model': model_name,
                        'feature_type': f,
                        'location_values': None,
                        'optimal_value': None,
                        'revenue_value': None,
                        'real_demand_values': None
                    }])
                else:
                    # Calculate OOS results
                    ml_location_values_oos, ml_optimal_value_oos, ml_revenue_value_oos, ml_real_demand_values_oos = self.oos_cost(
                        decision=ml_location_values)

                    # ✅ 直接创建当前 OOS 结果的 DataFrame
                    current_oos_result = pd.DataFrame([{
                        'model': model_name,
                        'feature_type': f,
                        'location_values': ml_location_values_oos,
                        'optimal_value': ml_optimal_value_oos,
                        'revenue_value': ml_revenue_value_oos,
                        'real_demand_values': ml_real_demand_values_oos
                    }])

                    # ✅ 直接 concat 到 ml_df_oos
                    ml_df_oos = pd.concat([ml_df_oos, current_oos_result], ignore_index=True)

                    print(f"  Runtime: {model_runtime:.2f}s")
                    print(f"  In-sample Optimal Value: {ml_optimal_value:.2f}")
                    print(f"  Out-of-sample Optimal Value: {ml_optimal_value_oos:.2f}")

                    # Save intermediate results to CSV

        print(f"\n{'=' * 80}")
        print("INSAMPLE EXPERIMENTS SUMMARY")
        print(f"{'=' * 80}")
        print(f"Two-Step Model:")
        print(f"  Runtime: {twostep_runtime:.2f}s")
        print(f"  In-sample Optimal Value: {twostep_optimal_value:.2f}")
        print(f"  Out-of-sample Optimal Value: {twostep_optimal_value_oos:.2f}")
        print(f"\nML Models:")
        print(f"  Total model-feature combinations: {len(ml_df)}")
        print(f"  In-sample results shape: {ml_df.shape}")
        print(f"  Out-of-sample results shape: {ml_df_oos.shape}")

        # Show best performing model-feature combination
        if len(ml_df) > 0:
            best_insample_idx = ml_df['optimal_value'].idxmin()
            best_oos_idx = ml_df_oos['optimal_value'].idxmin()
            print(
                f"\nBest In-sample: {ml_df.loc[best_insample_idx, 'model']} with {ml_df.loc[best_insample_idx, 'feature_type']} (Value: {ml_df.loc[best_insample_idx, 'optimal_value']:.2f})")
            print(
                f"Best Out-of-sample: {ml_df_oos.loc[best_oos_idx, 'model']} with {ml_df_oos.loc[best_oos_idx, 'feature_type']} (Value: {ml_df_oos.loc[best_oos_idx, 'optimal_value']:.2f})")

        return twostep_df, twostep_oos_df, ml_df, ml_df_oos

    def nonparametric_experiment(self, sample_size=1000):
        """
        Run nonparametric regression models and compare results with optimization for each seed
        """
        # Initialize lists to store results for each model
        model_names = ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'Neural Network']
        alpha1_dep_list_two_step = []
        beta_dep_distance_list_two_step = []
        beta_dep_education_list_two_step = []
        beta_dep_gdp_list_two_step = []
        intercept_dep_list_two_step = []
        num_data_list_step1 = []
        num_data_list_step2 = []
        mse_list_two_step = []
        mse_test_two_step = []
        runtime_two_step = []

        # Dictionary to store all results
        train_mse_results = {model: [] for model in model_names}
        test_mse_results = {model: [] for model in model_names}
        train_mae_results = {model: [] for model in model_names}
        test_mae_results = {model: [] for model in model_names}
        train_r2_results = {model: [] for model in model_names}
        test_r2_results = {model: [] for model in model_names}
        runtime_results = {model: [] for model in model_names}
        best_params_results = {model: [] for model in model_names}

        num_data_list_train = []
        num_data_list_test = []

        seed_list = range(16, 26)
        step2_orig_data = pd.DataFrame()
        step1_reg_data = pd.DataFrame()
        nonparametric_results_all = pd.DataFrame()

        # ✅ Initialize optimization results storage
        all_twostep_insample = pd.DataFrame()
        all_twostep_oos = pd.DataFrame()
        all_ml_insample = pd.DataFrame()
        all_ml_oos = pd.DataFrame()

        # create output_dir
        date_time = datetime.now().strftime("%Y%m%d")

        base_dir = "/Users/huangrongsun/Library/CloudStorage/OneDrive-TheOhioStateUniversity/EVChargingLocationWin"
        output_dir = os.path.join(base_dir, "nonparametric_results", date_time)

        os.makedirs(output_dir, exist_ok=True)

        for seed in seed_list:
            print(f"\n{'=' * 80}")
            print(f"Processing Seed: {seed}")
            print(f"{'=' * 80}")

            random_dates = self.assign_random_dates(seed)
            experiment.data_generation(seed)
            temp_data = experiment.regression_data
            temp_data['seed'] = seed

            step1_data, step1_regression_data = self.step1_regression_data(temp_data)
            step1_data = step1_data.dropna()
            step1_reg_data = pd.concat([step1_reg_data, step1_regression_data], ignore_index=True)
            self.step1_reg_data = step1_reg_data
            step2_orig_data = pd.concat([step2_orig_data, step1_data], ignore_index=True)
            self.step2_orig_data = step2_orig_data
            num_data_list_step2.append(len(step2_orig_data))
            num_data_list_step1.append(len(step1_reg_data))

            # Start time for two_step
            start_time = time.time()
            self.step1_regression(step1_reg_data)
            reg_results, coef_df = self.step2_regression(step2_orig_data)
            runtime_two_step.append(time.time() - start_time)

            reg_results['seed'] = seed
            coef_df['seed'] = seed
            reg_results = pd.concat([reg_results, reg_results], ignore_index=True)
            coef_df = pd.concat([coef_df, coef_df], ignore_index=True)

            # Get estimated results
            alpha1_dep = self.step1_intercept
            beta_distance_dep = self.step1_coef[0]
            beta_education_dep = self.step2_dep_coef[0]
            beta_gdp_dep = self.step2_dep_coef[1]
            intercept_dep = self.step2_dep_intercept
            mse_dep = self.step2_dep_mse
            mse_test = self.step2_dep_mse_test

            # Append results
            alpha1_dep_list_two_step.append(alpha1_dep)
            beta_dep_distance_list_two_step.append(beta_distance_dep)
            beta_dep_education_list_two_step.append(beta_education_dep)
            beta_dep_gdp_list_two_step.append(beta_gdp_dep)
            intercept_dep_list_two_step.append(intercept_dep)
            mse_list_two_step.append(mse_dep)
            mse_test_two_step.append(mse_test)

            print(f"Two-Step Results:")
            print(f"  Train MSE: {mse_dep:.2f}")
            print(f"  Test MSE: {mse_test:.2f}")
            print(f"  Train R²: {self.step2_dep_r2:.4f}")
            print(f"  Test R²: {self.step2_dep_r2_test:.4f}")

            train_data = self.training_data
            test_data = self.testing_data

            num_data_list_train.append(len(train_data))
            num_data_list_test.append(len(test_data))

            print(f"\nTraining data size: {len(train_data)}")
            print(f"Test data size: {len(test_data)}")

            feature_list = [
                ['normalized_education_level', 'normalized_gdp', 'sum_capacity'],
                ['normalized_education_level', 'normalized_gdp', 'exp_distance_capacity'],
                ['normalized_education_level', 'normalized_gdp', 'assumption_exp_distance_capacity'],
                ['normalized_education_level', 'normalized_gdp', 'step1_exp_distance_capacity']
            ]

            # Map feature_list to 4 feature type H1, H2, H3, H4
            feature_type_map = ['H1', 'H2', 'H3', 'H4']

            print(f"\n{'=' * 80}")
            print("Starting Nonparametric Regression Training...")
            print(f"{'=' * 80}")

            # ✅ Loop through all 4 feature versions
            for idx, features in enumerate(feature_list):
                feature_type = feature_type_map[idx]
                print(f"\n{'=' * 80}")
                print(f"Training with Feature Set {feature_type}: {features}")
                print(f"{'=' * 80}")

                nonparametric_df = self.nonparametric_regression_tuning(train_data, test_data, features, feature_type)
                nonparametric_df['Feature_Type'] = feature_type
                nonparametric_df['seed'] = seed
                nonparametric_df['Num_Train_Data'] = len(train_data)
                nonparametric_df['Num_Test_Data'] = len(test_data)

                # Store results for each model with this feature set
                for model_name in model_names:
                    if model_name in self.nonparametric_regression_results['decision_dependent']:
                        results = self.nonparametric_regression_results['decision_dependent'][model_name]

                        train_mse_results[model_name].append(results['train_metrics']['mse'])
                        test_mse_results[model_name].append(results['test_metrics']['mse'])
                        train_mae_results[model_name].append(results['train_metrics']['mae'])
                        test_mae_results[model_name].append(results['test_metrics']['mae'])
                        train_r2_results[model_name].append(results['train_metrics']['r2'])
                        test_r2_results[model_name].append(results['test_metrics']['r2'])
                        runtime_results[model_name].append(results['runtime'])
                        best_params_results[model_name].append(str(results['best_params']))
                    else:
                        train_mse_results[model_name].append(np.nan)
                        test_mse_results[model_name].append(np.nan)
                        train_mae_results[model_name].append(np.nan)
                        test_mae_results[model_name].append(np.nan)
                        train_r2_results[model_name].append(np.nan)
                        test_r2_results[model_name].append(np.nan)
                        runtime_results[model_name].append(np.nan)
                        best_params_results[model_name].append(None)

                nonparametric_results_all = pd.concat([nonparametric_results_all, nonparametric_df], ignore_index=True)
                output_file = os.path.join(output_dir, f"nonparametric_results_seed{seed}_featureset{feature_type}.csv")
                nonparametric_df.to_csv(output_file, index=False)
                print(f"Saved nonparametric results to {output_file}")

            # ============================================================
            # ✅ OPTIMIZATION EXPERIMENTS FOR THIS SEED
            # ============================================================
            print(f"\n{'=' * 80}")
            print(f"Running Optimization Experiments for Seed {seed}")
            print(f"{'=' * 80}")

            # Sample residuals for this seed
            rng = np.random.default_rng(seed=seed)  # Use current seed for sampling
            sampled_indices = rng.choice(len(self.step2_dep_residuals), size=sample_size, replace=False)
            sampled_step2_dep_residuals = [self.step2_dep_residuals[i] for i in sampled_indices]

            # Two-step optimization
            self.initialize_for_optimization()
            start_time = time.time()
            twostep_location_values, twostep_optimal_value, twostep_revenue_value, twostep_real_demand_values = self.insample_two_step(
                sampled_step2_dep_residuals)
            twostep_runtime = time.time() - start_time
            twostep_location_values_oos, twostep_optimal_value_oos, twostep_revenue_value_oos, twostep_real_demand_values_oos = self.oos_cost(
                decision=twostep_location_values)

            # Store two-step results with seed
            twostep_insample_df = pd.DataFrame([{
                'seed': seed,
                'model': 'Two-Step',
                'feature_type': 'N/A',
                'location_values': twostep_location_values,
                'optimal_value': twostep_optimal_value,
                'revenue_value': twostep_revenue_value,
                'real_demand_values': twostep_real_demand_values,
                'runtime': twostep_runtime
            }])

            twostep_oos_df = pd.DataFrame([{
                'seed': seed,
                'model': 'Two-Step',
                'feature_type': 'N/A',
                'location_values': twostep_location_values_oos,
                'optimal_value': twostep_optimal_value_oos,
                'revenue_value': twostep_revenue_value_oos,
                'real_demand_values': twostep_real_demand_values_oos
            }])

            all_twostep_insample = pd.concat([all_twostep_insample, twostep_insample_df], ignore_index=True)
            all_twostep_oos = pd.concat([all_twostep_oos, twostep_oos_df], ignore_index=True)

            print(f"Two-Step Optimization:")
            print(f"  Runtime: {twostep_runtime:.2f}s")
            print(f"  In-sample Optimal Value: {twostep_optimal_value:.2f}")
            print(f"  Out-of-sample Optimal Value: {twostep_optimal_value_oos:.2f}")

            # ML models optimization
            regression_models = ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'Neural Network']
            feature_types = ['H1', 'H2', 'H3', 'H4']

            print(f"\nRunning ML Optimization Experiments...")
            for model_name in regression_models:
                for idx, features_col in enumerate(feature_list):
                    f = feature_types[idx]
                    print(f"\n  {model_name} with {f} features: {features_col}")

                    model_start_time = time.time()
                    ml_location_values, ml_optimal_value, ml_revenue_value, ml_real_demand_values, mipgap, status = self.insample_ml_sample(
                        model_name, train_data, feature_type=f, features=features_col)
                    model_runtime = time.time() - model_start_time

                    current_result = pd.DataFrame([{
                        'seed': seed,
                        'model': model_name,
                        'feature_type': f,
                        'location_values': ml_location_values,
                        'optimal_value': ml_optimal_value,
                        'revenue_value': ml_revenue_value,
                        'real_demand_values': ml_real_demand_values,
                        'mipgap': mipgap,
                        'status': status,
                        'runtime': model_runtime
                    }])

                    all_ml_insample = pd.concat([all_ml_insample, current_result], ignore_index=True)

                    # save to file
                    insample_output_file = os.path.join(output_dir, f"ml_insample_seed{seed}_{model_name}_{f}.csv")
                    current_result.to_csv(insample_output_file, index=False)
                    print(f"    Saved in-sample optimization results to {insample_output_file}")

                    if ml_location_values is None:
                        print("    ⚠️ Skipping OOS evaluation due to failed in-sample optimization.")
                        current_oos_result = pd.DataFrame([{
                            'seed': seed,
                            'model': model_name,
                            'feature_type': f,
                            'location_values': None,
                            'optimal_value': None,
                            'revenue_value': None,
                            'real_demand_values': None
                        }])
                    else:
                        ml_location_values_oos, ml_optimal_value_oos, ml_revenue_value_oos, ml_real_demand_values_oos = self.oos_cost(
                            decision=ml_location_values)

                        current_oos_result = pd.DataFrame([{
                            'seed': seed,
                            'model': model_name,
                            'feature_type': f,
                            'location_values': ml_location_values_oos,
                            'optimal_value': ml_optimal_value_oos,
                            'revenue_value': ml_revenue_value_oos,
                            'real_demand_values': ml_real_demand_values_oos
                        }])

                        print(
                            f"    Runtime: {model_runtime:.2f}s | In-sample: {ml_optimal_value:.2f} | OOS: {ml_optimal_value_oos:.2f}")

                    all_ml_oos = pd.concat([all_ml_oos, current_oos_result], ignore_index=True)
                    # save to file
                    oos_output_file = os.path.join(output_dir, f"ml_oos_seed{seed}_{model_name}_{f}.csv")
                    current_oos_result.to_csv(oos_output_file, index=False)
                    print(f"    Saved OOS optimization results to {oos_output_file}")

            # Save optimization results for this seed
            seed_opt_dir = os.path.join(output_dir, f"seed_{seed}_optimization")
            os.makedirs(seed_opt_dir, exist_ok=True)

            twostep_insample_df.to_csv(os.path.join(seed_opt_dir, "twostep_insample.csv"), index=False)
            twostep_oos_df.to_csv(os.path.join(seed_opt_dir, "twostep_oos.csv"), index=False)
            all_ml_insample[all_ml_insample['seed'] == seed].to_csv(os.path.join(seed_opt_dir, "ml_insample.csv"),
                                                                    index=False)
            all_ml_oos[all_ml_oos['seed'] == seed].to_csv(os.path.join(seed_opt_dir, "ml_oos.csv"), index=False)

        # Create two-step results DataFrame
        results_df = pd.DataFrame({
            'Num Data Points(Step1)': num_data_list_step1,
            'Num Data Points(Step2)': num_data_list_step2,
            'Alpha1(Two Step)': alpha1_dep_list_two_step,
            'Beta Distance(Two Step)': beta_dep_distance_list_two_step,
            'Beta Education(Two Step)': beta_dep_education_list_two_step,
            'Beta GDP(Two Step)': beta_dep_gdp_list_two_step,
            'Intercept(Two Step)': intercept_dep_list_two_step,
            'MSE(Two Step)': mse_list_two_step,
            'MSE Test(Two Step)': mse_test_two_step,
            'R2(Two Step)': self.step2_dep_r2,
            'R2 Test(Two Step)': self.step2_dep_r2_test,
            'Runtime(Two Step)': runtime_two_step,
        })
        self.two_step_results = results_df
        self.nonparametric_results_all = nonparametric_results_all

        # Store optimization results
        self.all_twostep_insample = all_twostep_insample
        self.all_twostep_oos = all_twostep_oos
        self.all_ml_insample = all_ml_insample
        self.all_ml_oos = all_ml_oos

        # Save consolidated optimization results
        all_twostep_insample.to_csv(os.path.join(output_dir, "all_twostep_insample.csv"), index=False)
        all_twostep_oos.to_csv(os.path.join(output_dir, "all_twostep_oos.csv"), index=False)
        all_ml_insample.to_csv(os.path.join(output_dir, "all_ml_insample.csv"), index=False)
        all_ml_oos.to_csv(os.path.join(output_dir, "all_ml_oos.csv"), index=False)

        print(f"\n{'=' * 80}")
        print("TWO-STEP REGRESSION RESULTS")
        print(f"{'=' * 80}")
        print(self.two_step_results.to_string())

        print(f"\n{'=' * 80}")
        print("NONPARAMETRIC REGRESSION RESULTS (All Feature Sets)")
        print(f"{'=' * 80}")
        print(self.nonparametric_results_all.to_string())

        print(f"\n{'=' * 80}")
        print("OPTIMIZATION RESULTS SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total seeds processed: {len(seed_list)}")
        print(f"Two-step in-sample results: {len(all_twostep_insample)} rows")
        print(f"Two-step OOS results: {len(all_twostep_oos)} rows")
        print(f"ML in-sample results: {len(all_ml_insample)} rows")
        print(f"ML OOS results: {len(all_ml_oos)} rows")

        return self.two_step_results, self.nonparametric_results_all, all_twostep_insample, all_twostep_oos, all_ml_insample, all_ml_oos


# %%
if __name__ == "__main__":
    # Initialize the experiment
    experiment = synthetic_experiments(
        seed=16,
        year_range=range(2017, 2023),
        numCustomers=30,
        numStations=30,
        start_date_str="2017-01-01",
        end_date_str="2022-12-31",
    )

    # Run the various stages of the experiment
    # experiment.step2_verify()
    two_step_regression_results, nonparametric_regression = experiment.nonparametric_verify()
    # twostep_insample_results, twostep_oos_results, ml_results, ml_oos_results = experiment.insample_experiments_sample_size()

    # run nonparametric experiment with optimization
    # two_step_results, nonparametric_results_all, all_twostep_insample, all_twostep_oos, all_ml_insample, all_ml_oos = experiment.nonparametric_experiment(sample_size=1000)
# %%

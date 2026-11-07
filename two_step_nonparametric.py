#%%
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from datetime import datetime, timedelta
import random
import os
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.metrics.pairwise import haversine_distances
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error, max_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import random
import math
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import time

#%%
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
        self.noise_sd = 10

        # coefficients for decision-dependent regression model
        self.dep_alpha1 = 4
        self.dep_beta_distance = -0.05
        self.dep_beta_education = 100
        self.dep_beta_gdp = 1000
        self.dep_intercept = 1500

        self.test_sizeData = 5000

        self.seed_value = 16

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

    def step2_regression_data_step(self, data):
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

            # print(z_matrix, z_dist_temp, z_time_temp, z_capacity_temp)

            capacity_matrix = z_matrix @ np.diag(z_capacity_temp)
            # print(capacity_matrix)
            exp_term = np.exp(self.step1_intercept + self.step1_coef * z_dist_temp)
            # print(exp_term)

            exp_demand = capacity_matrix @ exp_term.reshape(-1, 1)
            exp_demand = exp_demand.flatten()
            # print(exp_demand)

            # Create a DataFrame for the demand and add it as a new column
            demand_df = pd.DataFrame(exp_demand, columns=['ested_exp_demand'])
            # print(demand_df)
            data_copy = data.copy()
            data_copy['ested_exp_demand'] = demand_df['ested_exp_demand'].values
            # print(data_copy)

            step2_data_list.append(data_copy)

            # Combine all customer data into one DataFrame
        step2_data = pd.concat(step2_data_list, ignore_index=True)
        self.step2_data = step2_data

        # decision-dependent regression
        training_data = self.step2_data[self.step2_data['Year'] <= 2020]
        self.step2_regression_data = training_data
        test_data = self.step2_data[self.step2_data['Year'] >= 2021]  # Combine test1_data (2021) and test2_data (2022)

        self.step2_Y_train_dep = training_data['demand'] - training_data['ested_exp_demand'] - training_data[
            'initial_demand']
        self.step2_Y_train_ind = training_data['demand'] - training_data['initial_demand']
        self.step2_X_train = training_data[['normalized_education_level', 'normalized_gdp']]

        self.test_Y_dep = test_data['demand'] - test_data['ested_exp_demand'] - test_data['initial_demand']
        self.test_Y_ind = test_data['demand'] - test_data['initial_demand']
        self.test_X = test_data[['normalized_education_level', 'normalized_gdp']]
        self.test_data = test_data

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
    
    def nonparametric_tuning(self):
        # Define models
        models = {
            'Decision Tree': DecisionTreeRegressor(),
            'Random Forest': RandomForestRegressor(),
            'Gradient Boosting': HistGradientBoostingRegressor(),
            'kNN': KNeighborsRegressor()
        }

        # Parameter grids for hyperparameter tuning
        param_grids = {
            'Decision Tree': {'max_depth': range(3, 10), 'min_samples_split': range(2, 10),
                              'random_state': range(40, 50)},
            'Random Forest': {'random_state': range(40, 50), 'n_estimators': [50, 100, 150], 'max_depth': range(3, 10),
                              'min_samples_split': range(2, 10)},
            'Gradient Boosting': {'max_depth': range(3, 10), 'learning_rate': [0.01, 0.1, 0.2]},
            'kNN': {'n_neighbors': range(3, 10)}
        }
        
        # # Parameter grids for hyperparameter tuning
        # param_grids = {
        #     'Decision Tree': {'max_depth': range(9, 10), 'min_samples_split': range(9, 10),
        #                       'random_state': range(49, 50)},
        #     'Random Forest': {'random_state': range(49, 50), 'n_estimators': [50], 'max_depth': range(9, 10),
        #                       'min_samples_split': range(9, 10)},
        #     'Gradient Boosting': {'max_depth': range(9, 10), 'learning_rate': [0.01]},
        #     'kNN': {'n_neighbors': range(9, 10)}
        # }

        # Initialize dictionaries to store best models and results
        self.best_models_dep = {}
        self.best_models_ind = {}
        self.nonparametric_regression_results = {'decision_dependent': {}, 'decision_independent': {}}
        self.residuals_dep = {}
        self.residuals_ind = {}
        self.residuals_test_dep = {}
        self.residuals_test_ind = {}

        # Tune for decision-dependent
        for model_name, model in models.items():
            grid_search_dep = GridSearchCV(estimator=model, param_grid=param_grids[model_name],
                                           scoring='neg_mean_squared_error', cv=5, verbose=1)
            grid_search_dep.fit(self.step2_X_train, self.step2_Y_train_dep)
            dep_best_model = grid_search_dep.best_estimator_

            # Store best model and predictions
            self.best_models_dep[model_name] = dep_best_model
            train_dep_pred = dep_best_model.predict(self.step2_X_train)
            test_y_pred_dep = dep_best_model.predict(self.test_X)

            # Compute metrics for decision-dependent model
            train_metrics_dep = {
                'mse': mean_squared_error(self.step2_Y_train_dep, train_dep_pred),
                'r2': r2_score(self.step2_Y_train_dep, train_dep_pred),
                'mae': mean_absolute_error(self.step2_Y_train_dep, train_dep_pred),
                'medae': median_absolute_error(self.step2_Y_train_dep, train_dep_pred),
                'max_error': max_error(self.step2_Y_train_dep, train_dep_pred)
            }
            test_metrics_dep = {
                'mse': mean_squared_error(self.test_Y_dep, test_y_pred_dep),
                'r2': r2_score(self.test_Y_dep, test_y_pred_dep),
                'mae': mean_absolute_error(self.test_Y_dep, test_y_pred_dep),
                'medae': median_absolute_error(self.test_Y_dep, test_y_pred_dep),
                'max_error': max_error(self.test_Y_dep, test_y_pred_dep)
            }

            self.nonparametric_regression_results['decision_dependent'][model_name] = {
                'train_metrics': train_metrics_dep,
                'test_metrics': test_metrics_dep
            }

            self.residuals_dep[model_name] = self.step2_Y_train_dep - train_dep_pred
            self.residuals_test_dep[model_name] = self.test_Y_dep - test_y_pred_dep

        # Tune for decision-independent
        for model_name, model in models.items():
            grid_search_ind = GridSearchCV(estimator=model, param_grid=param_grids[model_name],
                                           scoring='neg_mean_squared_error', cv=5, verbose=1)
            grid_search_ind.fit(self.step2_X_train, self.step2_Y_train_ind)
            ind_best_model = grid_search_ind.best_estimator_

            # Store best model and predictions
            self.best_models_ind[model_name] = ind_best_model
            train_ind_pred = ind_best_model.predict(self.step2_X_train)
            test_y_pred_ind = ind_best_model.predict(self.test_X)

            # Compute metrics for decision-independent model
            train_metrics_ind = {
                'mse': mean_squared_error(self.step2_Y_train_ind, train_ind_pred),
                'r2': r2_score(self.step2_Y_train_ind, train_ind_pred),
                'mae': mean_absolute_error(self.step2_Y_train_ind, train_ind_pred),
                'medae': median_absolute_error(self.step2_Y_train_ind, train_ind_pred),
                'max_error': max_error(self.step2_Y_train_ind, train_ind_pred)
            }
            test_metrics_ind = {
                'mse': mean_squared_error(self.test_Y_ind, test_y_pred_ind),
                'r2': r2_score(self.test_Y_ind, test_y_pred_ind),
                'mae': mean_absolute_error(self.test_Y_ind, test_y_pred_ind),
                'medae': median_absolute_error(self.test_Y_ind, test_y_pred_ind),
                'max_error': max_error(self.test_Y_ind, test_y_pred_ind)
            }

            self.nonparametric_regression_results['decision_independent'][model_name] = {
                'train_metrics': train_metrics_ind,
                'test_metrics': test_metrics_ind
            }

            self.residuals_ind[model_name] = self.step2_Y_train_ind - train_ind_pred
            self.residuals_test_ind[model_name] = self.test_Y_ind - test_y_pred_ind

        # Prepare the data for DataFrame
        data = []
        for model_name in set(self.nonparametric_regression_results['decision_dependent'].keys()).union(
                self.nonparametric_regression_results['decision_independent'].keys()):
            dep_results = self.nonparametric_regression_results['decision_dependent'].get(model_name,
                                                                                          {'train_metrics': {},
                                                                                           'test_metrics': {}})
            ind_results = self.nonparametric_regression_results['decision_independent'].get(model_name,
                                                                                            {'train_metrics': {},
                                                                                             'test_metrics': {}})

            # Append all results into the data list
            data.append([
                model_name,
                dep_results['train_metrics'].get('mse'), dep_results['test_metrics'].get('mse'),
                dep_results['train_metrics'].get('mae'), dep_results['test_metrics'].get('mae'),
                dep_results['train_metrics'].get('medae'), dep_results['test_metrics'].get('medae'),
                dep_results['train_metrics'].get('max_error'), dep_results['test_metrics'].get('max_error'),
                ind_results['train_metrics'].get('mse'), ind_results['test_metrics'].get('mse'),
                ind_results['train_metrics'].get('mae'), ind_results['test_metrics'].get('mae'),
                ind_results['train_metrics'].get('medae'), ind_results['test_metrics'].get('medae'),
                ind_results['train_metrics'].get('max_error'), ind_results['test_metrics'].get('max_error')
            ])

        df_columns = ['Model Name', 'Dep_Train_MSE', 'Dep_Test_MSE', 'Dep_Train_MAE', 'Dep_Test_MAE', 'Dep_Train_MedAE',
                      'Dep_Test_MedAE',
                      'Dep_Train_MaxError', 'Dep_Test_MaxError', 'Ind_Train_MSE', 'Ind_Test_MSE', 'Ind_Train_MAE',
                      'Ind_Test_MAE',
                      'Ind_Train_MedAE', 'Ind_Test_MedAE', 'Ind_Train_MaxError', 'Ind_Test_MaxError']

        self.nonparametric_df = pd.DataFrame(data, columns=df_columns)
        
        return self.nonparametric_df

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

    def step2_verify(self):
        alpha1_dep_list_two_step = []
        beta_dep_distance_list_two_step = []
        beta_dep_education_list_two_step = []
        beta_dep_gdp_list_two_step = []
        intercept_dep_list_two_step = []
        num_data_list_step1 = []
        num_data_list_step2 = []
        mse_list_two_step = []
        runtime_two_step = []
        mse_test_two_step = []
        scenario_labels = []

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
            step1_reg_data_part, step1_data_part = self.step1_regression_data(temp_data)
            self.step1_data_part = step1_data_part
            self.step1_reg_data_part = step1_reg_data_part
            step1_data_part = step1_data_part.dropna()

            data_storage[seed] = {
                'step1_reg': step1_reg_data_part,
                'step2_orig': step1_data_part
            }

        scenario_lists = [list(range(16, i)) for i in range(17, 25)]
        seeds_scenario_1 = list(range(16, 24)) + [25]  # 16-23 + 25
        seeds_scenario_2 = list(range(16, 27))  # 16-26
        seeds_scenario_3 = list(range(16, 28))  # 16-27

        scenarios = [
            ("Seeds 16-23 + 25", seeds_scenario_1),
            ("Seeds 16-26", seeds_scenario_2),
            ("Seeds 16-27", seeds_scenario_3),
        ]

        # add expanding-window scenarios
        for i, seeds in enumerate(scenario_lists, start=17):
            scenarios.append((f"Seeds 16-{i - 1}", seeds))

        for label, current_seeds in scenarios:
            print(f">>> Processing scenario: {label} with seeds {current_seeds}...")
            scenario_labels.append(label)

            # 3.1 拼接数据
            current_step1_reg_data = pd.DataFrame()
            current_step2_orig_data = pd.DataFrame()

            for s in current_seeds:
                current_step1_reg_data = pd.concat([current_step1_reg_data, data_storage[s]['step1_reg']],
                                                   ignore_index=True)
                self.current_step1_reg_data = current_step1_reg_data
                current_step2_orig_data = pd.concat([current_step2_orig_data, data_storage[s]['step2_orig']],
                                                    ignore_index=True)

            self.step2_orig_data = current_step2_orig_data

            num_data_list_step2.append(len(current_step2_orig_data))
            num_data_list_step1.append(len(current_step1_reg_data))

            # ---------------------------
            # 3.2 Two-Step Regression
            # ---------------------------
            start_time = time.time()
            self.step1_regression(current_step1_reg_data)
            self.step2_regression_data_step(current_step2_orig_data)
            runtime_two_step.append(time.time() - start_time)

            # 获取并存储 Two-Step 结果
            alpha1_dep_list_two_step.append(self.step1_intercept)
            beta_dep_distance_list_two_step.append(self.step1_coef[0])
            beta_dep_education_list_two_step.append(self.step2_dep_coef[0])
            beta_dep_gdp_list_two_step.append(self.step2_dep_coef[1])
            intercept_dep_list_two_step.append(self.step2_dep_intercept)
            mse_list_two_step.append(self.step2_dep_mse)
            mse_test_two_step.append(self.step2_dep_mse_test)

            print(f"Data points: {len(current_step2_orig_data)}")

            # ---------------------------
            # 3.3 Nonparametric Regression
            # ---------------------------
            temp_nonparametric_df = self.nonparametric_tuning()  # Non-parametric regression
            temp_nonparametric_df['num_data_list_step2'] = len(current_step2_orig_data)
            temp_nonparametric_df['scenario'] = label
            nonparametric_df = pd.concat([nonparametric_df, temp_nonparametric_df], ignore_index=True)

        # seed_list = range(16, 35)
        # full_data = pd.DataFrame()
        # step2_orig_data = pd.DataFrame()
        # nonparametric_df = pd.DataFrame()
        # step1_reg_data = pd.DataFrame()
        # for seed in seed_list:
        #     random_dates = self.assign_random_dates(seed)
        #     experiment.data_generation(seed)  # Generate data
        #     # print(f"Distance data for seed {seed},", self.distance_df)
        #     # print(f"Open Dates for seed {seed},", self.sorted_dates_df)
        #     temp_data = experiment.regression_data
        #     temp_data['seed'] = seed
        #
        #     step1_regression_data, step1_data = self.step1_regression_data(temp_data)  # Step 1 regression
        #     step1_reg_data = pd.concat([step1_reg_data, step1_regression_data], ignore_index=True)
        #     step2_orig_data = pd.concat([step2_orig_data, step1_data], ignore_index=True)
        #     num_data_list_step2.append(len(step2_orig_data))
        #     num_data_list_step1.append(len(step1_reg_data))
        #     self.step1_regression(step1_reg_data)
        #
        #     self.step2_regression_data_step(step2_orig_data)  # Step 2 regression
        #
        #     temp_nonparametric_df = self.nonparametric_tuning()  # Non-parametric regression
        #     temp_nonparametric_df['num_data_list_step2'] = len(step2_orig_data)
        #     temp_nonparametric_df['seed'] = seed
        #     nonparametric_df = pd.concat([nonparametric_df, temp_nonparametric_df], ignore_index=True)
        #
        #
        # # plot mse for step2 ve number of data points for each regression appracoh
        # grouped_data = nonparametric_df.groupby('Model Name')
        # plt.figure(figsize=(10, 6))
        # for name, group in grouped_data:
        #     plt.plot(group['num_data_list_step2'], group['Dep_Test_MSE'], label=name)
        # plt.title('MSE for Step 2 vs Number of Data Points')
        # plt.legend()
        # plt.xlabel('Number of Data Points v.s. MSE')
        # plt.ylabel('MSE')
        # plt.show()
        
        # combine them as a dataframe
        results_df = nonparametric_df
        self.two_step_results = results_df
        


#%%
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
    experiment.step2_verify()


# %%

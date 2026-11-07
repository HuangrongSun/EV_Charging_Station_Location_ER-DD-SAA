# %%
import os
import pandas as pd
import numpy as np
import joblib
import pickle
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from scipy.stats import zscore
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


class exponential_day_effect:
    def __init__(self, t_threshold, distance_threshold, numFacilities, numCustomers):
        self.t_threshold = t_threshold
        self.distance_threshold = distance_threshold
        self.numFacilities = numFacilities
        self.numCustomers = numCustomers

    def data_processing(self):
        # ===================================================
        # read ev registration data
        # ===================================================
        directory = 'original registration'
        # Get a list of all files in the directory
        files = os.listdir(directory)
        # Filter files that start with '20' and end with '.csv'
        csv_files = [file for file in files if file.startswith('20') and file.endswith('.csv')]
        # Combine all CSV files into one DataFrame
        combined_df = pd.concat([pd.read_csv(os.path.join(directory, file)) for file in csv_files], ignore_index=True)
        combined_df = combined_df.dropna(subset=['County'])
        # Drop data for years 2019, 2021, and 2023
        # combined_df = combined_df[(combined_df['Year'] != 2019) & (combined_df['Year'] != 2021) & (combined_df['Year'] != 2023)]
        # combine registrations for phev and bev
        ev_registration_df = combined_df.groupby(['Year', 'Month', 'County']).sum(numeric_only=True)[
            'EV Original Registrations'].reset_index()
        # Dictionary mapping month names to numerical values
        month_to_float = {'January': 12, 'February': 1, 'March': 2, 'April': 3, 'May': 4, 'June': 5,
                          'July': 6, 'August': 7, 'September': 8, 'October': 9, 'November': 10, 'December': 11}
        # Convert 'Month' column to numerical values using the month_to_float dictionary
        ev_registration_df['Month'] = ev_registration_df['Month'].map(month_to_float)
        # for january, the year should be -1
        ev_registration_df['Year'] = ev_registration_df.apply(
            lambda row: row['Year'] - 1 if row['Month'] == 12 else row['Year'], axis=1
        )
        # Calculate the number of days in each month
        month_days = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
                      7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
        # Compute the average number of EV Original Registrations per day
        ev_registration_df['avg'] = ev_registration_df.apply(
            lambda row: row['EV Original Registrations'] / month_days[int(row['Month'])], axis=1)
        self.registration_data = ev_registration_df

        # ===================================================
        # load ev charging stations data, drop duplicate for multiple stations opened in one day (drop all if distance is large, keep first if small)
        # ===================================================
        stations_cols = ['Station Name', 'Latitude', 'Longitude', 'Open Date', 'EV Level2 EVSE Num', 'EV DC Fast Count']
        stations_types = {'Station Name': str, 'Latitude': float, 'Longitude': float, 'EV Level2 EVSE Num': float,
                          'EV DC Fast Count': float}
        charging_stations_df = pd.read_csv('original registration/ny_stations.csv', usecols=stations_cols,
                                           dtype=stations_types, parse_dates=['Open Date'])
        # Group the data by year and count rows for each year
        number_of_stations = charging_stations_df.groupby(charging_stations_df['Open Date'].dt.year).size()
        charging_stations_df = charging_stations_df[
            (charging_stations_df['Open Date'].dt.year >= 2017) & (charging_stations_df['Open Date'].dt.year <= 2022)]
        charging_stations_df = charging_stations_df.sort_values(by=['Open Date'])
        charging_stations_df = charging_stations_df.reset_index()
        charging_stations_df['index'] = charging_stations_df.index
        # Fill NA values with 0 in specific columns
        charging_stations_df['EV Level2 EVSE Num'].fillna(0, inplace=True)
        charging_stations_df['EV DC Fast Count'].fillna(0, inplace=True)
        charging_stations_df['capacity'] = charging_stations_df['EV Level2 EVSE Num'] + charging_stations_df[
            'EV DC Fast Count']

        # ===================================================
        # load counties data
        # ===================================================
        counties_cols = ['County', 'Latitude', 'Longitude']
        counties_type = {'Latitude': float, 'Longitude': float, 'County': str}
        counties_data = pd.read_csv('original registration/county_list.csv', usecols=counties_cols, dtype=counties_type)

        # ===================================================
        # compute distance from each station to counties, filter distance within 50 miles
        # ===================================================
        counties_coords = np.radians(counties_data[['Latitude', 'Longitude']].to_numpy())
        stations_coords = np.radians(charging_stations_df[['Latitude', 'Longitude']].to_numpy())
        distance_matrix = haversine_distances(counties_coords, stations_coords) * 6371 * 0.621371
        # Convert distance matrix to DataFrame
        distance_df = pd.DataFrame(distance_matrix, index=counties_data['County'],
                                   columns=charging_stations_df['index'])
        # Reset index to convert 'County' from index to a column
        distance_df.reset_index(inplace=True)
        # Melt the DataFrame to reshape it
        distance_df_melted = pd.melt(distance_df, id_vars='County', var_name='index', value_name='Distance')
        distance_df_melted = distance_df_melted[distance_df_melted['Distance'] <= self.distance_threshold]
        # merge distance_df_melted to charging_stations_df
        county_station_dist = distance_df_melted.merge(
            charging_stations_df[['index', 'Station Name', 'Open Date', 'capacity']],
            on='index', how='inner')
        county_station_dist['Year'] = county_station_dist['Open Date'].dt.year
        county_station_dist = county_station_dist.sort_values(by=['County', 'Year'])
        self.county_station_dist = county_station_dist

        ev_registration_df = self.registration_data.groupby(['Year', 'County']).sum(numeric_only=True)[
            'EV Original Registrations'].reset_index()
        self.step2_registration = ev_registration_df

        # Define a function to process each group
        def get_next_open_dates(group):
            # Get unique open dates sorted
            unique_dates = pd.Series(group['Open Date'].unique()).sort_values().reset_index(drop=True)
            # Create a DataFrame from unique dates
            open_dates_df = pd.DataFrame(unique_dates, columns=['Open Date'])
            # Calculate the next open date for each unique date
            open_dates_df['Next Open Date'] = open_dates_df['Open Date'].shift(-1)
            group = group.merge(open_dates_df, on=['Open Date'], how='left')
            return group

        # Apply the function to each county group and combine the results
        distance_data = self.county_station_dist.groupby('County').apply(get_next_open_dates).reset_index(drop=True)
        # Calculate the interval (will be NaN where 'Next Open Date' is NaN)
        distance_data['interval'] = (distance_data['Next Open Date'] - distance_data['Open Date']).dt.days
        # Fill NaN intervals with 60
        distance_data['interval'] = distance_data['interval'].fillna(self.t_threshold)
        # Apply the threshold to the interval, ensuring no value exceeds T_threshold
        distance_data['time'] = distance_data['interval'].apply(lambda x: min(x, self.t_threshold))

        education_cols = ['County', 'Year', 'Education']
        education_types = {'County': str, 'Year': float, 'Education': float}
        self.education_df = pd.read_csv('data/17-22edu.csv', usecols=education_cols, dtype=education_types)

        # load gdp
        self.gdp_df = pd.read_csv('data/GDP.csv')

        results_df = distance_data.merge(ev_registration_df, on=['County', 'Year'], how='left').merge(self.education_df,
                                                                                                   on=['County',
                                                                                                       'Year'],
                                                                                                   how='inner').merge(
            self.gdp_df, on=['County', 'Year'], how='inner')

        results_df = results_df.sort_values(by=['County', 'Year'])
        results_df = results_df[results_df['GDP'] <= 4 * 10 ** 8]
        results_df = results_df[results_df['EV Original Registrations'] <= 3500]
        # Define columns to normalize
        columns_to_normalize = ['Education', 'GDP']
        # Create a StandardScaler object
        scaler = MinMaxScaler(feature_range=(0, 1))
        results_df[['normalized_education_level', 'normalized_gdp']] = scaler.fit_transform(
            results_df[columns_to_normalize])
        results_df = results_df.sort_values(['County', 'Open Date'])
        results_df = results_df.reset_index(drop=True)
        self.orig_data = results_df

        return self.orig_data

    def one_step_regression_model(self, data):
        num_data = len(data)
        customer_index = data.groupby(['County', 'Year']).tail(1).index
        customer_index = customer_index.to_list()
        num_customer_year = len(customer_index)
        customer_index.insert(0, 1)
        m = gp.Model('one_step_regression')
        # m.setParam('DualReductions', 1)
        m.setParam('OutputFlag', 1)
        m.setParam('TimeLimit', 7200)
        m.setParam('NonConvex', 2)
        m.setParam('MIPGap', 0.001)

        # Create variables
        dep_intercept = m.addVar(lb=-GRB.INFINITY, name='dep_intercept')
        beta_distance = m.addVar(lb=-GRB.INFINITY, name='beta_distance')
        beta_education = m.addVar(lb=-GRB.INFINITY, name='beta_education')
        beta_gdp = m.addVar(lb=-GRB.INFINITY, name='beta_gdp')
        intercept = m.addVar(lb=-GRB.INFINITY, name='intercept')
        dep_effect = m.addVars(num_data, lb=-GRB.INFINITY, name='dep_effect')
        est_demand = m.addVars(num_customer_year, lb=-GRB.INFINITY, name='est_demand')
        expr = m.addVars(num_data, lb=-GRB.INFINITY, name='expr')
        error = m.addVars(num_customer_year, lb=-GRB.INFINITY, name='error')

        for i in range(num_data):
            m.addConstr(expr[i] == dep_intercept + beta_distance * float(data.iloc[i]['Distance']),
                        name=f'exponent_{i}')
            m.addGenConstrExp(expr[i], dep_effect[i], name=f'exp_constraint_{i}')

        for i in range(num_customer_year):
            start_idx = customer_index[i]
            end_idx = customer_index[i + 1]

            # print(f'i: {i}, start_idx: {start_idx}, end_idx: {end_idx}')
            if start_idx == end_idx:
                index_range = [start_idx]
            elif end_idx == num_data - 1:
                index_range = list(range(start_idx, end_idx + 1))
            else:
                index_range = list(range(start_idx, end_idx))
            # print(index_range)
            initial_index = index_range[0]

            m.addConstr(
                est_demand[i] == intercept + beta_education * data.iloc[end_idx]['normalized_education_level']
                + beta_gdp * data.iloc[end_idx]['normalized_gdp']
                + gp.quicksum(dep_effect[j] * data.iloc[j]['capacity'] * data.iloc[j]['time'] for j in range(initial_index, end_idx + 1)),
                name=f'customer_constraint_{end_idx}'
            )

            m.addConstr(est_demand[i] + error[i] == data.iloc[end_idx]['EV Original Registrations'],
                        name='demand_constraint')

        m.setObjective(gp.quicksum(error[i] * error[i] for i in range(num_customer_year)) / num_customer_year, GRB.MINIMIZE)
        m.optimize()
        m.write('one_step_regression_capacity.lp')

        # Check solution status
        if m.status == GRB.OPTIMAL:
            # Solution is globally optimal
            # print("Global optimal solution found")
            objective_val = m.objVal
            return dep_intercept.x, beta_distance.x, beta_education.x, beta_gdp.x, intercept.x, objective_val, m.status

        elif m.status == GRB.SUBOPTIMAL:
            # Solver found a local optimal solution or terminated early
            # print("Suboptimal solution found (possibly local optimal)")
            objective_val = m.objVal
            return dep_intercept.x, beta_distance.x, beta_education.x, beta_gdp.x, intercept.x, objective_val, m.status

        elif m.status == GRB.INFEASIBLE:
            # print("Model is infeasible")
            return None, None, None, None, None, None, m.status

        elif m.status == GRB.UNBOUNDED:
            # print("Model is unbounded")
            return None, None, None, None, None, None, m.status

        elif m.status == GRB.TIME_LIMIT:
            if m.solCount > 0:
                # Solution found, but not proven optimal
                # print("Time limit reached. Returning best feasible solution.")
                objective_val = m.objVal
                return dep_intercept.x, beta_distance.x, beta_education.x, beta_gdp.x, intercept.x, objective_val, m.status
            else:
                # No feasible solution found
                # print("Time limit reached. No feasible solution found.")
                return None, None, None, None, None, None, m.status

        else:
            # print("Solver stopped with status:", m.status)
            return None, None, None, None, None, None, m.status

    def one_step_run(self):
        self.data_processing()
        results = self.one_step_regression_model(data=self.orig_data)
        return results


# %%
if __name__ == '__main__':
    # t_list = [1, 15, 30, 45, 60]
    t_list = [30]
    coef_results = pd.DataFrame()
    linear_regression_results = pd.DataFrame()
    step1_reg_df = pd.DataFrame()
    # step2_linear_results = pd.DataFrame()
    # step2_reg_results = pd.DataFrame()
    # nonparametric_results = pd.DataFrame()
    # experiments_results = pd.DataFrame()
    # cost_breakdown_results = pd.DataFrame()  # Corrected typo in variable name
    #
    for t in t_list:
        day_effect_model = exponential_day_effect(
            t_threshold=t,
            distance_threshold=30,
            numFacilities=25,
            numCustomers=5
        )

        results = day_effect_model.one_step_run()

# %%

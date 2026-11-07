# %%
import os
import pandas as pd
import numpy as np
import joblib
import pickle
import warnings
import time
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
from sklearn.neural_network import MLPRegressor
import random
import math
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV
from gurobi_ml import add_predictor_constr
pd.options.mode.chained_assignment = None  # Disables SettingWithCopyWarning


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

    def step_1_data(self):
        county_station_dist = self.county_station_dist[
            (self.county_station_dist['Open Date'].dt.year >= 2016) & (
                    self.county_station_dist['Open Date'].dt.year <= 2023)]
        county_station_dist['Year'] = county_station_dist['Open Date'].dt.year

        # ===================================================
        # for each county, compute increase demand
        # ===================================================

        # define function to count the dates between two dates
        def count_days(start_, end_):
            dates = pd.date_range(start=start_, end=end_)
            dates = dates.to_series()
            counts = dates.groupby([dates.index.year, dates.index.month]).size()
            counts.iloc[0] -= 1
            return counts

        grouped_counties = county_station_dist.groupby(['County', 'Year'])
        county_data_list = []

        for (county, year), group in grouped_counties:
            try:
                county_data = county_station_dist[county_station_dist['County'] == county]
                county_data = county_data[county_data['Year'] == year]
                county_data = county_data.sort_values(by='Open Date')
                county_data['end_day'] = county_data['Open Date'].shift(-1)
                county_data['end_day'] = pd.to_datetime(county_data['end_day'])
                county_data['interval'] = (county_data['end_day'] - county_data['Open Date']).dt.days
                county_data = county_data[:-1]
                increase_demand_list = []
                for i in range(len(county_data)):
                    start_day = county_data.iloc[i]['Open Date']
                    end_day = county_data.iloc[i]['end_day']
                    days_count = count_days(start_day, end_day)
                    totaldemand = pd.Series(dtype=float)
                    for (year, month), num_days in days_count.items():
                        lookup_data = self.registration_data[(self.registration_data['County'] == county) &
                                                             (self.registration_data['Year'] == year) &
                                                             (self.registration_data['Month'] == month)]
                        temp_demand = lookup_data['avg'] * num_days
                        totaldemand = totaldemand.add(temp_demand, fill_value=0)
                    increase_demand = totaldemand.sum()
                    increase_demand_list.append(increase_demand)
                county_data['increase demand'] = increase_demand_list  # Assign the list to the column
                county_data = county_data.drop_duplicates(subset=['Open Date'], keep=False)
                county_data = county_data.reset_index(drop=True)
                county_data = county_data.drop_duplicates(subset=['Open Date'], keep=False)
                county_data = county_data.iloc[:-1]
                county_data_list.append(county_data)
            except ValueError:
                increase_demand = []

        concatenated_data = pd.concat(county_data_list)
        concatenated_data = concatenated_data[(concatenated_data['Year'] >= 2017) & (concatenated_data['Year'] <= 2022)]
        self.step1_data = concatenated_data

    def step1_regression(self):
        # Filter step1_data
        step1_data = self.step1_data[self.step1_data['increase demand'] > 1e-5]
        # Further filter for years <= 2020
        step1_regression_data = step1_data[step1_data['Year'] <= 2020].copy()
        # calculate the duration time from the establishment date to the end date of the training year
        step1_regression_data.loc[:, 'interval'] = step1_regression_data['interval'].apply(
            lambda x: min(x, self.t_threshold))
        # Assign the updated DataFrame to the class attribute
        self.step1_regression_data = step1_regression_data

        # Prepare the features (X) and target (Y) for regression
        X = self.step1_regression_data['Distance'].values.reshape(-1, 1)
        Y = np.log(
            self.step1_regression_data['increase demand'] / self.step1_regression_data['capacity'] /
            self.step1_regression_data['interval'])

        # remove outliers
        # Remove outliers using Z-score method
        Y_zscores = zscore(Y)
        mask = (np.abs(Y_zscores) < 3)
        X = X[mask]
        Y = Y[mask]

        start_time = time.time()
        # Fit a linear regression model
        model = LinearRegression()
        model.fit(X, Y)

        # Predict the y values
        y_pred = model.predict(X)
        run_time = time.time() - start_time

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
        self.step1_runtime = run_time

        # Calculate number of bins based on the square root of the number of samples
        n_bins = math.ceil(math.sqrt(len(self.step1_residuals)))

        plt.figure(figsize=(10, 8))
        # Plot 1: Actual vs Predicted
        plt.scatter(X, Y, color='navy', label='Actual Data', alpha=0.7)
        plt.plot(X, y_pred, color='orange', linewidth=2, label='Linear Regression')
        plt.xlabel('Distance', fontsize=24)
        plt.ylabel(r'$R^k$', fontsize=24)
        # plt.title('Distance vs. $R^k$', fontsize=20)
        plt.legend(loc='upper left', fontsize=20, frameon=True, fancybox=True, edgecolor='black', framealpha=1.0)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()
        # Plot 2: Histogram of Residuals

        plt.figure(figsize=(10, 8))
        plt.hist(self.step1_residuals, bins=n_bins, color='navy', edgecolor='black', alpha=0.7)
        plt.xlabel('Residuals', fontsize=24)
        plt.ylabel('Frequency', fontsize=24)
        # plt.title(f'Distribution of Residuals in Step-1', fontsize=20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # plt.grid(True, linestyle='--', linewidth=0.5)
        # Adjust space between plots
        # Save the figure as a high-resolution image
        # plt.savefig('step1.png', dpi=300, bbox_inches='tight')
        # Display the plot
        plt.show()

        # Create a DataFrame to return the results
        # compute the p-value for this step-1 regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(X.flatten(), Y)

        # Create a DataFrame to return the results
        results = pd.DataFrame({
            'Data Size': [len(self.step1_regression_data)],
            'Coefficient': [self.step1_coef],
            'Intercept': [self.step1_intercept],
            'MSE': [self.step1_mse],
            'R^2': [self.step1_r2],
            'MAE': [self.step1_mae],
            'ME': [self.step1_me],
            'p-value': [p_value],
            'r-value': [r_value],
            'std_err': [std_err]
        })

        self.step1_results_df = results

        return results

    def step2_data_func(self):
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

        # # compute exponential term
        intercept = self.step1_intercept
        coef = self.step1_coef[0]
        #
        # def dist_regression(distance, coef, intercept):
        #     a = np.exp(coef * distance + intercept)
        #     return a
        #
        # distance_data['exp_term'] = distance_data['Distance'].apply(dist_regression, args=(coef,))
        # distance_data['time_exp'] = distance_data['exp_term'] * distance_data['time'] * distance_data['capacity']
        # distance_data['H2_exp_term'] = distance_data['Distance'].apply(dist_regression, args=(coef=-1,intercept=0))
        #
        # intercept = self.step1_intercept
        # coef = self.step1_coef[0]

        # 2. Vectorized calculation for 'exp_term'
        # This calculates exp(coef * distance + intercept) for the whole column at once
        distance_data['exp_term'] = np.exp((distance_data['Distance'] * coef) + intercept)

        # 3. Calculate 'time_exp' (This line was already correct)
        distance_data['time_exp'] =  distance_data['exp_term'] * distance_data['time'] * distance_data['capacity']

        # 4. Calculate 'H2_exp_term' with custom coefficients
        # Note: You used coef=-1 and intercept=0.
        # Mathematically, e^(-1 * d + 0) simplifies to e^(-d)
        distance_data['H2_exp_term'] = np.exp(distance_data['Distance'] * -1)
        distance_data['H2_time_exp'] = distance_data['H2_exp_term'] * distance_data['time'] * distance_data['capacity']
        distance_data['H1_time'] = distance_data['time'] * distance_data['capacity']

        self.distance_data = distance_data

        # print(distance_data)

        cols_to_sum = ['time_exp', 'H2_time_exp', 'H1_time']

        # # compute the time effect
        # grouped_df = distance_data.groupby(['County', 'Year'])
        # sum_exp_df = grouped_df['time_exp'].sum()
        sum_exp_df = distance_data.groupby(['County', 'Year'])[cols_to_sum].sum().reset_index()

        # 4. Rename columns to match your requirements
        sum_exp_df.rename(columns={
            'H2_time_exp': 'H2',
            'H1_time': 'H1'
        }, inplace=True)

        # Final Result
        # print(sum_exp_df)

        education_cols = ['County', 'Year', 'Education']
        education_types = {'County': str, 'Year': float, 'Education': float}
        self.education_df = pd.read_csv('data/17-22edu.csv', usecols=education_cols, dtype=education_types)

        # load gdp
        self.gdp_df = pd.read_csv('data/GDP.csv')

        results_df = sum_exp_df.reset_index()
        results_df = results_df.merge(ev_registration_df, on=['County', 'Year'], how='left').merge(self.education_df,
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

        self.step2_data = results_df

    def one_step_regression(self):
        # 1. 预处理数据
        whole_dist_df = self.distance_data.copy().reset_index(drop=True)
        dist_df = whole_dist_df[whole_dist_df['Year'] <= 2020]
        whole_reg_data = self.step2_data.copy().reset_index(drop=True)
        reg_data = whole_reg_data[whole_reg_data['Year'] <= 2020]
        print(reg_data)

        m = gp.Model('one_step_regression')
        m.setParam('NonConvex', 2)
        m.setParam('TimeLimit', 18000)
        m.setParam('MIPGap', 0.001)

        # --- 定义变量 ---
        dep_intercept = m.addVar(lb=-GRB.INFINITY, name='dep_intercept')
        beta_distance = m.addVar(lb=-GRB.INFINITY, name='beta_distance')
        beta_education = m.addVar(lb=-GRB.INFINITY, name='beta_education')
        beta_gdp = m.addVar(lb=-GRB.INFINITY, name='beta_gdp')
        intercept = m.addVar(lb=-GRB.INFINITY, name='intercept')

        # 3. 处理 distance_data 中的非线性项
        num_dist_rows = len(dist_df)
        dist_expr = m.addVars(num_dist_rows, lb=-GRB.INFINITY, name='dist_expr')
        dist_exp_val = m.addVars(num_dist_rows, lb=0, name='dist_exp_val')
        time_cap_exp = m.addVars(num_dist_rows, lb=-GRB.INFINITY, name='time_cap_exp')

        for i in range(num_dist_rows):
            # dist_expr = beta * distance + intercept
            m.addConstr(dist_expr[i] == beta_distance * float(dist_df.iloc[i]['Distance']) + dep_intercept)
            # dist_exp_val = exp(dist_expr)
            m.addGenConstrExp(dist_expr[i], dist_exp_val[i])
            # 考虑 time 和 capacity 的项
            m.addConstr(time_cap_exp[i] == dist_exp_val[i] * float(dist_df.iloc[i]['time']) * float(
                dist_df.iloc[i]['capacity']))

        # 4. 按 County-Year 分组求和
        grouped_indices = dist_df.groupby(['County', 'Year']).indices
        group_sums = {}
        for (county, year), indices in grouped_indices.items():
            group_sums[(county, year)] = gp.quicksum(time_cap_exp[idx] for idx in indices)

        # 5. 回归约束
        num_reg_rows = len(reg_data)
        est_demand = m.addVars(num_reg_rows, lb=-GRB.INFINITY, name='est_demand')
        error = m.addVars(num_reg_rows, lb=-GRB.INFINITY, name='error')

        for i in range(num_reg_rows):
            row = reg_data.iloc[i]
            c_y = (row['County'], row['Year'])
            current_sum_exp = group_sums.get(c_y, 0)

            # 线性回归方程
            m.addConstr(
                est_demand[i] == intercept +
                beta_education * float(row['normalized_education_level']) +
                beta_gdp * float(row['normalized_gdp']) +
                current_sum_exp
            )
            # 残差: y_true - y_pred
            m.addConstr(error[i] == float(row['EV Original Registrations']) - est_demand[i])

        # 6. 目标函数: MSE
        m.setObjective(gp.quicksum(error[i] * error[i] for i in range(num_reg_rows)) / num_reg_rows, GRB.MINIMIZE)

        m.optimize()

        # --- 结果提取与手动重算 ---
        if m.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT] and m.solCount > 0:
            # 获取最优系数
            coefs = {
                "beta_distance": beta_distance.X,
                "dep_intercept": dep_intercept.X,
                "beta_education": beta_education.X,
                "beta_gdp": beta_gdp.X,
                "intercept": intercept.X
            }

            self.gurobi_coefs = coefs

            # --- 手动重新计算训练集指标 (不依赖 Gurobi ObjVal) ---
            # 1. 计算每个观测点的 exp 项
            # --- 手动重新计算训练集指标 ---
            # 1. 先在 whole_dist_df 上计算完整列
            whole_dist_df['calc_exp'] = np.exp(
                whole_dist_df['Distance'] * coefs['beta_distance'] + coefs['dep_intercept'])
            # 注意：这里全部使用 whole_dist_df 的列，不要混用 dist_df
            whole_dist_df['calc_time_cap_exp'] = whole_dist_df['calc_exp'] * whole_dist_df['time'] * whole_dist_df[
                'capacity']

            # 2. 计算【所有年份】的分组和（这样后面 test_eval_df 才有数据）
            # 使用 whole_dist_df 而不是 dist_df
            all_sum_features = whole_dist_df.groupby(['County', 'Year'])['calc_time_cap_exp'].sum().reset_index()

            # 3. 合并到回归数据中
            # 使用 all_sum_features 进行 merge
            eval_df = whole_reg_data.merge(all_sum_features, on=['County', 'Year'], how='left').fillna(0)
            eval_df['y_pred'] = (
                    coefs['intercept'] +
                    coefs['beta_education'] * eval_df['normalized_education_level'] +
                    coefs['beta_gdp'] * eval_df['normalized_gdp'] +
                    eval_df['calc_time_cap_exp']
            )

            # 4. 计算残差和 MSE
            eval_df['residual'] = eval_df['EV Original Registrations'] - eval_df['y_pred']
            manual_mse = (eval_df['residual'] ** 2).mean()
            # separate into train and test
            train_eval_df = eval_df[eval_df['Year'] <= 2020]
            test_eval_df = eval_df[eval_df['Year'] > 2020]
            train_mse = (train_eval_df['residual'] ** 2).mean()
            train_mae = mean_absolute_error(train_eval_df['EV Original Registrations'], train_eval_df['y_pred'])
            train_me = max_error(train_eval_df['EV Original Registrations'], train_eval_df['y_pred'])
            train_r2 = r2_score(train_eval_df['EV Original Registrations'], train_eval_df['y_pred'])
            test_mse = (test_eval_df['residual'] ** 2).mean()
            test_mae = mean_absolute_error(test_eval_df['EV Original Registrations'], test_eval_df['y_pred'])
            test_me = max_error(test_eval_df['EV Original Registrations'], test_eval_df['y_pred'])
            test_r2 = r2_score(test_eval_df['EV Original Registrations'], test_eval_df['y_pred'])

            self.gurobi_df = eval_df

            return {
                "coefficients": coefs,
                "train_mse": train_mse,
                "test_mse": test_mse,
                "residuals": eval_df['residual'].tolist(),
                "status": m.status,
                "gurobi_obj": m.objVal,
                "mipgap": m.MIPGap,
                "train_mae": train_mae,
                "train_me": train_me,
                "train_r2": train_r2,
                "test_mae": test_mae,
                "test_me": test_me,
                "test_r2": test_r2
            }
        else:
            return {"status": m.status, "msg": "No feasible solution found"}

    def linear_gurobi_experiments(self, residuals_dep, step1_coef, step1_intercept, step2_intercept, education_coef,
                                  gdp_coef, experiments_type, ):
        # Generate random values
        r = self.revenue
        f = self.fixedcost
        c = self.experiments_distance * self.transportationcost
        chargercost = self.chargercost
        C = self.capacity
        p = self.penaltycost
        num_potential = self.num_exp_Facilities
        num_zip = self.num_exp_Customers
        # Select and fit the model
        residuals = list(residuals_dep)
        education_array = np.array(self.education)  # Convert education list to array
        gdp_array = np.array(self.gdp)  # Convert GDP list to array

        # Stack the arrays into a 2D array shape (2, num_customers)
        X = np.vstack((education_array, gdp_array))

        # Calculate predicted demand for each customer
        # self.step2_dep_coef has shape (2,) for [education_coef, gdp_coef]
        predicted_demands = education_coef * education_array + gdp_coef * gdp_array + step2_intercept  # This gives an array of predictions, one for each customer

        # Create initial demand array with base demand + predicted effects
        initial_demand = 15000 + predicted_demands

        # If you need it as a list:
        initial_demand = initial_demand.tolist()
        orig_demand = pd.DataFrame(np.exp(step1_coef * self.experiments_distance + step1_intercept))
        # orig_demand = pd.DataFrame(np.exp(-0.4 * self.experiments_distance + 4))
        self.orig_demand = orig_demand
        sce = len(residuals)
        exp_demand = {(i, j, k): orig_demand.iloc[i, j] for i in range(num_potential) for j in range(num_zip) for k in
                      range(sce)}
        prob = np.array([1.0 / sce] * sce)

        if experiments_type == 'insample':
            # Model
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

        if experiments_type == 'oos':
            # Model
            m = gp.Model('nonparametric_dep_oos')
            m.setParam('MIPGap', 0.01)

            decision = self.dep_decision

            noise = self.test_noise

            # Reuse exponential term from step1
            exponential_term = np.exp(self.dep_alpha1 + self.experiments_distance.to_numpy() * self.dep_beta_distance)

            # Repeat the decision list for test_sizeData rows
            decision_df = pd.DataFrame([decision] * self.test_sizeData)

            demand_contrib = decision_df.to_numpy() @ exponential_term  # shape: (test_sizeData, exp_numCustomers)

            '''
            # Calculate the test demand by adding intercept, education, GDP, and noise
            testDemand = demand_contrib + 15000 + \
                         self.education * self.dep_beta_education + \
                         self.gdp * self.dep_beta_gdp + \
                         self.dep_intercept + noise

            '''

            testDemand = np.zeros((self.test_sizeData, self.num_exp_Customers))
            for j in range(self.num_exp_Customers):
                testDemand[:, j] = (demand_contrib[:, j] + 15000 +
                                    self.education[j] * self.dep_beta_education +
                                    self.gdp[j] * self.dep_beta_gdp +
                                    self.dep_intercept + noise[:, j])

            self.dep_testDemand = testDemand

            sce = self.test_sizeData
            prob = np.array([1.0 / sce] * sce)  # Probability for each scenario

            # Variables
            charger = m.addVars(num_potential, vtype=GRB.INTEGER, ub=10, name='Charger')
            location = m.addVars(num_potential, vtype=GRB.BINARY, name='Location')
            assign = m.addVars(num_potential, num_zip, sce, lb=0.0, vtype=GRB.CONTINUOUS, name='Assign')
            unmetdemand = m.addVars(num_zip, sce, lb=0.0, vtype=GRB.CONTINUOUS, name='Unmet')

            # Constraints
            m.addConstrs((10 * location[i] >= charger[i] for i in range(num_potential)), name='Decision')
            m.addConstrs((charger[i] == decision[i] for i in range(num_potential)), name='Location')
            m.addConstrs(
                (
                    gp.quicksum(assign[(i, j, k)] for i in range(num_potential)) + unmetdemand[(j, k)] == testDemand[
                        (k, j)]
                    for j in range(num_zip) for k in range(sce)), name='Demand')
            m.addConstrs(
                (gp.quicksum(assign[(i, j, k)] for j in range(num_zip)) <= C[i] * charger[i] for i in
                 range(num_potential)
                 for k in range(sce)), name='Capacity')

            # Objective function components
            fixed_cost = location.prod(f)
            charger_cost = charger.prod(chargercost)
            transportation_cost = gp.quicksum(assign[(i, j, k)] * c.iloc[i, j] * prob[k] for i in range(num_potential)
                                              for j in range(num_zip) for k in range(sce))
            penalty_cost = gp.quicksum(unmetdemand[j, k] * p[j] * prob[k] for j in range(num_zip) for k in range(sce))
            revenue = gp.quicksum(r[j] * testDemand[k, j] * prob[k] for j in range(num_zip) for k in range(sce))

            # Set the objective function
            m.setObjective(fixed_cost + charger_cost + transportation_cost + penalty_cost - revenue, GRB.MINIMIZE)
            m.optimize()

            if m.status == GRB.OPTIMAL:
                # Extract decision variable values
                location_values = [charger[i].X for i in range(num_potential)]
                fixed_cost = sum(location[i].X * f[i] for i in range(num_potential))
                print('fixed cost oos deo', fixed_cost)
                charger_cost = sum(charger[i].X * chargercost[i] for i in range(num_potential))
                print('charger cost oos dep', charger_cost)
                optimal_value = m.objVal
                revenue_value = sum(r[j] * testDemand[(k, j)] * prob[k] for j in range(num_zip) for k in range(sce))
                penalty_cost = sum(
                    unmetdemand[j, k].X * p[j] * prob[k] for j in range(num_zip) for k in range(sce))
                real_demand_values = testDemand
                print('penalty cost oos dep', penalty_cost)
                transportation_cost = sum(
                    assign[(i, j, k)].X * c.iloc[i, j] * prob[k] for i in range(num_potential)
                    for j in range(num_zip) for k in range(sce))
                print('transportation cost oos dep', transportation_cost)
                return location_values, optimal_value, revenue_value, testDemand

    def nonparametric_regression_tuning(self, features, feature_type):
        train_data = self.step2_data[self.step2_data['Year'] <= 2020]
        test_data = self.step2_data[self.step2_data['Year'] >= 2021]
        # Prepare features and response
        x = train_data[features].values
        y = train_data['EV Original Registrations'].values

        # Define models including Neural Network
        # # models = {
        # #     'Decision Tree': DecisionTreeRegressor(),
        # #     'Random Forest': RandomForestRegressor(),
        # #     'Gradient Boosting': HistGradientBoostingRegressor(),
        # #     'kNN': KNeighborsRegressor(),
        #     'Neural Network': MLPRegressor(
        #         max_iter=5000,
        #         early_stopping=True,
        #         n_iter_no_change=50,
        #         validation_fraction=0.1,
        #         random_state=42
        #     )
        # }

        models = {
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
            'Neural Network': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu'],
                'solver': ['adam', 'lbfgs'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['adaptive'],
                'learning_rate_init': [0.001, 0.01],
            }
        }

        # param_grids = {
        #     # 'Decision Tree': {'max_depth': range(3, 10), 'min_samples_split': range(2, 10),
        #     #                   'random_state': range(40, 50)},
        #     # 'Random Forest': {'random_state': range(40, 50), 'n_estimators': [50, 100, 150], 'max_depth': range(3, 10),
        #     #                   'min_samples_split': range(2, 10)},
        #     # 'Gradient Boosting': {'max_depth': range(3, 10), 'learning_rate': [0.01, 0.1, 0.2]},
        #     # 'kNN': {'n_neighbors': range(3, 10)},
        #     'Neural Network': {
        #         'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        #         'activation': ['relu'],
        #         'solver': ['adam', 'lbfgs'],
        #         'alpha': [0.0001, 0.001, 0.01],
        #         'learning_rate': ['adaptive'],
        #         'learning_rate_init': [0.001, 0.01],
        #     }
        # }

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
        test_y = (test_data['EV Original Registrations']).values

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
        model_names = ['Neural Network', 'Decision Tree',
            'Random Forest',
            'Gradient Boosting',
            'kNN']
        train_mse_results = {model: [] for model in model_names}
        test_mse_results = {model: [] for model in model_names}
        train_mae_results = {model: [] for model in model_names}
        test_mae_results = {model: [] for model in model_names}
        train_r2_results = {model: [] for model in model_names}
        test_r2_results = {model: [] for model in model_names}
        runtime_results = {model: [] for model in model_names}  # ✅ Individual model runtimes
        best_params_results = {model: [] for model in model_names}


        feature_list = [
            ['normalized_education_level', 'normalized_gdp', 'H1'],
            ['normalized_education_level', 'normalized_gdp', 'H2'],
            ['normalized_education_level', 'normalized_gdp', 'time_exp'],
        ]

        # Map feature_list to 4 feature type H1, H2, H3, H4
        feature_type_map = ['H1', 'H2', 'H3']

        print(f"\n{'=' * 80}")
        print("Starting Nonparametric Regression Training...")
        print(f"{'=' * 80}")

        nonparametric_results_all = pd.DataFrame()

        # ✅ Loop through all 4 feature versions
        for idx, features in enumerate(feature_list):
            feature_type = feature_type_map[idx]
            print(f"\n{'=' * 80}")
            print(f"Training with Feature Set {feature_type}: {features}")
            print(f"{'=' * 80}")

            nonparametric_df = self.nonparametric_regression_tuning(features, feature_type)
            # ✅ Add Feature_Type column to identify which feature set
            nonparametric_df['Feature_Type'] = feature_type

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
                    runtime_results[model_name].append(results['runtime'])  # ✅ Individual runtime
                    best_params_results[model_name].append(str(results['best_params']))
                else:
                    # If model not found, append NaN
                    train_mse_results[model_name].append(np.nan)
                    test_mse_results[model_name].append(np.nan)
                    train_mae_results[model_name].append(np.nan)
                    test_mae_results[model_name].append(np.nan)
                    train_r2_results[model_name].append(np.nan)
                    test_r2_results[model_name].append(np.nan)
                    runtime_results[model_name].append(np.nan)
                    best_params_results[model_name].append(None)

            # ✅ Concatenate results for all feature versions
            nonparametric_results_all = pd.concat([nonparametric_results_all, nonparametric_df], ignore_index=True)
            # # save nonparametric_df to csv
            # output_file = os.path.join(output_dir, f"nonparametric_results_seed{seed}_featureset{feature_type}.csv")
            # nonparametric_df.to_csv(output_file, index=False)
            # print(f"Saved nonparametric results to {output_file}")

        return nonparametric_results_all

    def step2_linear_regression(self):
        training_data = self.step2_data[self.step2_data['Year'] <= 2020]
        test_data = self.step2_data[self.step2_data['Year'] >= 2021]

        self.step2_Y_train_dep = training_data['EV Original Registrations'] - training_data['time_exp']
        self.step2_Y_train_ind = training_data['EV Original Registrations']
        self.step2_X_train = training_data[['normalized_education_level', 'normalized_gdp']]

        self.test_Y_dep = test_data['EV Original Registrations'] - test_data['time_exp']
        self.test_Y_ind = test_data['EV Original Registrations']
        self.test_X = test_data[['normalized_education_level', 'normalized_gdp']]

        self.step2_n_train = len(self.step2_Y_train_dep)  # Number of observations in training
        self.step2_n_test = len(self.test_Y_dep)  # Number of observations in test

        # Fit Training dataset for decision-dependent
        dep_start_time = time.time()
        dep_model = LinearRegression()
        dep_model.fit(self.step2_X_train, self.step2_Y_train_dep)
        step2_dep_pred = dep_model.predict(self.step2_X_train)
        dep_run_time = time.time() - dep_start_time
        self.step2_dep_time = dep_run_time
        self.step2_dep_intercept = dep_model.intercept_
        self.step2_dep_coef = dep_model.coef_
        self.step2_dep_r2 = r2_score(self.step2_Y_train_dep, step2_dep_pred)
        self.step2_dep_mse = mean_squared_error(self.step2_Y_train_dep, step2_dep_pred)
        self.step2_dep_mae = mean_absolute_error(self.step2_Y_train_dep, step2_dep_pred)
        self.step2_dep_median_ae = median_absolute_error(self.step2_Y_train_dep, step2_dep_pred)
        self.step2_dep_max_error = max_error(self.step2_Y_train_dep, step2_dep_pred)
        self.step2_dep_residuals = self.step2_Y_train_dep - step2_dep_pred

        # test1 results for decision-dependent
        step2_test_dep_pred = dep_model.predict(self.test_X)
        self.step2_dep_r2_test = r2_score(self.test_Y_dep, step2_test_dep_pred)
        self.step2_dep_mse_test = mean_squared_error(self.test_Y_dep, step2_test_dep_pred)
        self.step2_dep_mae_test = mean_absolute_error(self.test_Y_dep, step2_test_dep_pred)
        self.step2_dep_median_ae_test = median_absolute_error(self.test_Y_dep, step2_test_dep_pred)
        self.step2_dep_max_error_test = max_error(self.test_Y_dep, step2_test_dep_pred)
        self.step2_dep_residuals_test = self.test_Y_dep - step2_test_dep_pred

        ind_start_time = time.time()
        # Fit Training dataset for decision-independent
        ind_model = LinearRegression()
        ind_model.fit(self.step2_X_train, self.step2_Y_train_ind)
        step2_ind_pred = ind_model.predict(self.step2_X_train)
        ind_run_time = time.time() - ind_start_time
        self.step2_ind_time = ind_run_time
        self.step2_ind_intercept = ind_model.intercept_
        self.step2_ind_coef = ind_model.coef_
        self.step2_ind_mse = mean_squared_error(self.step2_Y_train_ind, step2_ind_pred)
        self.step2_ind_mae = mean_absolute_error(self.step2_Y_train_ind, step2_ind_pred)
        self.step2_ind_median_ae = median_absolute_error(self.step2_Y_train_ind, step2_ind_pred)
        self.step2_ind_max_error = max_error(self.step2_Y_train_ind, step2_ind_pred)
        self.step2_ind_residuals = self.step2_Y_train_ind - step2_ind_pred
        self.step2_ind_r2 = r2_score(self.step2_Y_train_ind, step2_ind_pred)

        # test1 results for decision-independent
        step2_test_ind_pred = ind_model.predict(self.test_X)
        self.step2_ind_r2_test = r2_score(self.test_Y_ind, step2_test_ind_pred)
        self.step2_ind_mse_test = mean_squared_error(self.test_Y_ind, step2_test_ind_pred)
        self.step2_ind_mae_test = mean_absolute_error(self.test_Y_ind, step2_test_ind_pred)
        self.step2_ind_median_ae_test = median_absolute_error(self.test_Y_ind, step2_test_ind_pred)
        self.step2_ind_max_error_test = max_error(self.test_Y_ind, step2_test_ind_pred)
        self.step2_ind_residuals_test = self.test_Y_ind - step2_test_ind_pred

        ### Store results in DataFrame
        regression_results = pd.DataFrame({
            'Model': ['Decision-Dependent', 'Decision-Independent'],
            'MSE_Train': [self.step2_dep_mse, self.step2_ind_mse],
            'MAE_Train': [self.step2_dep_mae, self.step2_ind_mae],
            'Median_AE_Train': [self.step2_dep_median_ae, self.step2_ind_median_ae],
            'Max_Error_Train': [self.step2_dep_max_error, self.step2_ind_max_error],
            'MSE_Test': [self.step2_dep_mse_test, self.step2_ind_mse_test],
            'MAE_Test': [self.step2_dep_mae_test, self.step2_ind_mae_test],
            'Median_AE_Test': [self.step2_dep_median_ae_test, self.step2_ind_median_ae_test],
            'Max_Error_Test': [self.step2_dep_max_error_test, self.step2_ind_max_error_test],
            'R2_train': [self.step2_dep_r2, self.step2_ind_r2],
            'R2_test': [self.step2_dep_r2_test, self.step2_ind_r2_test]
        })

        # Create a DataFrame with the coefficients and intercepts in two rows
        results_coef = pd.DataFrame({
            'Model': ['Dependent', 'Independent'],
            'Intercept': [self.step2_dep_intercept, self.step2_ind_intercept],
            'Coef 1': [self.step2_dep_coef[0], self.step2_ind_coef[0]],
            'Coef 2': [self.step2_dep_coef[1], self.step2_ind_coef[1]]
        })

        # # Plotting for Decision-Dependent
        # plt.figure(figsize=(18, 6))
        #
        # # Plot 1: Education vs. Adjusted Demand Increase
        # plt.subplot(1, 3, 1)
        # plt.scatter(self.step2_X_train['normalized_education_level'], self.step2_Y_train_dep, alpha=0.7,
        #             label='Training Data')
        # plt.scatter(self.test_X['normalized_education_level'], self.test_Y_dep, color='red', label='Test Data')
        # x = np.linspace(0, 1, 100)
        # plt.plot(x, self.step2_dep_intercept + self.step2_dep_coef[0] * x, color='Orange', linestyle='-', linewidth=2)
        # plt.xlabel('Normalized Education Level')
        # plt.ylabel('Adjusted Demand Increase')
        # plt.legend()
        # plt.grid(True, linestyle='--', linewidth=0.5)
        #
        # # Plot 2: GDP vs. Adjusted Demand Increase
        # plt.subplot(1, 3, 2)
        # plt.scatter(self.step2_X_train['normalized_gdp'], self.step2_Y_train_dep, alpha=0.7, label='Training Data')
        # plt.scatter(self.test_X['normalized_gdp'], self.test_Y_dep, color='red', label='Test Data')
        # plt.plot(x, self.step2_dep_intercept + self.step2_dep_coef[1] * x, color='Orange', linestyle='-', linewidth=2)
        # plt.xlabel('Normalized GDP')
        # plt.legend()
        # plt.grid(True, linestyle='--', linewidth=0.5)
        #
        # # Plot 3: Histogram of Residuals for Decision-Dependent Model
        # plt.subplot(1, 3, 3)
        # plt.hist(self.step2_dep_residuals, bins=16, color='navy', edgecolor='black', alpha=0.7)
        # plt.xlabel('Residuals')
        # plt.ylabel('Frequency')
        # plt.grid(True, linestyle='--', linewidth=0.5)
        #
        # # Adjust layout to prevent overlap
        # plt.tight_layout()
        # plt.savefig('step2_dep.png', dpi=300, bbox_inches='tight')
        # plt.show()
        #
        # # Plotting for Decision-Independent
        # plt.figure(figsize=(18, 6))
        #
        # # Plot 1: Education vs. EV Original Registrations
        # plt.subplot(1, 3, 1)
        # plt.scatter(self.step2_X_train['normalized_education_level'], self.step2_Y_train_ind, alpha=0.7,
        #             label='Training Data')
        # plt.scatter(self.test_X['normalized_education_level'], self.test_Y_ind, color='red', label='Test Data')
        # plt.plot(x, self.step2_ind_intercept + self.step2_ind_coef[0] * x, color='Orange', linestyle='-', linewidth=2)
        # plt.xlabel('Normalized Education Level')
        # plt.ylabel('EV Original Registrations')
        # plt.legend()
        # plt.grid(True, linestyle='--', linewidth=0.5)
        #
        # # Plot 2: GDP vs. EV Original Registrations
        # plt.subplot(1, 3, 2)
        # plt.scatter(self.step2_X_train['normalized_gdp'], self.step2_Y_train_ind, alpha=0.7, label='Training Data')
        # plt.scatter(self.test_X['normalized_gdp'], self.test_Y_ind, color='red', label='Test Data')
        # plt.plot(x, self.step2_ind_intercept + self.step2_ind_coef[1] * x, color='Orange', linestyle='-', linewidth=2)
        # plt.xlabel('Normalized GDP')
        # plt.legend()
        # plt.grid(True, linestyle='--', linewidth=0.5)
        #
        # # Plot 3: Histogram of Residuals for Decision-Independent Model
        # plt.subplot(1, 3, 3)
        # plt.hist(self.step2_ind_residuals, bins=16, color='navy', edgecolor='black', alpha=0.7)
        # plt.xlabel('Residuals')
        # plt.ylabel('Frequency')
        # plt.grid(True, linestyle='--', linewidth=0.5)
        #
        # # Adjust layout to prevent overlap
        # plt.tight_layout()
        # plt.savefig('step2_ind.png', dpi=300, bbox_inches='tight')
        # plt.show()

        return regression_results, results_coef

    def plot_3d_multivariate(self):
        # Create the plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Define the independent variables (X1: Education, X2: GDP) and dependent variable (Y: Adjusted Demand Increase)
        x = self.step2_X_train['normalized_education_level']
        y = self.step2_X_train['normalized_gdp']
        z = self.step2_Y_train_dep

        x_test = self.test_X['normalized_education_level']
        y_test = self.test_X['normalized_gdp']
        z_test = self.test_Y_dep

        # Add the data points
        ax.scatter(x, y, z, c='blue', label='Training Data', alpha=0.7)
        ax.scatter(x_test, y_test, z_test, c='red', label='Testing Data', alpha=0.7)

        # Fit a regression plane using np.linalg.lstsq
        A = np.vstack([x, y]).T  # No intercept term
        plane_coef, _, _, _ = np.linalg.lstsq(A, z, rcond=None)

        # Create a meshgrid for the regression plane
        x_plane, y_plane = np.meshgrid(np.linspace(min(x), max(x), 50), np.linspace(min(y), max(y), 50))
        z_plane = plane_coef[0] * x_plane + plane_coef[1] * y_plane  # No intercept

        # Add the regression plane
        ax.plot_surface(x_plane, y_plane, z_plane, alpha=0.5, color='orange')

        # Add labels and title
        ax.set_xlabel('Normalized Education Level', fontsize=16, labelpad=15)
        ax.set_ylabel('Normalized GDP', fontsize=16, labelpad=15)
        ax.set_zlabel(r'$\tilde{d}$', fontsize=16, labelpad=15)
        ax.legend(fontsize=18)

        # Adjust the space between the ticks and labels for all axes
        ax.tick_params(axis='x', pad=5)  # Increase space between x ticks and labels
        ax.tick_params(axis='y', pad=5)  # Increase space between y ticks and labels
        ax.tick_params(axis='z', pad=5)  # Increase space between z ticks and labels

        ax.view_init(azim=-75, elev=10)

        # Show the plot
        plt.show()

        # Create the plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Define the independent variables (X1: Education, X2: GDP) and dependent variable (Y: Adjusted Demand Increase)
        x = self.step2_X_train['normalized_education_level']
        y = self.step2_X_train['normalized_gdp']
        z = self.step2_Y_train_dep

        x_test = self.test_X['normalized_education_level']
        y_test = self.test_X['normalized_gdp']
        z_test = self.test_Y_dep

        # Add the data points
        ax.scatter(x, y, z, c='blue', label='Training Data', alpha=0.7)
        ax.scatter(x_test, y_test, z_test, c='red', label='Testing Data', alpha=0.7)

        # Fit a regression plane using np.linalg.lstsq
        A = np.vstack([x, y]).T  # No intercept term
        plane_coef, _, _, _ = np.linalg.lstsq(A, z, rcond=None)

        # Create a meshgrid for the regression plane
        x_plane, y_plane = np.meshgrid(np.linspace(min(x), max(x), 50), np.linspace(min(y), max(y), 50))
        z_plane = plane_coef[0] * x_plane + plane_coef[1] * y_plane  # No intercept

        # Add the regression plane
        ax.plot_surface(x_plane, y_plane, z_plane, alpha=0.5, color='orange')

        # Add labels and title
        ax.set_xlabel('Normalized Education Level', fontsize=16, labelpad=15)
        ax.set_ylabel('Normalized GDP', fontsize=16, labelpad=15)
        ax.set_zlabel(r'$\tilde{d}$', fontsize=16, labelpad=15)
        ax.legend(fontsize=18)

        # Adjust the space between the ticks and labels for all axes
        ax.tick_params(axis='x', pad=5)  # Increase space between x ticks and labels
        ax.tick_params(axis='y', pad=5)  # Increase space between y ticks and labels
        ax.tick_params(axis='z', pad=5)  # Increase space between z ticks and labels

        ax.view_init(azim=-50, elev=10)

        # Show the plot
        plt.show()

        # Create the plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Define the independent variables (X1: Education, X2: GDP) and dependent variable (Y: Adjusted Demand Increase)
        x = self.step2_X_train['normalized_education_level']
        y = self.step2_X_train['normalized_gdp']
        z = self.step2_Y_train_dep

        x_test = self.test_X['normalized_education_level']
        y_test = self.test_X['normalized_gdp']
        z_test = self.test_Y_dep

        # Add the data points
        ax.scatter(x, y, z, c='blue', label='Training Data', alpha=0.7)
        ax.scatter(x_test, y_test, z_test, c='red', label='Testing Data', alpha=0.7)

        # Fit a regression plane using np.linalg.lstsq
        A = np.vstack([x, y]).T  # No intercept term
        plane_coef, _, _, _ = np.linalg.lstsq(A, z, rcond=None)

        # Create a meshgrid for the regression plane
        x_plane, y_plane = np.meshgrid(np.linspace(min(x), max(x), 50), np.linspace(min(y), max(y), 50))
        z_plane = plane_coef[0] * x_plane + plane_coef[1] * y_plane  # No intercept

        # Add the regression plane
        ax.plot_surface(x_plane, y_plane, z_plane, alpha=0.5, color='orange')

        # Add labels and title
        ax.set_xlabel('Normalized Education Level', fontsize=16, labelpad=15)
        ax.set_ylabel('Normalized GDP', fontsize=16, labelpad=15)
        ax.set_zlabel(r'$\tilde{d}$', fontsize=16, labelpad=15)
        ax.legend(fontsize=18)

        # Adjust the space between the ticks and labels for all axes
        ax.tick_params(axis='x', pad=5)  # Increase space between x ticks and labels
        ax.tick_params(axis='y', pad=5)  # Increase space between y ticks and labels
        ax.tick_params(axis='z', pad=5)  # Increase space between z ticks and labels

        ax.view_init(azim=-25, elev=10)

        # Show the plot
        plt.show()

        # Create the plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Define the independent variables (X1: Education, X2: GDP) and dependent variable (Y: Adjusted Demand Increase)
        x = self.step2_X_train['normalized_education_level']
        y = self.step2_X_train['normalized_gdp']
        z = self.step2_Y_train_ind

        x_test = self.test_X['normalized_education_level']
        y_test = self.test_X['normalized_gdp']
        z_test = self.test_Y_ind

        # Add the data points
        ax.scatter(x, y, z, c='blue', label='Training Data', alpha=0.7)
        ax.scatter(x_test, y_test, z_test, c='red', label='Testing Data', alpha=0.7)

        # Fit a regression plane using np.linalg.lstsq
        A = np.vstack([x, y]).T  # No intercept term
        plane_coef, _, _, _ = np.linalg.lstsq(A, z, rcond=None)

        # Create a meshgrid for the regression plane
        x_plane, y_plane = np.meshgrid(np.linspace(min(x), max(x), 50), np.linspace(min(y), max(y), 50))
        z_plane = plane_coef[0] * x_plane + plane_coef[1] * y_plane  # No intercept

        # Add the regression plane
        ax.plot_surface(x_plane, y_plane, z_plane, alpha=0.5, color='orange')

        # Add labels and title
        ax.set_xlabel('Normalized Education Level', fontsize=16, labelpad=15)
        ax.set_ylabel('Normalized GDP', fontsize=16, labelpad=15)
        ax.set_zlabel(r'$\tilde{d}$', fontsize=16, labelpad=15)
        ax.legend(fontsize=18)

        # Adjust the space between the ticks and labels for all axes
        ax.tick_params(axis='x', pad=5)  # Increase space between x ticks and labels
        ax.tick_params(axis='y', pad=5)  # Increase space between y ticks and labels
        ax.tick_params(axis='z', pad=5)  # Increase space between z ticks and labels

        ax.view_init(azim=-75, elev=10)

        # Show the plot
        plt.show()

        # Create the plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Define the independent variables (X1: Education, X2: GDP) and dependent variable (Y: Adjusted Demand Increase)
        x = self.step2_X_train['normalized_education_level']
        y = self.step2_X_train['normalized_gdp']
        z = self.step2_Y_train_ind

        x_test = self.test_X['normalized_education_level']
        y_test = self.test_X['normalized_gdp']
        z_test = self.test_Y_ind

        # Add the data points
        ax.scatter(x, y, z, c='blue', label='Training Data', alpha=0.7)
        ax.scatter(x_test, y_test, z_test, c='red', label='Testing Data', alpha=0.7)

        # Fit a regression plane using np.linalg.lstsq
        A = np.vstack([x, y]).T  # No intercept term
        plane_coef, _, _, _ = np.linalg.lstsq(A, z, rcond=None)

        # Create a meshgrid for the regression plane
        x_plane, y_plane = np.meshgrid(np.linspace(min(x), max(x), 50), np.linspace(min(y), max(y), 50))
        z_plane = plane_coef[0] * x_plane + plane_coef[1] * y_plane  # No intercept

        # Add the regression plane
        ax.plot_surface(x_plane, y_plane, z_plane, alpha=0.5, color='orange')

        # Add labels and title
        ax.set_xlabel('Normalized Education Level', fontsize=16, labelpad=15)
        ax.set_ylabel('Normalized GDP', fontsize=16, labelpad=15)
        ax.set_zlabel(r'$\tilde{d}$', fontsize=16, labelpad=15)
        ax.legend(fontsize=18)

        # Adjust the space between the ticks and labels for all axes
        ax.tick_params(axis='x', pad=5)  # Increase space between x ticks and labels
        ax.tick_params(axis='y', pad=5)  # Increase space between y ticks and labels
        ax.tick_params(axis='z', pad=5)  # Increase space between z ticks and labels

        ax.view_init(azim=-50, elev=10)

        # Show the plot
        plt.show()

        # Create the plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Define the independent variables (X1: Education, X2: GDP) and dependent variable (Y: Adjusted Demand Increase)
        x = self.step2_X_train['normalized_education_level']
        y = self.step2_X_train['normalized_gdp']
        z = self.step2_Y_train_ind

        x_test = self.test_X['normalized_education_level']
        y_test = self.test_X['normalized_gdp']
        z_test = self.test_Y_ind

        # Add the data points
        ax.scatter(x, y, z, c='blue', label='Training Data', alpha=0.7)
        ax.scatter(x_test, y_test, z_test, c='red', label='Testing Data', alpha=0.7)

        # Fit a regression plane using np.linalg.lstsq
        A = np.vstack([x, y]).T  # No intercept term
        plane_coef, _, _, _ = np.linalg.lstsq(A, z, rcond=None)

        # Create a meshgrid for the regression plane
        x_plane, y_plane = np.meshgrid(np.linspace(min(x), max(x), 50), np.linspace(min(y), max(y), 50))
        z_plane = plane_coef[0] * x_plane + plane_coef[1] * y_plane  # No intercept

        # Add the regression plane
        ax.plot_surface(x_plane, y_plane, z_plane, alpha=0.5, color='orange')

        # Add labels and title
        ax.set_xlabel('Normalized Education Level', fontsize=16, labelpad=15)
        ax.set_ylabel('Normalized GDP', fontsize=16, labelpad=15)
        ax.set_zlabel(r'$\tilde{d}$', fontsize=16, labelpad=15)
        ax.legend(fontsize=18)

        # Adjust the space between the ticks and labels for all axes
        ax.tick_params(axis='x', pad=5)  # Increase space between x ticks and labels
        ax.tick_params(axis='y', pad=5)  # Increase space between y ticks and labels
        ax.tick_params(axis='z', pad=5)  # Increase space between z ticks and labels

        ax.view_init(azim=-25, elev=10)

        # Show the plot
        plt.show()

    def plot_residuals_and_features(self):
        # Plot histogram of residuals for both decision-dependent and decision-independent models
        plt.figure(figsize=(14, 6))

        # Residuals histogram for decision-dependent
        plt.subplot(1, 2, 1)
        plt.hist(self.step2_dep_residuals, bins=30, alpha=0.7, label="Dependent", color="blue")
        plt.title("Histogram of Decision-dependent Residuals")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")

        # Residuals histogram for decision-independent
        plt.subplot(1, 2, 2)
        plt.hist(self.step2_ind_residuals, bins=30, alpha=0.7, label="Independent", color="orange")
        plt.title("Histogram of Decision-independent Residuals")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.show()

        # Plot response vs each feature for both decision-dependent and decision-independent models
        features = ['normalized_education_level', 'normalized_gdp']

        plt.figure(figsize=(12, 5))

        # Response vs feature for decision-dependent model
        plt.subplot(1, 2, 1)
        plt.scatter(self.step2_X_train['normalized_education_level'], self.step2_Y_train_dep, color='blue',
                    label='Actual',
                    alpha=0.7)
        plt.scatter(self.step2_X_train['normalized_education_level'],
                    self.step2_dep_intercept + self.step2_dep_coef[0] * self.step2_X_train[
                        'normalized_education_level'],
                    color='red', label='Predicted', alpha=0.5)
        plt.title(f"Education vs Adjusted Registrations", fontsize=20)
        plt.xlabel('Normalized Education', fontsize=18)
        plt.ylabel("Adjusted Registrations", fontsize=18)
        plt.legend()

        # Response vs feature for decision-independent model
        plt.subplot(1, 2, 2)
        plt.scatter(self.step2_X_train['normalized_gdp'], self.step2_Y_train_dep, color='blue', label='Actual',
                    alpha=0.7)
        plt.scatter(self.step2_X_train['normalized_gdp'],
                    self.step2_dep_intercept + self.step2_dep_coef[0] * self.step2_X_train['normalized_gdp'],
                    color='red', label='Predicted', alpha=0.5)
        plt.title(f"GDP vs Adjusted Registrations", fontsize=20)
        plt.xlabel('Normalized_GDP', fontsize=18)
        plt.ylabel("Adjusted Registrations", fontsize=18)
        plt.legend()

        plt.tight_layout()
        plt.show()

        for feature in features:
            plt.figure(figsize=(12, 5))

            # Response vs feature for decision-dependent model
            plt.subplot(1, 2, 1)
            plt.scatter(self.step2_X_train[feature], self.step2_Y_train_dep, color='blue', label='Actual',
                        alpha=0.7)
            plt.scatter(self.step2_X_train[feature],
                        self.step2_dep_intercept + self.step2_dep_coef[0] * self.step2_X_train[feature],
                        color='red', label='Predicted', alpha=0.5)
            plt.title(f"Decision-dependent: {feature} vs EV Original Registrations")
            plt.xlabel(feature)
            plt.ylabel("Adjusted Registrations")
            plt.legend()

            # Response vs feature for decision-independent model
            plt.subplot(1, 2, 2)
            plt.scatter(self.step2_X_train[feature], self.step2_Y_train_ind, color='orange', label='Actual',
                        alpha=0.7)
            plt.scatter(self.step2_X_train[feature],
                        self.step2_ind_intercept + self.step2_ind_coef[0] * self.step2_X_train[feature],
                        color='green', label='Predicted', alpha=0.5)
            plt.title(f"Decision-independent: {feature} vs EV Original Registrations")
            plt.xlabel(feature)
            plt.ylabel("EV Original Registrations")
            plt.legend()

            plt.tight_layout()
            plt.show()

    def nonparametric_tuning(self):
        # Define models
        # models = {
        #     'Decision Tree': DecisionTreeRegressor(),
        #     'Random Forest': RandomForestRegressor(),
        #     'Gradient Boosting': HistGradientBoostingRegressor(),
        #     'kNN': KNeighborsRegressor(),
        #     'Neural Network': MLPRegressor(
        #         max_iter=5000,
        #         early_stopping=True,
        #         n_iter_no_change=50,
        #         validation_fraction=0.1,
        #         random_state=42
        #     )
        # }

        models = {
            'Neural Network': MLPRegressor(
                max_iter=5000,
                early_stopping=True,
                n_iter_no_change=50,
                validation_fraction=0.1,
                random_state=42
            )
        }

        # Parameter grids for hyperparameter tuning
        # param_grids = {
        #     'Decision Tree': {'max_depth': range(3, 10), 'min_samples_split': range(2, 10),
        #                       'random_state': range(40, 50)},
        #     'Random Forest': {'random_state': range(40, 50), 'n_estimators': [50, 100, 150], 'max_depth': range(3, 10),
        #                       'min_samples_split': range(2, 10)},
        #     'Gradient Boosting': {'max_depth': range(3, 10), 'learning_rate': [0.01, 0.1, 0.2]},
        #     'kNN': {'n_neighbors': range(3, 10)},
        #     'Neural Network': {
        #         'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        #         'activation': ['relu'],
        #         'solver': ['adam', 'lbfgs'],
        #         'alpha': [0.0001, 0.001, 0.01],
        #         'learning_rate': ['adaptive'],
        #         'learning_rate_init': [0.001, 0.01],
        #     }
        # }

        param_grids = {
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
        self.best_models_dep = {}
        self.best_models_ind = {}
        self.nonparametric_regression_results = {'decision_dependent': {}, 'decision_independent': {}}
        self.residuals_dep = {}
        self.residuals_ind = {}
        self.residuals_test_dep = {}
        self.residuals_test_ind = {}

        # Tune for decision-dependent
        for model_name, model in models.items():
            run_start_time = time.time()
            grid_search_dep = GridSearchCV(estimator=model, param_grid=param_grids[model_name],
                                           scoring='neg_mean_squared_error', cv=5, verbose=1)
            grid_search_dep.fit(self.step2_X_train, self.step2_Y_train_dep)
            dep_best_model = grid_search_dep.best_estimator_
            run_time = time.time() - run_start_time

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
                'max_error': max_error(self.step2_Y_train_dep, train_dep_pred),
                "run_time": run_time
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
            start_time = time.time()
            grid_search_ind = GridSearchCV(estimator=model, param_grid=param_grids[model_name],
                                           scoring='neg_mean_squared_error', cv=5, verbose=1)
            grid_search_ind.fit(self.step2_X_train, self.step2_Y_train_ind)
            ind_best_model = grid_search_ind.best_estimator_
            run_time = time.time() - start_time

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
                'max_error': max_error(self.step2_Y_train_ind, train_ind_pred),
                "run_time": run_time
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
            # add runtime
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
                dep_results['train_metrics'].get('run_time'),
                ind_results['train_metrics'].get('mse'), ind_results['test_metrics'].get('mse'),
                ind_results['train_metrics'].get('mae'), ind_results['test_metrics'].get('mae'),
                ind_results['train_metrics'].get('medae'), ind_results['test_metrics'].get('medae'),
                ind_results['train_metrics'].get('max_error'), ind_results['test_metrics'].get('max_error'),
                ind_results['train_metrics'].get('run_time')
            ])

        df_columns = ['Model Name', 'Dep_Train_MSE', 'Dep_Test_MSE', 'Dep_Train_MAE', 'Dep_Test_MAE', 'Dep_Train_MedAE',
                      'Dep_Test_MedAE',
                      'Dep_Train_MaxError', 'Dep_Test_MaxError','Dep_RunTime', 'Ind_Train_MSE', 'Ind_Test_MSE', 'Ind_Train_MAE',
                      'Ind_Test_MAE',
                      'Ind_Train_MedAE', 'Ind_Test_MedAE', 'Ind_Train_MaxError', 'Ind_Test_MaxError', 'Ind_RunTime' ]

        self.nonparametric_df = pd.DataFrame(data, columns=df_columns)

        # Save best_models_dep
        joblib.dump(self.best_models_dep, f'capacity_best_models_dep_t{self.t_threshold}.joblib')

        # Save residuals_dep
        with open(f'capacity_residuals_dep_t{self.t_threshold}.pkl', 'wb') as f:
            pickle.dump(self.residuals_dep, f)

        # Save best_models_dep
        joblib.dump(self.best_models_ind, f'capacity_best_models_ind_t{self.t_threshold}.joblib')

        # Save residuals_dep
        with open(f'capacity_residuals_ind_t{self.t_threshold}.pkl', 'wb') as f:
            pickle.dump(self.residuals_ind, f)

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
        for seed in range(1, self.numCustomers + 1):  # Use a different range of seeds
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
            num_points=self.numFacilities,
            seed=42  # Different seed for randomization
        )

        random_points_df = pd.DataFrame(random_coordinates_random, columns=['Latitude', 'Longitude']).reset_index(
            drop=True)
        self.random_points_df = random_points_df

        c1 = np.radians(selected_coordinates_df[['Latitude', 'Longitude']].to_numpy())
        c2 = np.radians(random_points_df[['Latitude', 'Longitude']].to_numpy())

        # Compute distances in miles.
        dist = haversine_distances(c2, c1) * 6371 * 0.621371
        self.experiments_distance = pd.DataFrame(dist, index=range(self.numFacilities),
                                                 columns=range(self.numCustomers))

    def plot_ev_stations_and_customers(self):
        plt.figure(figsize=(10, 6))

        # Plot potential EV charging stations from random_points_df
        plt.scatter(self.random_points_df['Latitude'], self.random_points_df['Longitude'],
                    marker='^', color='blue', label='Potential Locations')

        # Add labels to potential locations
        for i, (lat, lon) in enumerate(zip(self.random_points_df['Latitude'], self.random_points_df['Longitude']), 1):
            plt.text(lat, lon + 0.01, f'({i})', color='blue', fontsize=12, ha='center')

        # Plot customer sites from selected_coordinates_df
        plt.scatter(self.selected_coordinates_df['Latitude'], self.selected_coordinates_df['Longitude'],
                    marker='o', color='red', label='Customer Sites')

        # Add labels to customer sites
        for i, (lat, lon) in enumerate(
                zip(self.selected_coordinates_df['Latitude'], self.selected_coordinates_df['Longitude']), 1):
            plt.text(lat, lon + 0.01, f'{i}', color='red', fontsize=12, ha='center')

        # Adding labels and legend
        plt.xlabel('Latitude', fontsize=20)
        plt.ylabel('Longitude', fontsize=20)
        # plt.legend(loc='best')
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=16)
        plt.grid(True)
        plt.show()

    def generateSyntheticData(self, seed):
        # Generate random values
        random.seed(seed)
        '''
        self.revenue = [20] * self.numCustomers
        self.fixedCost = [8000] * self.numFacilities  # Fixed cost
        self.chargerCost = [400] * self.numFacilities
        self.transportationCost = self.experiments_distance * 0.005  # Transportation cost (matrix)
        self.penaltyCost = [15] * self.numCustomers  # Penalty cost
        self.capacity = [500] * self.numFacilities
        self.gdp = 0.3
        self.education = 0.2
        '''
        # Generate random values
        self.revenue = [20] * self.numCustomers
        self.fixedCost = [8000] * self.numFacilities  # Fixed cost
        self.chargerCost = [400] * self.numFacilities
        self.transportationCost = self.experiments_distance * 0.005  # Transportation cost (matrix)
        self.penaltyCost = [10] * self.numCustomers  # Penalty cost
        self.capacity = [500] * self.numFacilities
        self.gdp = 0.3
        self.education = 0.2
        # Original GDP: 37923405.8
        # Original Education: 22.68

    def insample_ml(self, model_name, train_data, feature_type, features):
        """
        Use trained ML model to make decisions via optimization
        """
        # Get the trained model

        # 准备训练数据
        response = train_data['EV Original Registrations']
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

        random.seed(15)
        r = self.revenue
        f = self.fixedCost
        c = self.transportationCost
        chargerCost = self.chargerCost
        C = self.capacity
        p = self.penaltyCost
        num_potential = self.numFacilities
        num_zip = self.numCustomers
        # Select and fit the model
        residuals = list(residuals)
        sce = len(residuals)
        prob = [1 / sce] * sce

        # Model
        m = gp.Model('nonparametric_dep')
        m.setParam('MIPGap', 0.01)

        # Variables
        charger = m.addVars(num_potential, vtype=GRB.INTEGER, ub=10, name='Charger')
        location = m.addVars(num_potential, vtype=GRB.BINARY, name='Location')
        assign = m.addVars(num_potential, num_zip, sce, lb=0.0, vtype=GRB.CONTINUOUS, name='Assign')
        projected_demand = m.addVars(num_zip, sce, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='demand')
        estimated_demand = m.addVars(num_zip, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS,
                                     name='increase_demand')
        unmetdemand = m.addVars(num_zip, sce, lb=0.0, vtype=GRB.CONTINUOUS, name='Unmet')
        demand = m.addVars(num_zip, sce, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='demand')
        h_term = m.addVars(num_zip, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='h_term')

        # Constraints
        m.addConstrs((charger[i] <= 10 * location[i] for i in range(num_potential)), name='Location')
        m.addConstrs(
            (gp.quicksum(assign[(i, j, k)] for i in range(num_potential)) + unmetdemand[(j, k)] == demand[(j, k)]
             for j in range(num_zip) for k in range(sce)), name='Demand')
        m.addConstrs(
            (gp.quicksum(assign[(i, j, k)] for j in range(num_zip)) <= C[i] * charger[i] for i in
             range(num_potential)
             for k in range(sce)), name='Capacity')

        if feature_type == 'H1':
            # H1: sum_capacity
            m.addConstrs(
                (h_term[j] == gp.quicksum(charger[i] for i in range(num_potential)) for j in range(num_zip)),
                name='H1_Term'
            )
            for j in range(num_zip):
                # ✅ 正确的 DataFrame 创建方式
                feats = pd.DataFrame({
                    'normalized_education_level': [self.education],
                    'normalized_gdp': [self.gdp],
                    'H1': [h_term[j]]
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
                    'normalized_education_level': [self.education],
                    'normalized_gdp': [self.gdp],
                    'H2': [h_term[j]]
                })

                # 添加预测约束
                pred_constr = add_predictor_constr(
                    gp_model=m,
                    predictor=trained_model,
                    input_vars=feats,  # DataFrame 格式
                    output_vars=estimated_demand[j],
                    epsilon=1e-5
                )
        elif feature_type == 'H3':
            # H3: step1_exp_distance_capacity
            m.addConstrs(
                (h_term[j] == gp.quicksum(
                    np.exp(self.experiments_distance.iloc[i, j] * self.step1_coef[0] + self.step1_intercept) * charger[i]
                    for i in range(num_potential)) for j in range(num_zip)
                 ),
                name='H3_Term'
            )
            for j in range(num_zip):
                # ✅ 正确的 DataFrame 创建方式
                feats = pd.DataFrame({
                    'normalized_education_level': [self.education],
                    'normalized_gdp': [self.gdp],
                    'time_exp': [h_term[j]]
                })

                # 添加预测约束
                pred_constr = add_predictor_constr(
                    gp_model=m,
                    predictor=trained_model,
                    input_vars=feats,  # DataFrame 格式
                    output_vars=estimated_demand[j],
                    epsilon=1e-5
                )

        m.addConstrs(
            (demand[(j, k)] == 15000 +
             estimated_demand[j] +
             residuals[k]
             for j in range(num_zip) for k in range(sce)),
            name='Real_Demand')

        # max(real_demand, 0)
        m.addConstrs(
            (projected_demand[(j, k)] == gp.max_([demand[(j, k)], 0.0]) for j in range(num_zip) for k in
             range(sce)),
            name='Projected_Demand_Positive'
        )

        # Objective function components
        fixed_cost = location.prod(f)
        charger_cost = charger.prod(chargerCost)
        transportation_cost = gp.quicksum(assign[(i, j, k)] * c.iloc[i, j] * prob[k] for i in range(num_potential)
                                          for j in range(num_zip) for k in range(sce))
        penalty_cost = gp.quicksum(unmetdemand[j, k] * p[j] * prob[k] for j in range(num_zip) for k in range(sce))
        revenue = gp.quicksum(r[j] * projected_demand[(j, k)] * prob[k] for j in range(num_zip) for k in range(sce))

        # Set the objective function
        m.setObjective(fixed_cost + charger_cost + transportation_cost + penalty_cost - revenue, GRB.MINIMIZE)
        m.optimize()


        if m.status == GRB.OPTIMAL:
            # Extract decision variable values
            location_values = [charger[i].X for i in range(num_potential)]
            optimal_value = m.objVal
            revenue_value = sum(r[j] * demand[(j, k)].X * prob[k] for j in range(num_zip) for k in range(sce))
            real_demand_values = [projected_demand[(j, k)].X for j in range(num_zip) for k in range(sce)]
            print('Objective Breakdown:')
            print(f'Dependent Model: {model_name}')
            print(f'Fixed Cost: {fixed_cost.getValue()}')
            print(f'Charger Cost: {charger_cost.getValue()}')
            print(f'Transportation Cost: {transportation_cost.getValue()}')
            print(f'Penalty Cost: {penalty_cost.getValue()}')
            print(f'Revenue: {revenue.getValue()}')
            print(f'Total Objective Value: {optimal_value}')
            fixed_cost_value = fixed_cost.getValue()
            charger_cost_value = charger_cost.getValue()
            transportation_cost_value = transportation_cost.getValue()
            penalty_cost_value = penalty_cost.getValue()
            revenue_value = revenue.getValue()
            cost_breakdown = {
                'fixed_cost': fixed_cost_value,
                'charger_cost': charger_cost_value,
                'transportation_cost': transportation_cost_value,
                'penalty_cost': penalty_cost_value,
                'revenue': revenue_value
            }
            return location_values, optimal_value, revenue_value, real_demand_values, cost_breakdown

    def nn_experiments(self):
        self.input_data()  # Input data
        self.generateSyntheticData(seed=425)  # Generate synthetic data
        # regression_models = ['Neural Network', 'Random Forest', 'Gradient Boosting', 'Decision Tree', 'kNN']
        regression_models = ['Neural Network']
        # train_ml_data = self.train_ml
        # test_ml_data = self.test_ml

        train_ml_data = self.step2_data[self.step2_data['Year'] <= 2020]
        test_ml_data = self.step2_data[self.step2_data['Year'] >= 2021]

        # ✅ 初始化空的 DataFrame
        ml_df = pd.DataFrame()

        feature_list = [
            ['normalized_education_level', 'normalized_gdp', 'H1'],
            ['normalized_education_level', 'normalized_gdp', 'H2'],
            ['normalized_education_level', 'normalized_gdp', 'time_exp']
        ]

        # Map feature_list to 4 feature type H1, H2, H3, H4
        feature_types = ['H1', 'H2', 'H3']

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
                ml_location_values, ml_optimal_value, ml_revenue_value, ml_real_demand_values, ml_breakdown = self.insample_ml(
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
                    'runtime': model_runtime,
                    "fixed_cost": ml_breakdown['fixed_cost'],
                    "charger_cost": ml_breakdown['charger_cost'],
                    "transportation_cost": ml_breakdown['transportation_cost'],
                    "penalty_cost": ml_breakdown['penalty_cost'],
                    "revenue": ml_breakdown['revenue']
                }])

                # ✅ 直接 concat 到 ml_df
                ml_df = pd.concat([ml_df, current_result], ignore_index=True)

                print(f"  Runtime: {model_runtime:.2f}s")
                print(f"  In-sample Optimal Value: {ml_optimal_value:.2f}")

        print(f"\n{'=' * 80}")
        print("INSAMPLE EXPERIMENTS SUMMARY")
        print(f"{'=' * 80}")
        print(f"Two-Step Model:")
        print(f"\nML Models:")
        print(f"  Total model-feature combinations: {len(ml_df)}")
        print(f"  In-sample results shape: {ml_df.shape}")

        return ml_df

    def dep_experiments(self, model, residuals_dep, experiments_type, decision):
        # Generate random values
        random.seed(15)
        r = self.revenue
        f = self.fixedCost
        c = self.transportationCost
        chargerCost = self.chargerCost
        C = self.capacity
        p = self.penaltyCost
        num_potential = self.numFacilities
        num_zip = self.numCustomers
        # Select and fit the model
        residuals = list(residuals_dep)
        predicted_demand = self.best_models_dep[model].predict(np.array([[self.education, self.gdp]]))[0]
        print(predicted_demand)
        initial_demand = [15000 + predicted_demand] * self.numCustomers
        orig_demand = pd.DataFrame(
            np.exp(self.step1_coef[0] * self.experiments_distance + self.step1_intercept) * self.t_threshold)
        self.orig_demand = orig_demand
        sce = len(residuals)
        exp_demand = {(i, j, k): orig_demand.iloc[i, j] for i in range(num_potential) for j in range(num_zip) for k in
                      range(sce)}
        prob = np.array([1.0 / sce] * sce)

        if experiments_type == 'insample':
            # Model
            m = gp.Model('nonparametric_dep')
            m.setParam('MIPGap', 0.01)

            # Variables
            charger = m.addVars(num_potential, vtype=GRB.INTEGER, ub=10, name='Charger')
            location = m.addVars(num_potential, vtype=GRB.BINARY, name='Location')
            assign = m.addVars(num_potential, num_zip, sce, lb=0.0, vtype=GRB.CONTINUOUS, name='Assign')
            unmetdemand = m.addVars(num_zip, sce, lb=0.0, vtype=GRB.CONTINUOUS, name='Unmet')
            demand = m.addVars(num_zip, sce, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='demand')

            # Constraints
            m.addConstrs((charger[i] <= 10 * location[i] for i in range(num_potential)), name='Location')
            m.addConstrs(
                (gp.quicksum(assign[(i, j, k)] for i in range(num_potential)) + unmetdemand[(j, k)] == demand[(j, k)]
                 for j in range(num_zip) for k in range(sce)), name='Demand')
            m.addConstrs(
                (gp.quicksum(assign[(i, j, k)] for j in range(num_zip)) <= C[i] * charger[i] for i in
                 range(num_potential)
                 for k in range(sce)), name='Capacity')
            m.addConstrs(
                (demand[j, k] == initial_demand[j] + gp.quicksum(
                    exp_demand[(i, j, k)] * charger[i] for i in range(num_potential))
                 + residuals[k] for j in range(num_zip) for k in range(sce)), name='Real_Demand')

            # Objective function components
            fixed_cost = location.prod(f)
            charger_cost = charger.prod(chargerCost)
            transportation_cost = gp.quicksum(assign[(i, j, k)] * c.iloc[i, j] * prob[k] for i in range(num_potential)
                                              for j in range(num_zip) for k in range(sce))
            penalty_cost = gp.quicksum(unmetdemand[j, k] * p[j] * prob[k] for j in range(num_zip) for k in range(sce))
            revenue = gp.quicksum(r[j] * demand[(j, k)] * prob[k] for j in range(num_zip) for k in range(sce))

            # Set the objective function
            m.setObjective(fixed_cost + charger_cost + transportation_cost + penalty_cost - revenue, GRB.MINIMIZE)
            m.optimize()

            # Write the LP file
            lp_filename = f"nonparametric_dep_{model}.lp"
            m.write(lp_filename)

            if m.status == GRB.OPTIMAL:
                # Extract decision variable values
                location_values = [charger[i].X for i in range(num_potential)]
                optimal_value = m.objVal
                revenue_value = sum(r[j] * demand[(j, k)].X * prob[k] for j in range(num_zip) for k in range(sce))
                real_demand_values = []
                for j in range(self.numCustomers):
                    real_demand = initial_demand + sum(
                        orig_demand.iloc[i, j] * location_values[i] for i in range(self.numFacilities))
                    real_demand_values.append(real_demand)
                print('Objective Breakdown:')
                print(f'Dependent Model: {model}')
                print(f'Fixed Cost: {fixed_cost.getValue()}')
                print(f'Charger Cost: {charger_cost.getValue()}')
                print(f'Transportation Cost: {transportation_cost.getValue()}')
                print(f'Penalty Cost: {penalty_cost.getValue()}')
                print(f'Revenue: {revenue.getValue()}')
                print(f'Total Objective Value: {optimal_value}')
                fixed_cost_value = fixed_cost.getValue()
                charger_cost_value = charger_cost.getValue()
                transportation_cost_value = transportation_cost.getValue()
                penalty_cost_value = penalty_cost.getValue()
                revenue_value = revenue.getValue()
                cost_breakdown = {
                    'fixed_cost': fixed_cost_value,
                    'charger_cost': charger_cost_value,
                    'transportation_cost': transportation_cost_value,
                    'penalty_cost': penalty_cost_value,
                    'revenue': revenue_value
                }
                return location_values, optimal_value, revenue_value, real_demand_values, cost_breakdown

        if experiments_type == 'oos':
            # Model
            m = gp.Model('nonparametric_dep_oos')
            m.setParam('MIPGap', 0.01)

            # Variables
            charger = m.addVars(num_potential, vtype=GRB.INTEGER, ub=10, name='Charger')
            location = m.addVars(num_potential, vtype=GRB.BINARY, name='Location')
            assign = m.addVars(num_potential, num_zip, sce, lb=0.0, vtype=GRB.CONTINUOUS, name='Assign')
            unmetdemand = m.addVars(num_zip, sce, lb=0.0, vtype=GRB.CONTINUOUS, name='Unmet')
            demand = m.addVars(num_zip, sce, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='demand')

            # Constraints
            m.addConstrs((10 * location[i] >= charger[i] for i in range(num_potential)), name='Decision')
            m.addConstrs((charger[i] == decision[i] for i in range(num_potential)), name='Location')
            m.addConstrs(
                (gp.quicksum(assign[(i, j, k)] for i in range(num_potential)) + unmetdemand[(j, k)] == demand[(j, k)]
                 for j in range(num_zip) for k in range(sce)), name='Demand')
            m.addConstrs(
                (gp.quicksum(assign[(i, j, k)] for j in range(num_zip)) <= C[i] * charger[i] for i in
                 range(num_potential)
                 for k in range(sce)), name='Capacity')
            m.addConstrs((demand[j, k] == initial_demand[j] + gp.quicksum(
                exp_demand[(i, j, k)] * charger[i] for i in range(num_potential))
                          + residuals[k] for j in range(num_zip) for k in range(sce)),
                         name='demand_projection')

            # Objective function components
            fixed_cost = location.prod(f)
            charger_cost = charger.prod(chargerCost)
            transportation_cost = gp.quicksum(assign[(i, j, k)] * c.iloc[i, j] * prob[k] for i in range(num_potential)
                                              for j in range(num_zip) for k in range(sce))
            penalty_cost = gp.quicksum(unmetdemand[j, k] * p[j] * prob[k] for j in range(num_zip) for k in range(sce))
            revenue = gp.quicksum(r[j] * demand[(j, k)] * prob[k] for j in range(num_zip) for k in range(sce))

            # Set the objective function
            m.setObjective(fixed_cost + charger_cost + transportation_cost + penalty_cost - revenue, GRB.MINIMIZE)
            m.optimize()

            # Write the LP file
            lp_filename = f"nonparametric_dep_{model}.lp"
            m.write(lp_filename)

            if m.status == GRB.OPTIMAL:
                # Extract decision variable values
                location_values = [location[i].X for i in range(num_potential)]
                optimal_value = m.objVal
                revenue_value = sum(r[j] * demand[(j, k)].X * prob[k] for j in range(num_zip) for k in range(sce))
                increase_demand_values = []
                for j in range(self.numCustomers):
                    real_demand = initial_demand[j] + sum(
                        orig_demand.iloc[i, j] * location_values[i] for i in range(self.numFacilities))
                    increase_demand_values.append(real_demand)
                return location_values, optimal_value, revenue_value, increase_demand_values

    def ind_experiments(self, model, residuals_ind, experiments_type, decision):
        # Generate random values
        random.seed(15)
        r = self.revenue
        f = self.fixedCost
        c = self.transportationCost
        chargerCost = self.chargerCost
        C = self.capacity
        p = self.penaltyCost
        num_potential = self.numFacilities
        num_zip = self.numCustomers
        # Select and fit the model
        regression_model = model
        residuals_ind = list(residuals_ind)
        predicted_demand_ind = self.best_models_ind[model].predict(np.array([[self.education, self.gdp]]))[0]
        print(predicted_demand_ind)
        initial_demand = [15000 + predicted_demand_ind] * self.numCustomers
        sce = len(residuals_ind)
        prob = np.array([1.0 / sce] * sce)

        if experiments_type == 'insample':
            # Model
            m = gp.Model('nonparametric_ind')
            m.setParam('MIPGap', 0.01)

            # Variables
            charger = m.addVars(num_potential, vtype=GRB.INTEGER, ub=10, name='Charger')
            location = m.addVars(num_potential, vtype=GRB.BINARY, name='Location')
            assign = m.addVars(num_potential, num_zip, sce, lb=0.0, vtype=GRB.CONTINUOUS, name='Assign')
            unmetdemand = m.addVars(num_zip, sce, vtype=GRB.CONTINUOUS, name='unsatisfied_demand')
            demand = m.addVars(num_zip, sce, vtype=GRB.CONTINUOUS, name='demand')

            # Constraints
            m.addConstrs((charger[i] <= 10 * location[i] for i in range(num_potential)), name='Decision')
            m.addConstrs((gp.quicksum(assign[i, j, k] for i in range(num_potential)) + unmetdemand[j, k] == demand[j, k]
                          for j in range(num_zip) for k in range(sce)), name='Demand')
            m.addConstrs(
                (gp.quicksum(assign[i, j, k] for j in range(num_zip)) <= C[i] * charger[i] for i in
                 range(num_potential)
                 for k in range(sce)), name='Capacity')
            m.addConstrs((demand[j, k] == gp.max_([initial_demand[j] + residuals_ind[k], 0.0]) for j in range(num_zip)
                          for k in range(sce)), name='Real_demand')

            # Objective function components
            fixed_cost = location.prod(f)
            charger_cost = charger.prod(chargerCost)
            transportation_cost = gp.quicksum(assign[(i, j, k)] * c.iloc[i, j] * prob[k] for i in range(num_potential)
                                              for j in range(num_zip) for k in range(sce))
            penalty_cost = gp.quicksum(unmetdemand[j, k] * p[j] * prob[k] for j in range(num_zip) for k in range(sce))
            revenue = gp.quicksum(r[j] * demand[j, k] * prob[k] for j in range(num_zip) for k in range(sce))

            # Set the objective function
            m.setObjective(fixed_cost + charger_cost + transportation_cost + penalty_cost - revenue, GRB.MINIMIZE)
            m.optimize()

            # Write the LP file
            lp_filename = f"nonparametric_ind_{model}.lp"
            m.write(lp_filename)

            if m.status == GRB.OPTIMAL:
                decision_values = [location[i].X for i in range(num_potential)]
                print('ind_location', decision_values)
                # Extract decision variable values
                location_values = [charger[i].X for i in range(num_potential)]
                print('ind_charger', location_values)
                optimal_value = m.objVal
                revenue_value = revenue.getValue()
                real_demand_values = initial_demand
                print('Objective Breakdown:')
                print(f'Independent Model: {model}')
                print(f'Fixed Cost: {fixed_cost.getValue()}')
                print(f'Charger Cost: {charger_cost.getValue()}')
                print(f'Transportation Cost: {transportation_cost.getValue()}')
                print(f'Penalty Cost: {penalty_cost.getValue()}')
                print(f'Revenue: {revenue.getValue()}')
                print(f'Total Objective Value: {optimal_value}')
                fixed_cost_value = fixed_cost.getValue()
                charger_cost_value = charger_cost.getValue()
                transportation_cost_value = transportation_cost.getValue()
                penalty_cost_value = penalty_cost.getValue()
                revenue_value = revenue.getValue()
                cost_breakdown = {
                    'fixed_cost': fixed_cost_value,
                    'charger_cost': charger_cost_value,
                    'transportation_cost': transportation_cost_value,
                    'penalty_cost': penalty_cost_value,
                    'revenue': revenue_value
                }
                return location_values, optimal_value, revenue_value, real_demand_values, cost_breakdown

        if experiments_type == 'oos':
            # Model
            m = gp.Model('nonparametric_ind_oos')
            m.setParam('MIPGap', 0.01)

            # Variables
            charger = m.addVars(num_potential, vtype=GRB.INTEGER, ub=10, name='Charger')
            location = m.addVars(num_potential, vtype=GRB.BINARY, name='Location')
            assign = m.addVars(num_potential, num_zip, sce, lb=0.0, vtype=GRB.CONTINUOUS, name='Assign')
            unmetdemand = m.addVars(num_zip, sce, vtype=GRB.CONTINUOUS, name='unsatisfied_demand')
            demand = m.addVars(num_zip, sce, vtype=GRB.CONTINUOUS, name='demand')

            # Constraints
            m.addConstrs((10 * location[i] >= charger[i] for i in range(num_potential)), name='Decision')
            m.addConstrs((charger[i] == decision[i] for i in range(self.numFacilities)), name='Location')
            m.addConstrs((gp.quicksum(assign[i, j, k] for i in range(num_potential)) + unmetdemand[j, k] == demand[j, k]
                          for j in range(num_zip) for k in range(sce)), name='Demand')
            m.addConstrs(
                (gp.quicksum(assign[i, j, k] for j in range(num_zip)) <= C[i] * charger[i] for i in
                 range(num_potential)
                 for k in range(sce)), name='Capacity')
            m.addConstrs((demand[j, k] == gp.max_([initial_demand[j] + residuals_ind[k], 0.0]) for j in range(num_zip)
                          for k in range(sce)), name='Real_demand')

            # Objective function components
            fixed_cost = location.prod(f)
            transportation_cost = gp.quicksum(assign[(i, j, k)] * c.iloc[i, j] * prob[k] for i in range(num_potential)
                                              for j in range(num_zip) for k in range(sce))
            penalty_cost = gp.quicksum(unmetdemand[j, k] * p[j] * prob[k] for j in range(num_zip) for k in range(sce))
            revenue = gp.quicksum(r[j] * demand[j, k] * prob[k] for j in range(num_zip) for k in range(sce))

            # Set the objective function
            m.setObjective(fixed_cost + transportation_cost + penalty_cost - revenue, GRB.MINIMIZE)
            m.optimize()

            # Write the LP file
            lp_filename = f"nonparametric_ind_{model}.lp"
            m.write(lp_filename)

            if m.status == GRB.OPTIMAL:
                # Extract decision variable values
                location_values = [charger[i].X for i in range(num_potential)]
                optimal_value = m.objVal
                revenue_value = revenue.getValue()
                real_demand_values = initial_demand
                return location_values, optimal_value, revenue_value, real_demand_values

    def linear_dep_experiments(self, residuals_dep, experiments_type, decision):
        # Generate random values
        random.seed(15)
        r = self.revenue
        f = self.fixedCost
        c = self.transportationCost
        chargerCost = self.chargerCost
        C = self.capacity
        p = self.penaltyCost
        num_potential = self.numFacilities
        num_zip = self.numCustomers
        # Select and fit the model
        residuals = list(residuals_dep)
        predicted_demand = np.dot(self.step2_dep_coef, np.array([self.education, self.gdp])) + self.step2_dep_intercept
        print(predicted_demand)
        initial_demand = [15000 + predicted_demand] * self.numCustomers
        orig_demand = pd.DataFrame(
            np.exp(self.step1_coef[0] * self.experiments_distance + self.step1_intercept) * self.t_threshold)
        self.orig_demand = orig_demand
        sce = len(residuals)
        exp_demand = {(i, j, k): orig_demand.iloc[i, j] for i in range(num_potential) for j in range(num_zip) for k in
                      range(sce)}
        prob = np.array([1.0 / sce] * sce)

        if experiments_type == 'insample':
            # Model
            m = gp.Model('nonparametric_dep')
            m.setParam('MIPGap', 0.01)

            # Variables
            charger = m.addVars(num_potential, vtype=GRB.INTEGER, ub=10, name='Charger')
            location = m.addVars(num_potential, vtype=GRB.BINARY, name='Location')
            assign = m.addVars(num_potential, num_zip, sce, lb=0.0, vtype=GRB.CONTINUOUS, name='Assign')
            unmetdemand = m.addVars(num_zip, sce, lb=0.0, vtype=GRB.CONTINUOUS, name='Unmet')
            demand = m.addVars(num_zip, sce, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='demand')

            # Constraints
            m.addConstrs((charger[i] <= 10 * location[i] for i in range(num_potential)), name='Location')
            m.addConstrs(
                (gp.quicksum(assign[(i, j, k)] for i in range(num_potential)) + unmetdemand[(j, k)] == demand[(j, k)]
                 for j in range(num_zip) for k in range(sce)), name='Demand')
            m.addConstrs(
                (gp.quicksum(assign[(i, j, k)] for j in range(num_zip)) <= C[i] * charger[i] for i in
                 range(num_potential)
                 for k in range(sce)), name='Capacity')
            m.addConstrs((demand[j, k] == initial_demand[j] + gp.quicksum(
                exp_demand[(i, j, k)] * charger[i] for i in range(num_potential))
                          + residuals[k] for j in range(num_zip) for k in range(sce)),
                         name='demand_projection')

            # Objective function components
            fixed_cost = location.prod(f)
            charger_cost = charger.prod(chargerCost)
            transportation_cost = gp.quicksum(assign[(i, j, k)] * c.iloc[i, j] * prob[k] for i in range(num_potential)
                                              for j in range(num_zip) for k in range(sce))
            penalty_cost = gp.quicksum(unmetdemand[j, k] * p[j] * prob[k] for j in range(num_zip) for k in range(sce))
            revenue = gp.quicksum(r[j] * demand[(j, k)] * prob[k] for j in range(num_zip) for k in range(sce))

            # Set the objective function
            m.setObjective(fixed_cost + charger_cost + transportation_cost + penalty_cost - revenue, GRB.MINIMIZE)
            m.optimize()

            if m.status == GRB.OPTIMAL:
                # Extract decision variable values
                location_values = [charger[i].X for i in range(num_potential)]
                optimal_value = m.objVal
                revenue_value = revenue.getValue()

                # Check if any demand equals zero
                real_demand_values = []
                for j in range(self.numCustomers):
                    real_demand = initial_demand[j] + sum(
                        orig_demand.iloc[i, j] * location_values[i] for i in range(self.numFacilities))
                    real_demand_values.append(real_demand)

                increase_demand_values = []
                for j in range(self.numCustomers):
                    real_demand = initial_demand + sum(
                        orig_demand.iloc[i, j] * location_values[i] for i in range(self.numFacilities))
                    increase_demand_values.append(real_demand)
                fixed_cost_value = fixed_cost.getValue()
                charger_cost_value = charger_cost.getValue()
                transportation_cost_value = transportation_cost.getValue()
                penalty_cost_value = penalty_cost.getValue()
                revenue_value = revenue.getValue()
                cost_breakdown = {
                    'fixed_cost': fixed_cost_value,
                    'charger_cost': charger_cost_value,
                    'transportation_cost': transportation_cost_value,
                    'penalty_cost': penalty_cost_value,
                    'revenue': revenue_value
                }
                return location_values, optimal_value, revenue_value, increase_demand_values, cost_breakdown

        if experiments_type == 'oos':
            # Model
            m = gp.Model('nonparametric_dep_oos')
            m.setParam('MIPGap', 0.01)

            # Variables
            charger = m.addVars(num_potential, vtype=GRB.INTEGER, ub=10, name='Charger')
            location = m.addVars(num_potential, vtype=GRB.BINARY, name='Location')
            assign = m.addVars(num_potential, num_zip, sce, lb=0.0, vtype=GRB.CONTINUOUS, name='Assign')
            unmetdemand = m.addVars(num_zip, sce, lb=0.0, vtype=GRB.CONTINUOUS, name='Unmet')
            demand = m.addVars(num_zip, sce, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='demand')
            real_demand = m.addVars(num_zip, sce, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='real_demand')

            # Constraints
            m.addConstrs((10 * location[i] >= charger[i] for i in range(num_potential)), name='Decision')
            m.addConstrs((charger[i] == decision[i] for i in range(self.numFacilities)), name='Location')
            m.addConstrs(
                (gp.quicksum(assign[(i, j, k)] for i in range(num_potential)) + unmetdemand[(j, k)] == demand[(j, k)]
                 for j in range(num_zip) for k in range(sce)), name='Demand')
            m.addConstrs(
                (gp.quicksum(assign[(i, j, k)] for j in range(num_zip)) <= C[i] * location[i] for i in
                 range(num_potential)
                 for k in range(sce)), name='Capacity')
            m.addConstrs((demand[j, k] == initial_demand[j] + gp.quicksum(
                exp_demand[(i, j, k)] * location[i] for i in range(num_potential))
                          + residuals[k] for j in range(num_zip) for k in range(sce)), name='demand_projection')

            # Objective function components
            fixed_cost = location.prod(f)
            charger_cost = charger.prod(chargerCost)
            transportation_cost = gp.quicksum(assign[(i, j, k)] * c.iloc[i, j] * prob[k] for i in range(num_potential)
                                              for j in range(num_zip) for k in range(sce))
            penalty_cost = gp.quicksum(unmetdemand[j, k] * p[j] * prob[k] for j in range(num_zip) for k in range(sce))
            revenue = gp.quicksum(r[j] * demand[(j, k)] * prob[k] for j in range(num_zip) for k in range(sce))

            # Set the objective function
            m.setObjective(fixed_cost + charger_cost + transportation_cost + penalty_cost - revenue, GRB.MINIMIZE)
            m.optimize()

            if m.status == GRB.OPTIMAL:
                # Extract decision variable values
                location_values = [charger[i].X for i in range(num_potential)]
                optimal_value = m.objVal
                revenue_value = sum(r[j] * demand[(j, k)].X * prob[k] for j in range(num_zip) for k in range(sce))

                # Check if any demand equals zero
                real_demand_values = []
                for j in range(self.numCustomers):
                    real_demand = initial_demand[j] + sum(
                        orig_demand.iloc[i, j] * location_values[i] for i in range(self.numFacilities))
                    real_demand_values.append(real_demand)

                print('Objective Breakdown:')
                print('Dependent Model: Linear Regression')
                print(f'Fixed Cost: {fixed_cost.getValue()}')
                print(f'Charger Cost: {charger_cost.getValue()}')
                print(f'Transportation Cost: {transportation_cost.getValue()}')
                print(f'Penalty Cost: {penalty_cost.getValue()}')
                print(f'Revenue: {revenue.getValue()}')
                print(f'Total Objective Value: {optimal_value}')
                return location_values, optimal_value, revenue_value, real_demand_values

    def linear_ind_experiments(self, residuals_ind, experiments_type, decision):
        # Generate random values
        random.seed(15)
        r = self.revenue
        f = self.fixedCost
        c = self.transportationCost
        chargerCost = self.chargerCost
        C = self.capacity
        p = self.penaltyCost
        num_potential = self.numFacilities
        num_zip = self.numCustomers
        # Select and fit the model
        residuals_ind = list(residuals_ind)
        predicted_demand_ind = np.dot(self.step2_ind_coef,
                                      np.array([self.education, self.gdp])) + self.step2_ind_intercept
        initial_demand = [15000 + predicted_demand_ind] * self.numCustomers
        sce = len(residuals_ind)
        prob = np.array([1.0 / sce] * sce)

        if experiments_type == 'insample':
            # Model
            m = gp.Model('linear_ind')
            m.setParam('MIPGap', 0.01)

            # Variables
            charger = m.addVars(num_potential, vtype=GRB.INTEGER, ub=10, name='Charger')
            location = m.addVars(num_potential, vtype=GRB.BINARY, name='Location')
            assign = m.addVars(num_potential, num_zip, sce, lb=0.0, vtype=GRB.CONTINUOUS, name='Assign')
            unmetdemand = m.addVars(num_zip, sce, vtype=GRB.CONTINUOUS, name='unsatisfied_demand')
            demand = m.addVars(num_zip, sce, vtype=GRB.CONTINUOUS, name='demand')

            # Constraints
            m.addConstrs((10 * location[i] >= charger[i] for i in range(num_potential)), name='Decision')
            m.addConstrs((gp.quicksum(assign[i, j, k] for i in range(num_potential)) + unmetdemand[j, k] == demand[j, k]
                          for j in range(num_zip) for k in range(sce)), name='Demand')
            m.addConstrs(
                (gp.quicksum(assign[i, j, k] for j in range(num_zip)) <= C[i] * charger[i] for i in
                 range(num_potential)
                 for k in range(sce)), name='Capacity')
            m.addConstrs((demand[j, k] == gp.max_([initial_demand[j] + residuals_ind[k], 0.0]) for j in range(num_zip)
                          for k in range(sce)), name='Real_demand')

            # Objective function components
            fixed_cost = location.prod(f)
            charger_cost = charger.prod(chargerCost)
            transportation_cost = gp.quicksum(assign[(i, j, k)] * c.iloc[i, j] * prob[k] for i in range(num_potential)
                                              for j in range(num_zip) for k in range(sce))
            penalty_cost = gp.quicksum(unmetdemand[j, k] * p[j] * prob[k] for j in range(num_zip) for k in range(sce))
            revenue = gp.quicksum(r[j] * demand[j, k] * prob[k] for j in range(num_zip) for k in range(sce))

            # Set the objective function
            m.setObjective(fixed_cost + charger_cost + transportation_cost + penalty_cost - revenue, GRB.MINIMIZE)
            m.optimize()

            if m.status == GRB.OPTIMAL:
                # Extract decision variable values
                location_values = [charger[i].X for i in range(num_potential)]
                optimal_value = m.objVal
                revenue_value = revenue.getValue()
                real_demand_values = initial_demand
                print('Objective Breakdown:')
                print('Independent Model: Linear Regression')
                print(f'Fixed Cost: {fixed_cost.getValue()}')
                print(f'Charger Cost: {charger_cost.getValue()}')
                print(f'Transportation Cost: {transportation_cost.getValue()}')
                print(f'Penalty Cost: {penalty_cost.getValue()}')
                print(f'Revenue: {revenue.getValue()}')
                print(f'Total Objective Value: {optimal_value}')
                fixed_cost_value = fixed_cost.getValue()
                charger_cost_value = charger_cost.getValue()
                transportation_cost_value = transportation_cost.getValue()
                penalty_cost_value = penalty_cost.getValue()
                revenue_value = revenue.getValue()
                cost_breakdown = {
                    'fixed_cost': fixed_cost_value,
                    'charger_cost': charger_cost_value,
                    'transportation_cost': transportation_cost_value,
                    'penalty_cost': penalty_cost_value,
                    'revenue': revenue_value
                }
                return location_values, optimal_value, revenue_value, real_demand_values, cost_breakdown

        if experiments_type == 'oos':
            # Model
            m = gp.Model('linear_ind_oos')
            m.setParam('MIPGap', 0.01)

            # Variables
            charger = m.addVars(num_potential, vtype=GRB.INTEGER, ub=10, name='Charger')
            location = m.addVars(num_potential, vtype=GRB.BINARY, name='Location')
            assign = m.addVars(num_potential, num_zip, sce, lb=0.0, vtype=GRB.CONTINUOUS, name='Assign')
            unmetdemand = m.addVars(num_zip, sce, vtype=GRB.CONTINUOUS, name='unsatisfied_demand')
            demand = m.addVars(num_zip, sce, vtype=GRB.CONTINUOUS, name='demand')

            # Constraints
            m.addConstrs((10 * location[i] >= charger[i] for i in range(num_potential)), name='Decision')
            m.addConstrs((charger[i] == decision[i] for i in range(self.numFacilities)), name='Location')
            m.addConstrs((gp.quicksum(assign[i, j, k] for i in range(num_potential)) + unmetdemand[j, k] == demand[j, k]
                          for j in range(num_zip) for k in range(sce)), name='Demand')
            m.addConstrs(
                (gp.quicksum(assign[i, j, k] for j in range(num_zip)) <= C[i] * location[i] for i in
                 range(num_potential)
                 for k in range(sce)), name='Capacity')
            m.addConstrs((demand[j, k] == gp.max_([initial_demand[j] + residuals_ind[k], 0.0]) for j in range(num_zip)
                          for k in range(sce)), name='Real_demand')

            # Objective function components
            fixed_cost = location.prod(f)
            charger_cost = charger.prod(chargerCost)
            transportation_cost = gp.quicksum(assign[(i, j, k)] * c.iloc[i, j] * prob[k] for i in range(num_potential)
                                              for j in range(num_zip) for k in range(sce))
            penalty_cost = gp.quicksum(unmetdemand[j, k] * p[j] * prob[k] for j in range(num_zip) for k in range(sce))
            revenue = gp.quicksum(r[j] * demand[j, k] * prob[k] for j in range(num_zip) for k in range(sce))

            # Set the objective function
            m.setObjective(fixed_cost + charger_cost + transportation_cost + penalty_cost - revenue, GRB.MINIMIZE)
            m.optimize()

            if m.status == GRB.OPTIMAL:
                # Extract decision variable values
                location_values = [charger[i].X for i in range(num_potential)]
                optimal_value = m.objVal
                revenue_value = revenue.getValue()
                real_demand_values = initial_demand
                return location_values, optimal_value, revenue_value, real_demand_values

    def linear_dep_experiments_deterministic(self, residuals_dep, experiments_type, decision):
        # Generate random values
        random.seed(15)
        r = self.revenue
        f = self.fixedCost
        c = self.transportationCost
        chargerCost = self.chargerCost
        C = self.capacity
        p = self.penaltyCost
        num_potential = self.numFacilities
        num_zip = self.numCustomers
        # Select and fit the model
        residuals = list(residuals_dep)
        predicted_demand = np.dot(self.step2_dep_coef, np.array([self.education, self.gdp]))
        print(predicted_demand)
        initial_demand = [15000 + predicted_demand] * self.numCustomers
        orig_demand = pd.DataFrame(
            np.exp(self.step1_coef[0] * self.experiments_distance + self.step1_intercept) * self.t_threshold)
        self.orig_demand = orig_demand
        exp_demand = {(i, j): orig_demand.iloc[i, j] for i in range(num_potential) for j in range(num_zip)}

        if experiments_type == 'insample':
            # Model
            m = gp.Model('nonparametric_dep')
            m.setParam('MIPGap', 0.01)

            # Variables
            charger = m.addVars(num_potential, vtype=GRB.INTEGER, ub=10, name='Charger')
            location = m.addVars(num_potential, vtype=GRB.BINARY, name='Location')
            assign = m.addVars(num_potential, num_zip, lb=0.0, vtype=GRB.CONTINUOUS, name='Assign')
            unmetdemand = m.addVars(num_zip, lb=0.0, vtype=GRB.CONTINUOUS, name='Unmet')
            demand = m.addVars(num_zip, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='demand')

            # Constraints
            m.addConstrs((charger[i] <= 10 * location[i] for i in range(num_potential)), name='Location')
            m.addConstrs(
                (gp.quicksum(assign[(i, j)] for i in range(num_potential)) + unmetdemand[(j)] == demand[(j)]
                 for j in range(num_zip)), name='Demand')
            m.addConstrs(
                (gp.quicksum(assign[(i, j)] for j in range(num_zip)) <= C[i] * charger[i] for i in
                 range(num_potential)), name='Capacity')
            m.addConstrs((demand[(j)] == initial_demand[j] + gp.quicksum(
                exp_demand[(i, j)] * charger[i] for i in range(num_potential)) for j in range(num_zip)),
                         name='Real_Demand')

            # Objective function components
            fixed_cost = location.prod(f)
            charger_cost = charger.prod(chargerCost)
            transportation_cost = gp.quicksum(assign[(i, j)] * c.iloc[i, j] for i in range(num_potential)
                                              for j in range(num_zip))
            penalty_cost = gp.quicksum(unmetdemand[j] * p[j] for j in range(num_zip))
            revenue = gp.quicksum(r[j] * demand[(j)] for j in range(num_zip))

            # Set the objective function
            m.setObjective(fixed_cost + charger_cost + transportation_cost + penalty_cost - revenue, GRB.MINIMIZE)
            m.optimize()

            if m.status == GRB.OPTIMAL:
                # Extract decision variable values
                location_values = [charger[i].X for i in range(num_potential)]
                optimal_value = m.objVal
                revenue_value = sum(r[j] * demand[(j)].X for j in range(num_zip))
                real_demand_values = []
                for j in range(self.numCustomers):
                    real_demand = initial_demand[j] + sum(
                        orig_demand.iloc[i, j] * location_values[i] for i in range(num_potential))
                    real_demand_values.append(real_demand)
                self.dep_decision_deterministic = location_values
                return location_values, optimal_value, revenue_value, real_demand_values

    def insample_experiments(self):
        '''
        self.best_models_dep = joblib.load(f'capacity_best_models_dep_t{self.t_threshold}.joblib')

        # Restore residuals_dep
        with open(f'capacity_residuals_dep_t{self.t_threshold}.pkl', 'rb') as f:
            self.residuals_dep = pickle.load(f)

        self.best_models_ind = joblib.load(f'capacity_best_models_ind_t{self.t_threshold}.joblib')

        # Restore residuals_dep
        with open(f'capacity_residuals_ind_t{self.t_threshold}.pkl', 'rb') as f:
            self.residuals_ind = pickle.load(f)

            # Initialize insample_results with 'Decision-dependent' and 'Decision-independent' as dictionaries
            self.insample_results = {
                'Decision-dependent': {},
                'Decision-independent': {}
            }

            cost_breakdown_data = []

            # First loop for decision-dependent models
            for name, model in self.best_models_dep.items():
                residuals_dep = self.residuals_dep[name]
                dep_location, dep_obj, dep_revenue, real_demand, dep_costbreakdown = self.dep_experiments(name,
                                                                                                          residuals_dep,
                                                                                                          'insample',
                                                                                                          decision=None)

                print('residuals dep length', len(residuals_dep))
                # Add the result to 'Decision-dependent'
                self.insample_results['Decision-dependent'][name] = [dep_location, dep_obj, dep_revenue,
                                                                     real_demand]

                cost_breakdown_data.append({
                    'Model Name': name,
                    'Type': 'Decision-dependent',
                    'Fixed Cost': dep_costbreakdown['fixed_cost'],
                    'Charger Cost': dep_costbreakdown['charger_cost'],
                    'Transportation Cost': dep_costbreakdown['transportation_cost'],
                    'Penalty Cost': dep_costbreakdown['penalty_cost'],
                    'Revenue': dep_costbreakdown['revenue']
                })

            # Second loop for decision-independent models
            for name, model in self.best_models_ind.items():
                residuals_ind = self.residuals_ind[name]
                ind_location, ind_obj, ind_revenue, real_demand, ind_costbreakdown = self.ind_experiments(name,
                                                                                                          residuals_ind,
                                                                                                          'insample',
                                                                                                          decision=None)
                # Add the result to 'Decision-independent'
                self.insample_results['Decision-independent'][name] = [ind_location, ind_obj, ind_revenue,
                                                                       real_demand]
                cost_breakdown_data.append({
                    'Model Name': name,
                    'Type': 'Decision-independent',
                    'Fixed Cost': ind_costbreakdown['fixed_cost'],
                    'Charger Cost': ind_costbreakdown['charger_cost'],
                    'Transportation Cost': ind_costbreakdown['transportation_cost'],
                    'Penalty Cost': ind_costbreakdown['penalty_cost'],
                    'Revenue': ind_costbreakdown['revenue']
                })

            # Handle linear regression results for decision-dependent and decision-independent
            linear_residual_dep = self.step2_dep_residuals
            dep_location, dep_obj, dep_revenue, real_demand, dep_costbreakdown = self.linear_dep_experiments(
                linear_residual_dep, 'insample', decision=None)
            # Add the result to 'Decision-dependent'
            self.insample_results['Decision-dependent']['linear_regression'] = [dep_location, dep_obj, dep_revenue,
                                                                                real_demand]
            cost_breakdown_data.append({
                'Model Name': 'linear_regression',
                'Type': 'Decision-dependent',
                'Fixed Cost': dep_costbreakdown['fixed_cost'],
                'Charger Cost': dep_costbreakdown['charger_cost'],
                'Transportation Cost': dep_costbreakdown['transportation_cost'],
                'Penalty Cost': dep_costbreakdown['penalty_cost'],
                'Revenue': dep_costbreakdown['revenue']
            })

            linear_residual_ind = self.step2_ind_residuals  # Corrected reference to independent residuals
            ind_location, ind_obj, ind_revenue, real_demand, ind_costbreakdown = self.linear_ind_experiments(
                linear_residual_ind, 'insample', decision=None)
            # Add the result to 'Decision-independent'
            self.insample_results['Decision-independent']['linear_regression'] = [ind_location, ind_obj, ind_revenue,
                                                                                  real_demand]

            cost_breakdown_data.append({
                'Model Name': 'linear_regression',
                'Type': 'Decision-independent',
                'Fixed Cost': ind_costbreakdown['fixed_cost'],
                'Charger Cost': ind_costbreakdown['charger_cost'],
                'Transportation Cost': ind_costbreakdown['transportation_cost'],
                'Penalty Cost': ind_costbreakdown['penalty_cost'],
                'Revenue': ind_costbreakdown['revenue']
            })

            # Convert insample_results to a DataFrame
            data = []
            # Union of keys from both dictionaries
            for name in set(self.insample_results['Decision-dependent'].keys()).union(
                    self.insample_results['Decision-independent'].keys()):
                # Get results from both decision-dependent and decision-independent, or default to None if not present
                dep_result = self.insample_results['Decision-dependent'].get(name, [None, None, None, None])
                ind_result = self.insample_results['Decision-independent'].get(name, [None, None, None, None])

                data.append([name] + dep_result + ind_result)

            # Corrected columns: missing comma for 'Dep_zero_demand'
            df_columns = ['Model Name',
                          'Dep_Location', 'Dep_Obj', 'Dep_Revenue', 'Dep_increase_demand',
                          'Ind_Location', 'Ind_Obj', 'Ind_Revenue', 'Ind_demand']

            # Create DataFrame from results
            self.insample_df = pd.DataFrame(data, columns=df_columns)
            self.cost_breakdown_df = pd.DataFrame(cost_breakdown_data)

        '''


        # Initialize insample_results with 'Decision-dependent' and 'Decision-independent' as dictionaries
        self.insample_results = {
            'Decision-dependent': {},
            'Decision-independent': {}
        }

        cost_breakdown_data = []

        # First loop for decision-dependent models
        for name, model in self.best_models_dep.items():
            residuals_dep = self.residuals_dep[name]
            dep_location, dep_obj, dep_revenue, real_demand, dep_costbreakdown = self.dep_experiments(name,
                                                                                                      residuals_dep,
                                                                                                      'insample',
                                                                                                      decision=None)

            print('residuals dep length', len(residuals_dep))
            # Add the result to 'Decision-dependent'
            self.insample_results['Decision-dependent'][name] = [dep_location, dep_obj, dep_revenue,
                                                                 real_demand]

            cost_breakdown_data.append({
                'Model Name': name,
                'Type': 'Decision-dependent',
                'Fixed Cost': dep_costbreakdown['fixed_cost'],
                'Charger Cost': dep_costbreakdown['charger_cost'],
                'Transportation Cost': dep_costbreakdown['transportation_cost'],
                'Penalty Cost': dep_costbreakdown['penalty_cost'],
                'Revenue': dep_costbreakdown['revenue']
            })

        # Second loop for decision-independent models
        for name, model in self.best_models_ind.items():
            residuals_ind = self.residuals_ind[name]
            ind_location, ind_obj, ind_revenue, real_demand, ind_costbreakdown = self.ind_experiments(name,
                                                                                                      residuals_ind,
                                                                                                      'insample',
                                                                                                      decision=None)
            # Add the result to 'Decision-independent'
            self.insample_results['Decision-independent'][name] = [ind_location, ind_obj, ind_revenue,
                                                                   real_demand]
            cost_breakdown_data.append({
                'Model Name': name,
                'Type': 'Decision-independent',
                'Fixed Cost': ind_costbreakdown['fixed_cost'],
                'Charger Cost': ind_costbreakdown['charger_cost'],
                'Transportation Cost': ind_costbreakdown['transportation_cost'],
                'Penalty Cost': ind_costbreakdown['penalty_cost'],
                'Revenue': ind_costbreakdown['revenue']
            })

        # Handle linear regression results for decision-dependent and decision-independent
        linear_residual_dep = self.step2_dep_residuals
        dep_location, dep_obj, dep_revenue, real_demand, dep_costbreakdown = self.linear_dep_experiments(
            linear_residual_dep, 'insample', decision=None)
        # Add the result to 'Decision-dependent'
        self.insample_results['Decision-dependent']['linear_regression'] = [dep_location, dep_obj, dep_revenue,
                                                                            real_demand]
        cost_breakdown_data.append({
            'Model Name': 'linear_regression',
            'Type': 'Decision-dependent',
            'Fixed Cost': dep_costbreakdown['fixed_cost'],
            'Charger Cost': dep_costbreakdown['charger_cost'],
            'Transportation Cost': dep_costbreakdown['transportation_cost'],
            'Penalty Cost': dep_costbreakdown['penalty_cost'],
            'Revenue': dep_costbreakdown['revenue']
        })

        linear_residual_ind = self.step2_ind_residuals  # Corrected reference to independent residuals
        ind_location, ind_obj, ind_revenue, real_demand, ind_costbreakdown = self.linear_ind_experiments(
            linear_residual_ind, 'insample', decision=None)
        # Add the result to 'Decision-independent'
        self.insample_results['Decision-independent']['linear_regression'] = [ind_location, ind_obj, ind_revenue,
                                                                              real_demand]

        cost_breakdown_data.append({
            'Model Name': 'linear_regression',
            'Type': 'Decision-independent',
            'Fixed Cost': ind_costbreakdown['fixed_cost'],
            'Charger Cost': ind_costbreakdown['charger_cost'],
            'Transportation Cost': ind_costbreakdown['transportation_cost'],
            'Penalty Cost': ind_costbreakdown['penalty_cost'],
            'Revenue': ind_costbreakdown['revenue']
        })

        # Convert insample_results to a DataFrame
        data = []
        # Union of keys from both dictionaries
        for name in set(self.insample_results['Decision-dependent'].keys()).union(
                self.insample_results['Decision-independent'].keys()):
            # Get results from both decision-dependent and decision-independent, or default to None if not present
            dep_result = self.insample_results['Decision-dependent'].get(name, [None, None, None, None])
            ind_result = self.insample_results['Decision-independent'].get(name, [None, None, None, None])

            data.append([name] + dep_result + ind_result)

        # Corrected columns: missing comma for 'Dep_zero_demand'
        df_columns = ['Model Name',
                      'Dep_Location', 'Dep_Obj', 'Dep_Revenue', 'Dep_increase_demand',
                      'Ind_Location', 'Ind_Obj', 'Ind_Revenue', 'Ind_demand']

        # Create DataFrame from results
        self.insample_df = pd.DataFrame(data, columns=df_columns)
        self.cost_breakdown_df = pd.DataFrame(cost_breakdown_data)

    def linear_gurobi_experiments(self, residuals_dep, step1_coef, step1_intercept, step2_intercept, education_coef,
                                  gdp_coef, experiments_type):
        # 1. 基础参数准备 (保持原始大小写)
        random.seed(15)
        r = self.revenue
        f = self.fixedCost
        c = self.transportationCost
        chargerCost = self.chargerCost
        C = self.capacity
        p = self.penaltyCost
        num_potential = self.numFacilities
        num_zip = self.numCustomers

        # 2. 需求预测逻辑与标量处理
        residuals = list(residuals_dep)
        education_val = np.array(self.education)
        gdp_val = np.array(self.gdp)

        # 计算 Step 2 偏移量并扩展为数组
        predicted_shift = (education_coef * education_val) + (gdp_coef * gdp_val) + step2_intercept
        initial_demand = np.full(num_zip, 15000 + predicted_shift)

        # 3. 决策相关需求因子 (Step 1)
        orig_demand = pd.DataFrame(np.exp(step1_coef * self.experiments_distance + step1_intercept))
        self.orig_demand = orig_demand

        sce = len(residuals)
        exp_demand = {(i, j, k): orig_demand.iloc[i, j] for i in range(num_potential) for j in range(num_zip) for k in
                      range(sce)}
        prob = np.array([1.0 / sce] * sce)

        # 4. 构建 Gurobi 模型
        m = gp.Model('nonparametric_dep')
        m.setParam('MIPGap', 0.01)
        m.setParam('OutputFlag', 0)

        charger = m.addVars(num_potential, vtype=GRB.INTEGER, ub=10, lb=0, name='Charger')
        location = m.addVars(num_potential, vtype=GRB.BINARY, name='Location')
        assign = m.addVars(num_potential, num_zip, sce, lb=0.0, name='Assign')
        unmetdemand = m.addVars(num_zip, sce, lb=0.0, name='Unmet')
        demand_var = m.addVars(num_zip, sce, lb=-GRB.INFINITY, name='demand')

        # 约束
        m.addConstrs((charger[i] <= 10 * location[i] for i in range(num_potential)))
        m.addConstrs(
            (gp.quicksum(assign[i, j, k] for i in range(num_potential)) + unmetdemand[j, k] == demand_var[j, k] for j in
             range(num_zip) for k in range(sce)))
        m.addConstrs(
            (gp.quicksum(assign[i, j, k] for j in range(num_zip)) <= C[i] * charger[i] for i in range(num_potential) for
             k in range(sce)))
        m.addConstrs((demand_var[j, k] == initial_demand[j] + gp.quicksum(
            exp_demand[i, j, k] * charger[i] for i in range(num_potential)) + residuals[k] for j in range(num_zip) for k
                      in range(sce)))

        # 目标函数
        fixed_cost_expr = location.prod(f)
        charger_cost_expr = charger.prod(chargerCost)
        transportation_cost_expr = gp.quicksum(
            assign[i, j, k] * c.iloc[i, j] * prob[k] for i in range(num_potential) for j in range(num_zip) for k in
            range(sce))
        penalty_cost_expr = gp.quicksum(unmetdemand[j, k] * p[j] * prob[k] for j in range(num_zip) for k in range(sce))
        revenue_expr = gp.quicksum(r[j] * demand_var[j, k] * prob[k] for j in range(num_zip) for k in range(sce))

        m.setObjective(
            fixed_cost_expr + charger_cost_expr + transportation_cost_expr + penalty_cost_expr - revenue_expr,
            GRB.MINIMIZE)

        if experiments_type == 'oos':
            for i in range(num_potential):
                m.addConstr(charger[i] == self.dep_decision[i])

        m.optimize()

        # 5. 结果提取与转换为 DataFrame
        if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            charger_values = [charger[i].X for i in range(num_potential)]

            # calculate breakdown cost
            fixed_cost_value = fixed_cost_expr.getValue()
            charger_cost_value = charger_cost_expr.getValue()
            transportation_cost_value = transportation_cost_expr.getValue()
            penalty_cost_value = penalty_cost_expr.getValue()
            revenue_value = revenue_expr.getValue()

            # 创建结果字典
            results_dict = {
                'Experiment_Type': experiments_type,
                'Optimal_Obj': m.objVal,
                'Total_Revenue': revenue_expr.getValue(),
                'Fixed_Cost': fixed_cost_expr.getValue(),
                'Charger_Cost': charger_cost_expr.getValue(),
                'Transportation_Cost': transportation_cost_expr.getValue(),
                'Penalty_Cost': penalty_cost_expr.getValue(),
                'Total_Chargers': sum(charger_values),
                'Charger_Decisions': [charger_values],
                'fixed_cost_value': fixed_cost_value,
                'charger_cost_value': charger_cost_value,
                'transportation_cost_value': transportation_cost_value,
                'penalty_cost_value': penalty_cost_value,
                'revenue_value': revenue_value
            }

            if experiments_type == 'insample':
                self.dep_decision = charger_values

            # 转换为 DataFrame (单行)
            res_df = pd.DataFrame([results_dict])

            # 返回原始元组以便兼容旧代码，同时包含新生成的 DataFrame
            return charger_values, m.objVal, results_dict['Total_Revenue'], res_df
        else:
            return None, None, None, None


# %%
if __name__ == '__main__':
    # t_list = [1, 15, 30, 45, 60]
    t_list = [30]
    coef_results = pd.DataFrame()
    linear_regression_results = pd.DataFrame()
    step1_reg_df = pd.DataFrame()
    step2_linear_results = pd.DataFrame()
    step2_reg_results = pd.DataFrame()
    nonparametric_results = pd.DataFrame()
    experiments_results = pd.DataFrame()
    cost_breakdown_results = pd.DataFrame()  # Corrected typo in variable name
    #
    for t in t_list:
        day_effect_model = exponential_day_effect(
            t_threshold=t,
            distance_threshold=30,
            numFacilities=25,
            numCustomers=5
        )

        # # Running the series of methods for each threshold
        day_effect_model.data_processing()  # Process the data
        day_effect_model.step_1_data()  # Step 1 data processing
        start_time = time.time()
        step1_results = day_effect_model.step1_regression()  # Perform regression in step 1
        step1_results['time_threshold'] = t
        step1_reg_df = pd.concat([step1_reg_df, step1_results], ignore_index=True)
        day_effect_model.step2_data_func()  # Process step 2 data
        step2_regression_results, step2_linear_coef = day_effect_model.step2_linear_regression()  # Step 2 linear regression
        two_step_time = time.time() - start_time
        # gurobi_results = day_effect_model.one_step_regression()
        # nn_df = day_effect_model.nonparametric_verify()  # Step 2 NN regression
        # nn_experiments_results = day_effect_model.nn_experiments()
        # # # day_effect_model.plot_3d_multivariate()
        # # #
        # step2_regression_results['time_threshold'] = t
        # step2_linear_coef['time_threshold'] = t
        # # # # step2_linear_results = pd.concat([step2_linear_results, step2_linear_coef], ignore_index=True)
        # # # step2_reg_results = pd.concat([step2_reg_results, step2_regression_results], ignore_index=True)
        # #
        day_effect_model.nonparametric_tuning()
        nonparametric_df = day_effect_model.nonparametric_df
        # nonparametric_df['time_threshold'] = t
        # nonparametric_results = pd.concat([nonparametric_results, nonparametric_df], ignore_index=True)
        # # # #
        # day_effect_model.input_data()  # Input data
        # day_effect_model.generateSyntheticData(seed=425)  # Generate synthetic data
        # coeffs = gurobi_results['coefficients']
        # resids = gurobi_results['residuals']
        # #
        # # # # Call the optimization function (In-Sample)
        # # # charger_vals, obj, rev, res_df = day_effect_model.linear_gurobi_experiments(
        # # #     residuals_dep=resids,
        # # #     step1_coef=coeffs['beta_distance'],
        # # #     step1_intercept=coeffs['dep_intercept'],
        # # #     step2_intercept=coeffs['intercept'],
        # # #     education_coef=coeffs['beta_education'],
        # # #     gdp_coef=coeffs['beta_gdp'],
        # # #     experiments_type='insample'
        # # # )
        # #
        # # # day_effect_model.plot_ev_stations_and_customers()
        # day_effect_model.insample_experiments()  # Run in-sample experiments
        # insample_results = day_effect_model.insample_df
        # insample_results['time_threshold'] = t
        # # experiments_results = pd.concat([experiments_results, insample_results], ignore_index=True)
        # #
        # cost_breakdown = day_effect_model.cost_breakdown_df
        # # cost_breakdown['time_threshold'] = t
        # # cost_breakdown_results = pd.concat([cost_breakdown_results, cost_breakdown], ignore_index=True)
        # # day_effect_model.oos_experiments()
        # # day_effect_model.plot_ev_stations_and_customers()
        # # Assuming the results DataFrame is stored in day_effect_model.insample_df
        # # Add a column for the t_threshold value
# %%

'''
current_results_df = day_effect_model.insample_df.copy()
current_results_df['t_threshold'] = t_threshold
intercepts.append([t_threshold, day_effect_model.step1_intercept])
coefs.append([t_threshold, day_effect_model.step1_coef])
regression_results_df = day_effect_model.nonparametric_df.copy()
regression_results_df['t_threshold'] = t_threshold
regression_results.append(regression_results_df)

# Append the current DataFrame to the results list
results_df.append(current_results_df)


# day_effect_model.oos_experiments()
# day_effect_model.sensitivity_analysis()

# Concatenating the results DataFrames into one
combined_experiments_results_df = pd.concat(results_df, ignore_index=True)
combined_regression_results_df = pd.concat(regression_results, ignore_index=True)
combined_experiments_results_df['Dep_Location_Sum'] = combined_experiments_results_df.apply(lambda row: sum(row['Dep_Location']), axis=1)
combined_experiments_results_df['Ind_Location_Sum'] = combined_experiments_results_df.apply(lambda row: sum(row['Ind_Location']), axis=1)
'''
# %%

import pandas as pd
import numpy
import matplotlib.pyplot as plt
import glob
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import mode
import json

def read_df_week(week_id):
    """
        Reads one weeks csv into dataframe.

        :param week_id: id of week that is in the name of CSV file
        :return: returns pandas dataframe
    """
    single_week_path = r"..\unsupervised_dataset\scenario_week_example_"+ str(week_id) + r".csv"
    df_week = pd.read_csv(single_week_path, index_col=None, header=0)
    return df_week

def visualise_week(df_week):
    """
        Visualises data features of one week.

        :param df_week: week dataframe
    """
    fig = plt.figure(figsize=(16,8))
    plt.subplot(511)
    df_week["Load current "].plot(legend=True, color='k')
    plt.subplot(512)
    df_week["Pressure"].plot(legend=True, color='b')
    plt.subplot(513)
    df_week["Turbine current"].plot(legend=True, color='g')
    plt.subplot(514)
    df_week["Turbine speed"].plot(legend=True, color='c')
    plt.subplot(515)
    df_week["Turbine voltage"].plot(legend=True, color='y')
    

def preprocess_df(df_week):
    """
        adds new columns to the dataframe by smoothing features for future use and detection

        :param df_week: week dataframe
        :return: returns modified pandas dataframe
    """
    window = 5
    df_week["Pressure_sm"] = df_week["Pressure"].rolling(window=window).mean().fillna(method='bfill')
    df_week["Load current_sm"] = df_week["Load current "].rolling(window=window).mean().fillna(method='bfill')
    df_week["Turbine current_sm"] = df_week["Turbine current"].rolling(window=window).mean().fillna(method='bfill')
    df_week["Turbine voltage_sm"] = df_week["Turbine voltage"].rolling(window=window).mean().fillna(method='bfill')
    df_week["Turbine speed_sm"] = df_week["Turbine speed"].rolling(window=window).mean().fillna(method='bfill')

    return df_week 


def get_clusters(df_week, columns):
    """
        Uses Kmeans clustering algorithm to find 2 clusters, and adds cluster labels as a column to the dataframe.
        :param df_week: week dataframe
        :param columns: columns that are used for clustering
        :return: returns modified pandas dataframe
    """
    kmeans = KMeans(n_clusters=2, random_state=0).fit(df_week[columns])
    df_week['cluster_label'] = kmeans.labels_
    df_week['cluster_label'] = df_week['cluster_label'].rolling(window=80).apply(lambda x: mode(x)[0]).fillna(method='bfill')
    return df_week



def visualise_PCA(df_week, columns):
    """
        makes PCA on specified columns and visualises 2 components
        :param df_week: week dataframe
        :param columns: columns that are used for PCA
    """
    pca = PCA(n_components=2)
    pca.fit(df_week[columns])
    pca = PCA(n_components=2)
    components = pca.fit_transform(df_week[columns])

    plt.scatter(components[:, 0], components[:, 1],
                c=df_week['cluster_label'])
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    

def visualise_detected_clusters(dw_week):
    """
        visualises week features and detected clusters (detected leakages)
        :param df_week: week dataframe
    """
    fig = plt.figure(figsize=(16,8))
    plt.subplot(611)
    df_week["Load current "].plot(legend=True, color='k')
    plt.subplot(612)
    df_week["Pressure"].plot(legend=True, color='b')
    plt.subplot(613)
    df_week["Turbine current"].plot(legend=True, color='g')
    plt.subplot(614)
    df_week["Turbine speed"].plot(legend=True, color='c')
    plt.subplot(615)
    df_week["Turbine voltage"].plot(legend=True, color='y')
    plt.subplot(616)
    df_week["cluster_label"].plot(legend=True, color='y')
    
def add_leakage_labels(df_week):
    """
        checks if cluster labels correspond to the right state (leakage or non leakage) and corrects if its wrong
        :param df_week: week dataframe
        :return: returns modified pandas dataframe
    """
    df_label_group = df_week.groupby(by="cluster_label", dropna=True).mean().reset_index()
    if float(df_label_group.loc[df_label_group.cluster_label==1]["Pressure"]) < float(df_label_group.loc[df_label_group.cluster_label==0]["Pressure"]):
        #print("labels are ok")
        df_week["leakage"] = df_week["cluster_label"]
    else:
        df_week["leakage"] = df_week["cluster_label"].replace({1:0, 0:1})
        #print("switch kmeans labels")

    return df_week


def get_leakages_end_periods(df_week):
    """
        Gets timestamps of when state changes and construct list of states and timestamps.
        If we detected there is only one state we assign state based on average pressure.
        :param df_week: week dataframe
        :return: returns list of leakages and list of end periods in seconds
    """
    df_week["change_of_state"] = df_week["leakage"].diff().fillna(0)
    end_periods = df_week.loc[(df_week['change_of_state'] != 0)].index.tolist()
    leakages = []
    end_periods_seconds = []
    if len(end_periods) > 100:
        # TODO: naredi za primer ko je samo en cluster - samo eno stanje, trenutno hardcodan stanje no leak (0)
        
        if df_week["Pressure_sm"].mean() > 0.85:
            leakages.append(0)
        else:
            leakages.append(1)
        end_periods_seconds.append(604800)
    elif len(end_periods) ==0:
        leakages.append(int(df_week["leakage"].loc[df_week.index == 5]))
        end_periods_seconds.append(604800)
    else:
        for end_period in end_periods:
            leakages.append(int(df_week["leakage"].loc[df_week.index == end_period-1]))
            end_periods_seconds.append(end_period*10)
    return leakages, end_periods_seconds




def pipeline_one_week(week_id):
    """
        full pipeline for detecting leakages for a single week
        :param week_id: id of week for which we want to detect leakages
        :return: returns list of leakages and list of end periods in seconds
    """
    columns_smooth = ['Load current_sm', 'Pressure_sm', 'Turbine current_sm', 'Turbine speed_sm', 'Turbine voltage_sm']
    df_week = read_df_week(week_id)
    #visualise_week(df_week)
    df_week = preprocess_df(df_week)
    df_week = get_clusters(df_week, columns_smooth)

    df_week = add_leakage_labels(df_week)

    leakages, end_periods_seconds = get_leakages_end_periods(df_week)
    return leakages, end_periods_seconds

def get_phase1_result_for_all_files():
    """
        full pipeline for detecting leakages for all weeks.
        generates a json file that is used then for submission
        :param week_id: id of week for which we want to detect leakages
        :return: returns list of leakages and list of end periods in seconds
    """
    unsupervised_datasets_path = r'..\unsupervised_dataset'
    prediction_results = []

    for week in range(100):
        filename = unsupervised_datasets_path + r"\scenario_week_example_" + str(week) + r".csv"
        name = r"scenario_week_example_" + str(week) + r".csv"
        print(name)
        columns_smooth = ['Load current_sm', 'Pressure_sm', 'Turbine current_sm', 'Turbine speed_sm', 'Turbine voltage_sm']
        df_week = read_df_week(week)
        df_week = preprocess_df(df_week)
        df_week = get_clusters(df_week, columns_smooth)

        df_week = add_leakage_labels(df_week)


        leakages, end_periods_seconds = get_leakages_end_periods(df_week)
        
        prediction_results.append({
            "file_name" : name,
            "end_periods": end_periods_seconds,
            "leakages": leakages
        })
    
    json_results = {"prediction_results" : prediction_results}
    with open("../results_phase1.json", "w") as outfile:
        json.dump(json_results, outfile)
        
if __name__ == "__main__":
    get_phase1_result_for_all_files()
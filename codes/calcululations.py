import numpy as np
import pandas as pd


def calculate_polarization(df, param):
    """
    This method calculates polarization values.

    Parameters :
    - df : dataframe which consists of column : velocity
    - param : parameter, a dictionary consisting of all the variables

    returns :
    - averaged out polarization values
    """
    df = pd.DataFrame(df)
    time_steps = param["steps"]
    sum_of_polar = []
    population = param["population"]
    df.reset_index(inplace=True)
    df = df.sort_values(["t"])
    k = time_steps / 10
    k = int(k)
    # calculated from the time point where the system has stabilized into a polarized/swarm/torus state
    for i in range(k, (time_steps)):
        vel_at_each_sec = df["velocity"].loc[df["t"] == i]
        vel_sum = np.sum(vel_at_each_sec)
        # vel_x_sum = np.sum(np.sum(np.array(grouped['vel_x'].loc[grouped['t']==i]))) # of each agent
        # vel_y_sum = np.sum(np.sum(np.array(grouped['vel_y'].loc[grouped['t']==i]))) #I HAVE NO IDEA WHY THIS WORKS, IT'S NOT SUPPOSED TO BE LIKE THIS I WILL GET BACK TO IT
        norm_at_each_sec = np.linalg.norm(vel_sum)
        polarization_at_each = norm_at_each_sec / population
        sum_of_polar.append(polarization_at_each)
    polar = np.mean(sum_of_polar)
    print(polar)


def calculate_angular_momentum(df, param):
    """
    This method calculates angular momentum values at each frame.

    Parameters :
    - df : dataframe which consists of column : velocity
    - param : parameter, a dictionary consisting of all the variables

    returns :
    - a print statement if angular momentum at any frame is higher than 0.5
    """
    df = pd.DataFrame(df)
    time_steps = param["steps"]
    population = param["population"]
    df.reset_index(drop=True, inplace=True)

    df = df.sort_values(["t", 'obj_id'])
    sum_m_values = []
    for k in range(1, (time_steps + 1)):
        pos_at_each_sec = np.array(df["position"].loc[df["t"] == k])

        # vel_at_each_sec = df["velocity"].loc[df["t"] == k]

        pos_sum = np.array(np.sum(pos_at_each_sec))
        c_group = (pos_sum / population)
        

        sum_of_cross = []

        # add warning for <2

        for i in range(2, population + 1):
            pos_ = np.array(df["position"].loc[(df["t"] == k) & (df["obj_id"] == i)]) 
            pos_ = (pos_[0])
            r_ic = pos_ - c_group
            vel_i = np.array(df["velocity"].loc[(df["t"] == k) & (df["obj_id"] == i)])
            cross = np.cross(createUnitVector(r_ic), vel_i[0])
            sum_of_cross.append(cross)

        cross_sum = np.sum(sum_of_cross)

        norm_of_cross = np.linalg.norm(cross_sum)
        m_at_each = norm_of_cross / population
        # print(m_at_each)
        sum_m_values.append(m_at_each)
    for i in sum_m_values:
        if i > 0.5:
            print("Possibility of a torsion.")



def createUnitVector(vector):
    """
    This method normalizes vectors using np.linalg.norm.

    Parameters :
    - vector : a 1-D vector/numpy array/list

    """
    norm = np.linalg.norm(vector)
    if norm != 0:
        unit_vector = vector / norm
    else:
        unit_vector = vector
    return unit_vector


def return_split_x(velocity):
    """
    This method splits a [x,y] list into its component : x.

    Parameters :
    - velocity : takes in [x, y]

    """
    x_ = velocity[0]
    return x_


def return_split_y(velocity):
    """
    This method splits a [x,y] list into its component : y.

    Parameters :
    - velocity : takes in [x, y]

    returns :
    - y coordinate

    """
    y_ = velocity[1]
    return y_

def return_list(row):

    x = row['pos_x']
    y = row['pos_y']
    list_ = np.array([x, y])
    return list_

def updated_dataframe(results):
    """
    This method re-arranges the dataframes in two ways :
    - df is a dataframe consisting of 'vel_x', 'vel_y', 'pos_x' and 'pos_y'.
    - grouped is a dataframe consisting of each agent's velocity at each second in a list.

    """
    results.arrange_variables()
    df = results["variables"]["myFish"]
    df.reset_index(inplace=True)
    df["vel_x"] = df["velocity"].apply(return_split_x)
    df["vel_y"] = df["velocity"].apply(return_split_y)
    df["pos_x"] = df["position0"]
    df["pos_y"] = df["position1"]

    df['position'] = df.apply(return_list, axis = 1)

    df.reset_index(inplace=True)
    df = df.sort_values(["t", "obj_id"])

    grouped = df.groupby(by="t").agg(
        {
            "vel_x": lambda d: list(d),
            "vel_y": lambda d: list(d),
            "pos_x": lambda d: list(d),
            "pos_y": lambda d: list(d),
        }
    )
    grouped.reset_index(inplace=True)
    grouped.set_index("t")
    # df.to_csv('raw_data.csv')

    return df, grouped


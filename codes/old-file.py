from agentpy import Agent, Model, Space, AgentList, animate

import numpy as np
from agentpy import agent
from bokeh.plotting import figure, ColumnDataSource, output_file, save, show
from bokeh.models import HoverTool, Legend, Span
import matplotlib.pyplot as plt
import agentpy
import IPython

import pandas as pd

'''
Old file for storage.
'''

def createUnitVector(vector):
    norm = np.linalg.norm(vector)
    if norm != 0:
        unit_vector = vector/norm 
    else:
        unit_vector = vector
    return unit_vector    

class myFish(Agent):

    '''
Initializing a single fish.
'''
    def setup(self):
        
        self.velocity = createUnitVector(np.random.random(2) - 0.5) #the directions would be random, can't take from a gaussian, plut let's just only deal with unit vectors
         #distance of the repulsion zone
     #distance of the attraction zone
        # self.neighbor_r = 0
        # self.neighbor_a = 0
        # self.speed = p.speed
        # self.max_angle =  #maximum angle of turning

    def setupSpace(self, space):

        self.space = space #the "environment" of the agents that AgentPy takes in
        self.neighbors = space.neighbors #neighbors have the attribute space
        self.position = space.positions[self] #position of the object in the space

    def updateVelocity(self):
        '''
        if n_r > 0:
            position_vector = - np.sum(r_ij/norm(r_ij)
        '''
        position = self.position #position of the ith agent
        velocity = self.velocity 

        neighbors_in_repulsion_zone = self.neighbors(self, self.p.distance_r)
        neighbors_in_repulsion_zone = neighbors_in_repulsion_zone.to_list()
        neighbor_r = len(neighbors_in_repulsion_zone) #might not need it at all
        r_position = (neighbors_in_repulsion_zone.position)

        alpha_angle = self.p.angle_of_perception

        distance_o = self.p.distance_o
        neighbors_in_orientation_zone = (self.neighbors(self, distance_o)).to_list()

        for j in neighbors_in_orientation_zone:
            for i in neighbors_in_repulsion_zone:
                i_agent_id = i.id
                j_agent_id = j.id
                if i_agent_id == j_agent_id:
                    index = list(neighbors_in_orientation_zone).index(j)
                    del neighbors_in_orientation_zone[index]

        neighbor_o = len(neighbors_in_orientation_zone)
        o_velocity = (neighbors_in_orientation_zone.velocity)

        neighbors_in_attraction_zone = (self.neighbors(self, self.p.distance_a)).to_list() #using the neighbors attribute of each agent through agentpy
        # neighbors_in_attraction_zone = every_neighbors_in_attraction_zone - neighbors_in_orientation_zone - neighbors_in_repulsion_zone
        for i in neighbors_in_attraction_zone:
            for j in neighbors_in_repulsion_zone:
                i_agent_id = i.id
                j_agent_id = j.id

                if i_agent_id == j_agent_id:
                    index = list(neighbors_in_attraction_zone).index(i)
                    del neighbors_in_attraction_zone[index]

        for i in neighbors_in_attraction_zone:
            for j in neighbors_in_orientation_zone:
                i_agent_id = i.id
                j_agent_id = j.id

                if i_agent_id == j_agent_id:
                    index = list(neighbors_in_attraction_zone).index(i)
                    del neighbors_in_attraction_zone[index]

        #Add a warning if the r_o is greater than r_a

        neighbor_a = len(neighbors_in_attraction_zone)
        a_position = (neighbors_in_attraction_zone.position)

        if neighbor_r > 0 :
        #repulsion rule has the most priority
            # print(r_position, position)
            diff = np.array(r_position - position, dtype=float)
            
            
             #[cj-ci, ck - ci, and so on]
            norm = np.linalg.norm(diff, axis=1) #keeping dimensions, here we get one array of all the values of norm in an array so [[1],[2]] which we then divide by every "array in the array" in diff
            # print(norm)

            diff[norm > 0] = diff[norm > 0] / norm[norm > 0][:, np.newaxis]
            # print(diff)
            direction_r = np.sum(diff, axis=0, keepdims=True)
            direction = - direction_r[0] #taking the first/0th member of [[x, y, z]]
            # print(direction)
            # direction = createUnitVector(direction)
            # print(direction)


        else:
            '''
        Need to find a better way to do this but I had to take conditions for when neighbor_a is 0 and neighbor_o is not
        
        '''

            if (neighbor_o > 0) and (neighbor_a > 0):
                diff = np.array(o_velocity , dtype= float)
                
                norm = np.linalg.norm(diff, axis=1)
                diff[norm > 0] = diff[norm > 0] / norm[norm > 0][:, np.newaxis]

                direction_o = np.sum(diff, axis=0, keepdims=True)
                # direction_o = createUnitVector(direction_o)

                diff = np.array(a_position - (position), dtype= float)
                norm = np.linalg.norm(diff, axis=1)
                diff[norm > 0] = diff[norm > 0] / norm[norm > 0][:, np.newaxis]       
                direction_a = np.sum(diff, axis=0, keepdims=True)
                # direction_a = createUnitVector(direction_a)

                direction = (direction_o[0] + direction_a[0])*0.5
                direction = (direction)

            elif neighbor_a == 0 and neighbor_o > 0 : 

                diff = np.array(o_velocity, dtype= float)
                norm = np.linalg.norm(diff, axis=1)
                diff[norm > 0] = diff[norm > 0] / norm[norm > 0][:, np.newaxis]
                direction_o = np.sum(diff, axis=0, keepdims=True)
                direction = direction_o[0]
                direction = (direction)

            elif neighbor_o == 0 and neighbor_a > 0 :

                diff = np.array(a_position - (position), dtype=float)
                norm = np.linalg.norm(diff, axis=1)
                diff[norm > 0] = diff[norm > 0] / norm[norm > 0][:, np.newaxis]

                direction_a = np.sum(diff, axis=0, keepdims=True)

                direction = direction_a[0]


            else:
                direction = velocity
            # direction = createUnitVector(direction)

            direction_U = createUnitVector(direction)
            angle_between_dir_and_i = np.arccos(np.clip(np.dot(direction_U, velocity), -1.0, 1.0))

            if angle_between_dir_and_i <= self.p.max_angle:
                direction = direction 

            else:
                max_angle = (self.p.max_angle)
                
                cross = createUnitVector(np.linalg.det((velocity, direction)))   
                rot = np.array([[np.cos(max_angle*cross), -np.sin(max_angle*cross)], [np.sin(max_angle*cross), np.cos(max_angle*cross)]])  # positive keeps same matrix
                direction  = np.dot(rot, velocity)
                
        v4 = np.zeros(2)
        d = self.p.border_distance
        s = self.p.border_strength
        for i in range(2):
            if position[i] < d:
                v4[i] += s
            elif position[i] > self.space.shape[i] - d:
                v4[i] -= s
            # #Clipping the values given to arccos so that it remains between -1 and 1
        # direction = createUnitVector(direction)

        noise = np.random.normal(0, 0.05, 2) 

        velocity += direction + v4 + noise
        self.velocity = createUnitVector(velocity)
 
    def update_position(self):
        self.space.move_by(self, (self.velocity))

class fishModel(Model):
    def setup(self):
        """ Initializes the agents and network of the model. 
        
        """
        
        self.space = Space(self, shape=[self.p.size]*2)
        self.agents = AgentList(self, self.p.population, myFish)
        self.space.add_agents(self.agents, random=True)
        self.agents.setupSpace(self.space)

        
    def step(self):
        """ Defines the models' events per simulation step. """

        self.agents.updateVelocity()  
        self.agents.update_position() 
        self.agents.record('velocity')
        self.space.record_positions('position')
     #not giving me pos at every step?


param = {
    "distance_r" : 1,
    "distance_o" : 38,
    "distance_a" : 40,
    "speed" : 1,
    "size" : 50,
    'seed' : 7,
    'steps' : 50,
    "population" : 100,
    'border_strength' : 10,
    'border_distance': 1,
    'ndim' : 2,
    'max_angle' : 0.017,
    'angle_of_perception' :2
}


model = fishModel(param)

results = model.run()
import matplotlib.animation as anim

def animation_plot(m, p):
    projection = '3d' if p['ndim'] == 3 else None
    fig = plt.figure(figsize=(7,7))
    fig.patch.set_alpha(0.)
    ax = fig.add_subplot(111, projection=projection)

    animation = agentpy.animate(m(p), fig, ax, animation_plot_single)
    animation.save('scatter_3.gif',
         fps = 20, savefig_kwargs={'transparent': True, 'facecolor': 'none'})
    return IPython.display.HTML(animation.to_jshtml(fps=20))
    

def animation_plot_single(m, ax):

    ndim = m.p.ndim
    ax.set_title(f"Boids Flocking Model {ndim}D t={m.t}")
    pos = m.space.positions.values()
    pos = np.array(list(pos)).T  # Transform
    ax.clear()

    ax.scatter(*pos, s=4, c='w', marker="o")

    ax.set_frame_on(False)
    ax.set_xlim(0, m.p.size)
    ax.set_ylim(0, m.p.size)
    if ndim == 3:
        ax.set_zlim(0, m.p.size)
    
    ax.set_axis_off()



data = animation_plot(fishModel, param)

results.arrange_variables()
def return_same(x):
    return x

def return_split_x(velocity):
    x_ = velocity[0]
    return x_

def return_split_y(velocity):
    y_ = velocity[1]
    return y_

df = results['variables']['myFish']
df.reset_index(inplace = True)
df['vel_x'] = df['velocity'].apply(return_split_x)
df['vel_y'] = df['velocity'].apply(return_split_y)
# print(df)
df['pos_x'] = df['position0']
df['pos_y'] = df['position1']

df.to_csv('raw_data.csv')


# print(pos)

# df['pos_x'] = df['position'].apply(return_split_x)
# df['pos_y'] = df['position'].apply(return_split_y)


x_ = list(df['vel_x'])
y_ = list(df['vel_y'])

x_pos = list(df['pos_x'])
y_pos = list(df['pos_y'])
# output_file("line.html")
def return_list(row):
    new_pos = []
    new_pos.append(row['pos_x'])
    new_pos.append(row['pos_y'])
    return new_pos

df['position'] = df.apply(lambda row: return_list(row), axis=1)
# print(df['position'])
p = figure()

# add a circle renderer with a size, color, and alpha
p.circle(x_pos, y_pos, size=5, color="navy", alpha=0.5)

# show(p)

def agg_func(x):
    list_ = []
    list_.append(x)
    return list_
df.reset_index(inplace = True)
df = df.sort_values(['t', 'obj_id'])
def return_append(x):
    list_ = []
    list_.append(x)
    return list_
grouped = df.groupby(by='t').agg({'vel_x' : lambda d: list(d), 'vel_y' : lambda d: list(d), 'pos_x': lambda d: list(d), 'pos_y' : lambda d: list(d)})
grouped.reset_index(inplace = True)
grouped.set_index('t')

pos_x = grouped['pos_x'].loc[grouped['t']==1]
pos_y = grouped['pos_y'].loc[grouped['t']==1]
vel_x = grouped['vel_x'].loc[grouped['t']==1]
vel_y = grouped['vel_y'].loc[grouped['t']==1]

# vector_data = (pos_x, pos_y, vel_x, vel_y)
# vectorfield = hv.VectorField(vector_data)

# add a circle renderer with a size, color, and alpha
# output_file("line.html")
# for i in range(1, (param['steps'])):
#     p = figure()
#     pos_x = list(grouped['pos_x'].loc[grouped['t']==i])
#     pos_y = list(grouped['pos_y'].loc[grouped['t']==i])
#     p.circle(pos_x, pos_y, size=5, color="navy", alpha=0.5)

# show(p)
# pos_y = grouped['pos_y']['t'==1]
# vel_x = grouped['vel_x']['t'==1]
# vel_y = grouped['vel_y']['t'==1]






# data1 = animation_plot_quiver(fishModel, param)


with open("datafucj.html", "w") as file:
    file.write(data.data)

# with open("data_uh.html", "w") as file:
#     file.write(data1.data)

# pos_x = list(grouped['pos_x'])
# pos_y = list(grouped['pos_y'])
# U = list(grouped['vel_x'])
# V = list(grouped['vel_y'])

# fig, ax = plt.subplots(1,1)
# Q = ax.quiver(pos_x, pos_y, U, V, pivot='mid', color='r')

# ax.set_xlim(-param['size'],  param['size'])
# ax.set_ylim(-param['size'],  param['size'])
print(df)

def calculate_polarization(df):
    time_steps = param['steps']
    sum_of_polar = []
    population = param['population']
    k =  time_steps / 10
    k = int(k)
    for i in range(k, (time_steps)):
        vel_at_each_sec = df['velocity'].loc[df['t']==i]
        vel_sum = (np.sum(vel_at_each_sec))
        # vel_x_sum = np.sum(np.sum(np.array(grouped['vel_x'].loc[grouped['t']==i]))) # of each agent
        # vel_y_sum = np.sum(np.sum(np.array(grouped['vel_y'].loc[grouped['t']==i]))) #I HAVE NO IDEA WHY THIS WORKS, IT'S NOT SUPPOSED TO BE LIKE THIS I WILL GET BACK TO IT     
        norm_at_each_sec = np.linalg.norm(vel_sum)
        polarization_at_each = norm_at_each_sec/population
        sum_of_polar.append(polarization_at_each)
    polar = np.mean(sum_of_polar)
    print(polar)

calculate_polarization(df=df)

# def calculate_angular_momentum(df):
#     time_steps = param['steps']
#     population = param['population']
#     sum_m_values = []
#     for i in range(1, (time_steps)):
#         pos_at_each_sec = df['position'].loc[df['t'] == i]
#         print(pos_at_each_sec)
#         vel_at_each_sec = df['velocity'].loc[df['t']==i]
#         c_group = np.sum(np.sum(pos_at_each_sec))
#         c_group = c_group / population
#         print(np.array(pos_at_each_sec))
#         print(c_group)
#         r_ic = np.array(pos_at_each_sec) - c_group
#         cross = np.cross(r_ic, vel_at_each_sec)
#         cross_sum = np.sum(cross)
#         norm_of_cross = np.linalg.norm(cross_sum)
#         m_at_each = norm_of_cross/population
#         sum_m_values.append(m_at_each)
#     m = np.mean(sum_m_values)
#     print(m)

def calculate_angular_momentum(df):
    time_steps = param['steps']
    population = param['population']
    sum_m_values = []
    for k in range(1, (time_steps)):
        pos_at_each_sec = np.array(df['position'].loc[df['t'] == k])

        vel_at_each_sec = df['velocity'].loc[df['t']== k]

        pos_sum = np.sum(np.sum(pos_at_each_sec))
        c_group = pos_sum / population

        sum_of_cross = []
        for i in range(2, population):
            pos_ = np.array(df['position'].loc[df['t'] == k & df['obj_id'] == i])
            r_ic = pos_ - c_group
            vel_i = df['velocity'].loc[df['obj_id'] == i & df['t'] == k]
            cross = np.cross(r_ic, vel_i)
            sum_of_cross.append(cross)

        cross_sum = np.sum(sum_of_cross)

        norm_of_cross = np.linalg.norm(cross_sum)

        m_at_each = norm_of_cross/population

        sum_m_values.append(m_at_each)

    m = np.mean(sum_m_values)
    print(m)

# calculate_angular_momentum(df=df)


        


import plotly.graph_objects as go

# fig = go.Figure(data = go.Cone(

#     colorscale='Blues',
#     sizemode="absolute"
#     ))


# fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=0.8),
#                              camera_eye=dict(x=1.2, y=1.2, z=0.6)))

import plotly.figure_factory as ff

x=    (grouped['pos_x'].loc[grouped['t']==1]) # of each agent
    
y=    (grouped['pos_y'].loc[grouped['t']==1])
    
u=    (grouped['vel_x'].loc[grouped['t']==1])
v=    (grouped['vel_y'].loc[grouped['t']==1])


fig = ff.create_quiver(x, y, u, v,
                       scale=1.2, arrow_scale=1,
                       
                       line=dict(width=1.25, color='#8f180b'))

fig.update_layout(width=50, height=50,
                  title_text='Quiver Kinetics', title_x=0.5,
                  xaxis_title="Size", xaxis_range=[0, param['size']], 
                  yaxis_range=[0, param['size']],
                  yaxis_title="Size ",   
                 )

frames = []
for i in range(1, param['steps']):
    figaux = ff.create_quiver(x=(grouped['pos_x'].loc[grouped['t']==i]),
              y=((grouped['pos_y'].loc[grouped['t']==i])),
                 u=(grouped['vel_x'].loc[grouped['t']==i]),
              v = (grouped['vel_y'].loc[grouped['t']==i]), scale=1.2, arrow_scale=1,
                       )

    frames.append(go.Frame(data=[figaux.data[0]]))

fig.update_layout(updatemenus=[dict(type='buttons',
                  showactive=False,
                  buttons=[dict(
                            label='Play',
                            method='animate',
                            args=[None, dict(frame=dict(duration= param['steps'], redraw=False),
                                             transition=dict(duration=0),
                                             fromcurrent=True,
                                             mode='immediate')])])] )
fig.update(frames=frames)

   

# fig.show()

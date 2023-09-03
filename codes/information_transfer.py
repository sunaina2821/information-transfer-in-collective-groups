from agentpy import Agent, Model, Space, AgentList, animate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from quiver import *
from quiver import animation_plot
from calcululations import *


class myFish(Agent):

    """
    Initializing a single fish."""

    def setup(self):

        self.velocity = createUnitVector(
                np.random.random(2) - 0.5
            )  

        self.information = 0 # 0 for uninformed and 1 for informed
        self.distance_r = self.p.distance_r
        # the directions would be random, can't take from a gaussian, plut let's just only deal with unit vectors
        # distance of the repulsion zone
        # distance of the attraction zone

        self.speed = self.p.speed
        self.omega = self.p.omega_for_uninformed

        # self.max_angle =  #maximum angle of turning

    def setupSpace(self, space):

        self.space = space  # the "environment" of the agents that AgentPy takes in
        self.neighbors = space.neighbors  # neighbors have the attribute space
        self.position = space.positions[self]  # position of the object in the space

    def segregationOfAgents(self):
        """
        Updates the information of agents. 
        
        """

        proportion = param["proportion_of_informed"]
        population = param["population"]
        timesteps = param["steps"]
        individuals_nearest = proportion*population
        individuals_nearest = int(individuals_nearest)

        if self.id == 5:
            agent_perturb = self                
            for i in range(1, 5): #random distance around which they bend
                neighbors = agent_perturb.neighbors(agent_perturb, i)
                if self.information == 1:
                    break
                if len(neighbors) in range(individuals_nearest - 3, individuals_nearest + 5):
                        # print(len(neighbors))
                    break

            for i in neighbors:
                    i.omega = 1
                    i.information = 1
        
    def identify_agent_perturb(self):
        d = 0.08           
        omega = self.omega
        if self.information == 1:

            position = self.position
            neighbors = self.neighbors(self, 10)
            
            for i in neighbors:
                if i.id == 5:
                    perturb_position = i.position
                    diff = np.array(perturb_position - position , dtype=float)
                    norm = np.linalg.norm(
                diff)

                    diff[norm > 0] = diff[norm > 0] / norm[norm > 0][:, np.newaxis]
                    # direction_r = np.sum(diff, keepdims=True)
                    direction = - diff
                    self.velocity = 5*direction 

            # [cj-ci, ck - ci, and so on]
        
            # keeping dimensions, here we get one array of all the values of norm in an array so [[1],[2]] which we then divide by every "array in the array" in diff
            # print(norm)

            
            # print(diff)


            # new_omega = (1 - d)*omega + 0.4*d
            # self.omega = new_omega
            # omega = self.omega

            # o_velocity = neighbors.velocity

            # weight_o_velocity = np.mean(np.array(o_velocity*(omega))) #weighted velocity of neighbors
            # weighted_average_velocity = (1 - omega)*direction + weight_o_velocity #weighted velocity of private 
            # individual_weighted_velocity = (1 - d)*weighted_average_velocity + d*np.sign(weighted_average_velocity)

    def updateVelocity(self):

        """
        if n_r > 0:
            position_vector = - np.sum(r_ij/norm(r_ij)
        """
            
        position = self.position  # position of the ith agent
        velocity = self.velocity

        omega = self.omega

        d = 0.08

        if self.information == 1:
            new_omega = (1 - d)*omega + 0.4*d
            self.omega = new_omega
            omega = self.omega

        # private_information = (1 - omega)*np.array(velocity)

        '''
        neighbors = self.neighbors(self, 3)
        neighbors = neighbors.to_list()
        neighbor_no = len(neighbors) 
        neighbor_velocity = neighbors.velocity

        if neighbor_no > 1 : 
            sum_of_velocities = np.sum(a=neighbor_velocity, axis=0, keepdims=True)
            sum_of_velocities = sum_of_velocities[0]
        elif neighbor_no == 0:
            sum_of_velocities = list(np.zeros(2))
        else:
            sum_of_velocities = np.array((neighbor_velocity))
            sum_of_velocities = sum_of_velocities[0]

        sum_of_velocities = np.array(sum_of_velocities)

        social_information = (0.6*sum_of_velocities)/neighbor_no

        average_velocity = private_information + social_information
        
        d = 0.0819
        # # will add noise later

        velocity += (1 - d)*average_velocity + d*np.sign(average_velocity) 
    '''

        noise = np.random.normal(0, 0.05, 2)
        
        neighbors_in_repulsion_zone = self.neighbors(self, self.distance_r)
        neighbors_in_repulsion_zone = neighbors_in_repulsion_zone.to_list()
        neighbor_r = len(neighbors_in_repulsion_zone)  # might not need it at all
        r_position = neighbors_in_repulsion_zone.position

        alpha_angle = self.p.angle_of_perception

        distance_o = self.p.distance_o

        neighbors_in_orientation_zone = (self.neighbors(self, distance_o)).to_list()
        
        neighbors_in_attraction_zone = (
            self.neighbors(self, self.p.distance_a)
        ).to_list()  

        for j in neighbors_in_orientation_zone:
            for i in neighbors_in_repulsion_zone:
                i_agent_id = i.id
                j_agent_id = j.id
                if i_agent_id == j_agent_id:
                    index = list(neighbors_in_orientation_zone).index(j)
                    del neighbors_in_orientation_zone[index]

        for (i,k) in zip(neighbors_in_orientation_zone, neighbors_in_attraction_zone):
            if i.id == 5:
                    index = list(neighbors_in_orientation_zone).index(i)
                    del neighbors_in_orientation_zone[index]
            if k.id == 5:
                    index = list(neighbors_in_attraction_zone).index(k)
                    del neighbors_in_attraction_zone[index]

        # neighbors_in_orientation_zone.insert(1, self)
        neighbor_o = len(neighbors_in_orientation_zone)
        o_velocity = neighbors_in_orientation_zone.velocity


        # using the neighbors attribute of each agent through agentpy
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

        # Add a warning if the r_o is greater than r_a

        neighbor_a = len(neighbors_in_attraction_zone)
        a_position = neighbors_in_attraction_zone.position

        if neighbor_r > 0:
            # repulsion rule has the most priority
            # print(r_position, position)
        
            diff = np.array(r_position - position, dtype=float)
            # [cj-ci, ck - ci, and so on]
            
            norm = np.linalg.norm(
                diff, axis=1
            )  
            
            # keeping dimensions, here we get one array of all the values of norm in an array so [[1],[2]] which we then divide by every "array in the array" in diff
            # print(norm)

            diff[norm > 0] = diff[norm > 0] / norm[norm > 0][:, np.newaxis]
            # print(diff)
            direction_r = np.sum(diff, axis=0, keepdims=True)
            direction = -direction_r[0]  # taking the first/0th member of [[x, y, z]]

        else:
            """
            Need to find a better way to do this but I had to take conditions for when neighbor_a is 0 and neighbor_o is not

            """

            if (neighbor_o > 0) and (neighbor_a > 0):
                
                o_velocity = np.array(o_velocity) #neighbor's velocity
                weight_o_velocity = np.mean(o_velocity*(omega), axis = 0, keepdims=True) #weighted velocity of neighbors

                weighted_average_velocity = (1 - omega)*velocity + weight_o_velocity #weighted velocity of private 

                individual_weighted_velocity = (1 - d)*weighted_average_velocity + d*np.sign(weighted_average_velocity)
                
                
                # diff = np.array(individual_weighted_velocity, dtype=float)

                # norm = np.linalg.norm(diff)
                # diff[norm > 0] = diff[norm > 0] / norm[norm > 0][:, np.newaxis]

                # direction_o = np.sum(diff, axis=0, keepdims=True)
                # direction_o = createUnitVector(direction_o)
                direction_o = individual_weighted_velocity
                direction_o = createUnitVector(direction_o)

                diff = np.array(a_position - (position), dtype=float)
                norm = np.linalg.norm(diff, axis=1)
                diff[norm > 0] = diff[norm > 0] / norm[norm > 0][:, np.newaxis]
                direction_a = np.sum(diff, axis=0, keepdims=True)
                # direction_a = createUnitVector(direction_a)

                direction = (direction_o[0] + direction_a[0]) * 0.5
                direction = direction

            elif neighbor_a == 0 and neighbor_o > 0:

                o_velocity = np.array(o_velocity) #neighbor's velocity
                weight_o_velocity = np.mean(o_velocity*(omega), axis= 0 , keepdims= True)
                weighted_average_velocity = (1 - omega)*velocity + weight_o_velocity
                individual_weighted_velocity = (1 - d)*weighted_average_velocity + d*np.sign(weighted_average_velocity)

                # diff = np.array(individual_weighted_velocity, dtype=float)
                # norm = np.linalg.norm(diff, axis=1)
                # diff[norm > 0] = diff[norm > 0] / norm[norm > 0][:, np.newaxis]
                # direction_o = np.sum(diff, axis=0, keepdims=True)

                direction = direction_o[0]
                direction = direction

            elif neighbor_o == 0 and neighbor_a > 0:

                diff = np.array(a_position - (position), dtype=float)
                norm = np.linalg.norm(diff, axis=1)
                diff[norm > 0] = diff[norm > 0] / norm[norm > 0][:, np.newaxis]

                direction_a = np.sum(diff, axis=0, keepdims=True)

                direction = direction_a[0]

            else:
                direction = velocity
            # direction = createUnitVector(direction)

            direction_U = createUnitVector(direction)
            angle_between_dir_and_i = np.arccos(
                np.clip(np.dot(direction_U, velocity), -1.0, 1.0)
            )

            if angle_between_dir_and_i <= self.p.max_angle:
                direction = direction

            else:
                max_angle = self.p.max_angle
                cross = createUnitVector(np.linalg.det((velocity, direction)))
                rot = np.array(
                    [
                        [np.cos(max_angle * cross), -np.sin(max_angle * cross)],
                        [np.sin(max_angle * cross), np.cos(max_angle * cross)],
                    ]
                )  # positive keeps same matrix
                direction = np.dot(rot, velocity)

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

        velocity = velocity + direction + noise + v4 

        if self.id != 5:    
            self.velocity = createUnitVector(velocity)
            
        else:
            self.velocity = np.zeros(2)

    def update_position(self):    

        self.velocity = self.speed*self.velocity
        
        if self.id != 5:
            self.space.move_by(self, (self.velocity))
        else:
            self.space.move_to(self, (np.array([25, 25])))


class fishModel(Model):
    def setup(self):
        """Initializes the agents and network of the model."""
        self.space = Space(self, shape=[self.p.size] * 2)
        self.agents = AgentList(self, self.p.population, myFish)
        
        # self.agent_perturb = Agent(self)

        #might be able to add "density" as an option here

        self.space.add_agents(self.agents, random=True)
        # self.space.add_agents(self.agent_perturb, [25,25])
                    
        self.agents.setupSpace(self.space)
        # self.agents.segregationOfAgents()
        # self.agents.identify_agent_perturb()
        # self.agents_perturb.setupSpace(self.space)


    def step(self):

        """Defines the models' events per simulation step."""
        # self.agents.velocity_formulation()
        self.agents.segregationOfAgents()
        self.agents.identify_agent_perturb()
        self.agents.updateVelocity()
        self.agents.update_position()
        self.agents.record("velocity")
        self.space.record_positions("position")

if __name__ == "__main__":

    param = {
        "distance_r": 1,
        "range_of_size": 200,
        "distance_o": 5,
        "distance_a": 30,
        "speed": 1,
        "size": 50,
        "seed": 30,
        "steps": 200,
        "population": 100,
        "border_strength": 2,
        "border_distance": 1,
        "ndim": 2,
        "max_angle": 0.28,
        "angle_of_perception": 2,
        "proportion_of_informed" : 0.1,
        "omega_for_uninformed" : 0.4,
    }

    list_ = []

    model = fishModel(param)
    results = model.run()
    df, grouped = updated_dataframe(results)
    print(df['velocity'])
    m = calculate_angular_momentum(df, param)

    p = calculate_polarization(df, param)

    data = animation_plot(fishModel, param)

    with open("scatter_plot.html", "w") as file:
        file.write(data.data)

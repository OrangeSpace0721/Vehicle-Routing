# -*- coding: utf-8 -*-
"""
Created on Wed May 10 16:32:28 2023

@author: 31196462
"""
import numpy as np
import pandas as pd
import itertools
import networkx as nx
import copy as cp
import random as rd

# Caculate the Euclidean distance between 2 points
def dist(x1, x2):
    return np.sqrt((x2[0]-x1[0])**2 + (x2[1]-x1[1])**2)

# Define the capacity constraint per vehicle
Q = np.array([[200]])
    
# Read the data
df = pd.read_csv("multi depot data.csv", index_col=("CUST_NO"))

# Seperate depot and customer data
df_depots = df[df["TYPE"] == "D"]
df_customer = df[df["TYPE"] == "C"]

cluster = len(df_depots)

# Calculate the distance of each customer to the different depots
df_distance = pd.DataFrame(np.zeros((len(df_customer), cluster)), columns = df_depots.index, index = df_customer.index)

for cluster in range(cluster):
    depot_coordinate = df_depots.iloc[cluster, 0:2]
    for customer in range(len(df_customer)):
        customer_coordinate = df_customer.iloc[customer, 0:2]
        df_distance.iloc[customer, cluster] = dist(depot_coordinate, customer_coordinate)
        
# Assign the customer to the minimum distance depot into a disctionary
minimum_distance_dict = {}

for k in range(cluster+1):
     key = "Depot{0}".format(k)
     value = df_customer.loc[df_distance.iloc[:, k] == df_distance.min(axis=1)]
     minimum_distance_dict[key] = value
     
# Calculate the distance matrix
def distmatrix(df):
    n = len(df)

    d = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            x1 = df.iloc[i,0:2]
            x2 = df.iloc[j,0:2]
            dij = dist(x1,x2)
            d[i,j] = np.inf if i == j else dij
            
    return d

distance_matrix_dict = {}

for i in range(cluster+1):
    key = "Depot{0}".format(i)
    new_row = pd.DataFrame(df_depots.iloc[i, :]).T
    minimum_distance_dict[key] = pd.concat([new_row, minimum_distance_dict[key]])
    value = distmatrix(minimum_distance_dict[key])
    distance_matrix_dict[key] = value


# Define the demand list, the idex of q represents
# the demand on each node
q = np.array([df['DEMAND']]) 

    
# Calculate the cost of the original route
def costlist(viable_routes, d):
    
    cost = np.zeros(len(viable_routes))
    
    for i in range(len(viable_routes)):
        for j in range(len(viable_routes[i])-1):
            cost[i] += d[viable_routes[i][j], viable_routes[i][j+1]]
            
    total_cost = sum(cost)
    
    return cost, total_cost

# Local Search - 2 opt
def opt2(route, node1, node2):
    
    altered_route = cp.deepcopy(route)
    altered_route[node1+1:node2] = reversed(route[node1+1:node2])
    
    return altered_route

def cost(route, d):
    
    cost = 0

    for j in range(len(route)-1):
        cost += d[route[j], route[j+1]]
        
    return cost

def edge(route):
    
    n = len(route) - 1
    edge = []
    
    for nodes in range(n):
        e = (route[nodes], route[nodes+1])
        edge.append(e)
    
    return edge

def arc(best_routes):
    arcs = []
    
    for routes in best_routes:
        arc = edge(routes)
        arcs.append(arc)
    
    arcs = [item for sublist in arcs for item in sublist]
        
    return arcs

def compare(route1, route2):
    set1 = set(route1)
    set2 = set(route2)
    
    if len(set1.intersection(set2)) < 0.5 * len(set1):
        return True
    
    else:
        return False
    
def convert_to_savings(arcs, savings):
    savings_list = []
    
    for routes in arcs:
        arc_list = []
        
        for arc in routes:
            arc_list.append(savings[arc[0], arc[1]])
            
        savings_list.append(arc_list)
        
    return savings_list

def return_indices(array, matrix):
    indices = []
    
    for items in array:
        index = np.where(matrix == items)
        indices.append(index[1][0])
    
    return indices

def entropy(solutions_generated, n):
    entropy_matrix = np.zeros((n,n))
    
    for solutions in solutions_generated:
        for arcs in solutions:
            entropy_matrix[arcs] += 1
    
    e = -( ( (entropy_matrix/100) * np.log(entropy_matrix/100 + 1)).sum() / ( n * np.log(n) ) )
    
    return e

def ruin(d, routes, method = "radial", F = 0.3):
    # Number of customer nodes
    n = len(d) - 1
    
    # Number of ruined nodes
    A = rd.randint(0, round(F * n))
    
    match method:
        
        case "radial":
        # Randomly choose a centre and k nearest neighbour to be removed from service
            centre = rd.randint(1, n)
            distance_list = list(d[centre, :])
            distance_list.sort()
            
            nearest_distance = distance_list[0:A]
            
            nearest_neighbour = return_indices(nearest_distance, d)
            
            for route in range(len(routes)):
                for node in nearest_neighbour:
                    if node != 0:
                        if node in routes[route]:
                            routes[route].remove(node)
            
            return nearest_neighbour, routes
                            
        case "random":
        # Randomly choosen nodes will be removed from service
            random_selection = rd.sample(range(1, n), A)
            
            for route in range(len(routes)):
                for node in random_selection:
                    if node != 0:
                        if node in routes[route]:
                            routes[route].remove(node)
                            
            return random_selection, routes

def recreate(d, nodes, partial_solution):
    route_demand = []
    
    # Create empty routes
    for i in range(len(nodes)):
        partial_solution.append([0,0])
    
    # Calculate the demand of the routes
    for route in partial_solution:
        demand = 0
        
        for node in route:
            demand += q[0,node]
    
        route_demand.append(demand)
    
    # Recreate the solution using best insertion
    while len(nodes) != 0:
        best_profit = -np.inf
        
        for node in nodes:
            for route in range(len(partial_solution)):
                for k in range(len(partial_solution[route])-1):
                    i, j = partial_solution[route][k], partial_solution[route][k+1]
                    
                    # Starting a new route is profit negative
                    if i == j:
                        profit = - d[i, node] - d[node, j]
                    else:
                        profit = d[i,j] - d[i, node] - d[node, j]
                    
                    if route_demand[route] + q[0,node] <= Q:
                        if profit > best_profit:
                            best_profit = cp.deepcopy(profit)
                            best_route = cp.deepcopy(route)
                            best_k = cp.deepcopy(k)
                            best_node = cp.deepcopy(node)
        
        # Insert the best node and remove from the list
        partial_solution[best_route].insert(best_k+1, best_node)
        route_demand[best_route] += q[0,best_node]
        nodes.remove(best_node)
    
    # Remove empty routes
    partial_solution = [route for route in partial_solution if len(route) > 2]
    
    return partial_solution

def clarke_and_wright(d, q, Q, choosen_alpha, bias = "random"):
    # Perform the savings heuristic to generate a feasible cycle
    n = d.shape[0]
    components = np.arange(n)
    
    # Calculate the savings matrix
    savings = np.zeros((n, n))
    for i in range(1, n):
        for j in range(1, n):
            # Ignore an arc if the two nodes in question exceeds the capacity
            check = []
            
            for p in range(np.shape(Q)[1]):
                check.append( (q[:, i] + q[:, j] <= Q[:, p]).all() )
                
            if any(check) == True:
                s = d[i, 0] + d[0, j] - d[i, j]
                savings[i, j] = s
            
            if i == j:
                savings[i, j] = 0
                        
    # Define nodes and basic routes
    nodes = np.arange(n)
    routes = [[0, i, 0] for i in nodes[1:]]
            
    # Connect the routes by checking for maximum savings until all nodes are connected
    # or there are no more savings to be had
    entry_requirement = list(np.ones((n)))
    exist_requirement = list(np.ones((n)))
    
    
    while len(np.unique(components)) > 1:
        
        # Create a masked array so we only select none zero min savings
        masked_savings = np.ma.masked_array(savings, mask=(savings==0))
        
        max_saving = np.max(savings)
        min_saving = np.min(masked_savings)
        
        if max_saving == 0:
            break
        
        # If alpha = 1 it will be a random construction, if alpha = 0 it will be a deterministic c&w
        threshold = max_saving - choosen_alpha*(max_saving - min_saving)
        
        savings_list = savings.flatten()
        max_savings_list = []
        
        # Select the restricted canidate list based on the quality of the canidate
        # determined by the threshold
        for items in savings_list:
            if items >= threshold and items <= max_saving:
                max_savings_list.append(items)
        
        
        max_savings_list.sort(reverse=True)
        max_savings_lists_len = len(max_savings_list)
        

        match bias:
            # Random bias, all elements of RCL are given the same probability
            case "random":
                probability = list(np.ones(max_savings_lists_len) / max_savings_lists_len)
            
            # Linear bias = 1/r
            case "linear":
                bias_array = np.ones(max_savings_lists_len) / (np.arange(max_savings_lists_len) + 1)
                probability = list(bias_array / sum(bias_array))
            
            # Inverse log bias = log^-1(r+1)
            case "log":
                bias_array = np.log(np.arange(max_savings_lists_len)+2)
                probability = list(bias_array / sum(bias_array))
            
            # Negative exponential bias = e^(-r)
            case "exponential":
                bias_array = np.exp(-(np.arange(max_savings_lists_len)+1))
                probability = list(bias_array / sum(bias_array))
                
        # Choice a random savings based on the bias distribution
        if choosen_alpha == 0:
            selected_savings = max_saving
        
        else:
            selected_savings = rd.choices(max_savings_list, probability)
        
        index = np.where(savings == selected_savings)
        i, j = index[0][0], index[1][0]
        
        if components[i] != components[j]:
            # Merge the routes for the two nodes
            new_route = routes[i-1] + routes[j-1]
            unique = np.unique(new_route)
            demand = np.zeros(Q.shape[0])
            
            # If the capacity constraint is violated, ignore the route
            for u in unique:
                demand += q[:, u]
            
            check = []
            
            for p in range(np.shape(Q)[1]):
                check.append( (demand <= Q[:, p]).all() )
                
            if any(check) == True and entry_requirement[j] == 1 and exist_requirement[i] == 1:
                components[components == components[j]] = components[i]
                savings[i, :] = 0
                savings[:, j] = 0
                entry_requirement[j] = 0
                exist_requirement[i] = 0
                for l in range(len(new_route)):
                    if new_route[l] != 0:
                        routes[new_route[l]-1] = new_route
    
            else:
                savings[i,j] = 0
        
        else:
            savings[i, j] = 0
    # Remove empty routes
    routes = [route for route in routes if len(route) > 0]
    #    return routes
    
    for i in range(len(routes)):
        j = routes[i]
        if len(j) < 4:
            # Remove infeasible route
            demand = q[:, j[1]]
            check = []
        
            for p in range(np.shape(Q)[1]):
                check.append( (demand <= Q[:, p]).all() )
            
            
            if any(check) == False:
                routes[i] = []
    
    # Remove empty routes
    routes = [route for route in routes if len(route) > 0]
    
    unique_routes = []
    [unique_routes.append(item) for item in routes if item not in unique_routes]
    
    # Collect the lists of viable routes (more than the basic 0-i-0 route)
    viable_routes = []
    for i in unique_routes:
            viable_routes.append(i)
    
    # Reformat the list for easier readibility and calculation
    for i in range(len(viable_routes)):
        viable_routes[i] = list(pd.unique(viable_routes[i]))
        viable_routes[i].append(0)

                
    # Perform 2-opt
    for routes in range(len(viable_routes)):
        old_route_cost = cost(viable_routes[routes], d)
        for nodes1 in range(len(viable_routes[routes])):
            if nodes1 > 0:
                for nodes2 in range(nodes1, len(viable_routes[routes])):
                    if nodes2 < len(viable_routes[routes]):
                        altered_route = opt2(viable_routes[routes], nodes1, nodes2)
                        new_route_cost = cost(altered_route, d)
                        if new_route_cost < old_route_cost:
                            viable_routes[routes] = altered_route
    
    new_cost, new_total_cost = costlist(viable_routes, d)
    
    return new_total_cost, viable_routes


def reactive_grasp(alpha, max_iteration, warm_up, d, bias = "random"):

    i = 0
    j = 0
    alpha_length = len(alpha)
    distribution = list(np.ones(alpha_length)/alpha_length)
    
    choosen_count = list(np.zeros(alpha_length))
    average_cost = list(np.zeros(alpha_length))
    cost_array = []
    
    # The update parameter for the reactive grasp
    p = list(np.zeros(alpha_length))
    best_cost = np.Inf
    best_routes = []
    
    while i <= max_iteration:
        
        if i >= warm_up:
            # Reevaluate alpha distribution every 20 interation
            if i % 20 == 0:
                true_average = np.array(average_cost) / np.array(choosen_count)
                p = (best_cost / true_average)**10
                distribution = p/sum(p)
        
        if i < warm_up:
            # During warm up, try every alpha value equally 
            if i % (warm_up/alpha_length) == 0:
                choosen_alpha = alpha[j]
                j += 1
        
        else:
            # Choose the hyperparameter alpha by a reactive distribution
            choosen_alpha = rd.choices(alpha, distribution)[0]
        
        cost, routes = clarke_and_wright(d, q, Q, choosen_alpha, bias)
    
        # Update the best cost of the route found
        
        if cost < best_cost:
            best_cost = cp.deepcopy(cost)
            print(best_cost)
            best_routes = cp.deepcopy(routes)
            print(best_routes)
            
        # Update the average cost list corresponding to the alpha list
        
        index = alpha.index(choosen_alpha)
        choosen_count[index] += 1
        average_cost[index] += cost
        cost_array.append(cost)

        i += 1
    
    variance = np.std(cost_array)**2
    
    return best_routes, best_cost, true_average, distribution, variance


def simulated_annealing(solution, solution_cost, variance, max_iteration, cooling_rate, d):
    # Decision of value of standard_deviation can massively affect the quality of solution
    i = 0
    tempt = variance
    best_cost = cp.deepcopy(solution_cost)
    best_routes = cp.deepcopy(solution)
    
    while i < max_iteration:
        
        if tempt < 10**(-6):
            break
        
        #Randomly choose a ruin method to use
        method = rd.randint(1, 2)
        ruined_routes = cp.deepcopy(best_routes)
        match method:
            case 1:
                # Ruin and recreate using radial ruin
                nodes, routes = ruin(d, ruined_routes)
                
            case 2:
                # Ruin and recreate using random ruin
                nodes, routes = ruin(d, ruined_routes, "random")
        
        new_routes = recreate(d, nodes, routes)
        
        # Perform 2-opt
        for routes in range(len(new_routes)):
            old_route_cost = cost(new_routes[routes], d)
            for nodes1 in range(len(new_routes[routes])):
                if nodes1 > 0:
                    for nodes2 in range(nodes1, len(new_routes[routes])):
                        if nodes2 < len(new_routes[routes]):
                            altered_route = opt2(new_routes[routes], nodes1, nodes2)
                            new_route_cost = cost(altered_route, d)
                            if new_route_cost < old_route_cost:
                                new_routes[routes] = altered_route
        
        new_cost = costlist(new_routes, d)[1]
        
        if best_cost > new_cost:
            # Always accept if the new solution is better
            best_cost = cp.deepcopy(new_cost)
            best_routes = cp.deepcopy(new_routes)
            
        else:
            # Accept a worst solution with a given probability
            accept_probability = np.exp(-(new_cost-best_cost)/tempt)
            probability = [accept_probability, 1-accept_probability]
            choice = [1, 0]
            decision = rd.choices(choice, probability)
            
            if decision == 1:
                best_cost = cp.deepcopy(new_cost)
                best_routes = cp.deepcopy(new_routes)
        
        # Reduce the temperature
        tempt *= cooling_rate
        i += 1
        
    return best_cost, best_routes


alpha = list(np.arange(11)/10)

'''
best_routes, best_cost, true_average, distribution, variance = reactive_grasp(alpha, 200, 22, distance_matrix_dict["Depot2"], bias = "exponential")

solution_cost, solution = simulated_annealing(best_routes, best_cost, variance, 1000, 0.95, distance_matrix_dict["Depot2"])
'''
results = []
for i in range(cluster+1):
    output1 = reactive_grasp(alpha, 200, 22, distance_matrix_dict["Depot{0}".format(i)], bias = "exponential")
    output2 = simulated_annealing(output1[0], output1[1], output1[4], 1000, 0.95, distance_matrix_dict["Depot{0}".format(i)])
    
    # Reformat the route according to original label
    for routes in range(len(output2[1])):
        for nodes in range(len(output2[1][routes])):
            output2[1][routes][nodes] = minimum_distance_dict["Depot{0}".format(i)].index[output2[1][routes][nodes]]
            
    results.append(output2)
        
# Local Search - Multiroute


'''
# Visualise the routes

G = nx.Graph()
G.add_nodes_from(list(df.index))

arcs = []

def edge(route):
    
    n = len(route) - 1
    edge = []
    
    for nodes in range(n):
        e = (route[nodes], routes[nodes+1])
        edge.append(e)
    
    return edge

for routes in best_routes:
    arc = edge(routes)
    arcs.append(arc)

arcs = [item for sublist in arcs for item in sublist]

G.add_edges_from(arcs)

pos = df[['XCOORD', 'YCOORD']].apply(tuple, axis=1).to_dict()


nx.write_gexf(G, "reactive_network.gexf")
'''
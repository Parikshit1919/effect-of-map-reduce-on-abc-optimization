import numpy as np
import random
import operator
import time

class Centroid():
    def __init__(self, cl, acc):
        self.cl = cl
        self.acc = acc
        self.count = 1

    def append(self, data):
        for i, val in enumerate(self.acc):
            self.acc[i] += data[i]
            self.count += 1

    def getCentroid(self):
        return self.acc / self.count



def readDatabase(filename, has_id, class_position):
    filepath = 'databases\\' + filename
    with open(filepath) as f:
        
        lines = (line for line in f if '?' not in line)
        dataset = np.loadtxt(lines, delimiter = ',')

    
    np.random.shuffle(dataset)

    
    if class_position == 'first':
        classes = dataset[:, 0]
        dataset = np.delete(dataset, 0, axis = 1)
    else:   
        classes = dataset[:, -1]
        dataset = np.delete(dataset, -1, axis = 1)

    if has_id:
        
        dataset = np.delete(dataset, 0, axis = 1)

    
    arr_max = np.max(dataset, axis = 0)
    arr_min = np.min(dataset, axis = 0) 

    rows, cols = np.shape(dataset)
    for i in range(rows):
        for j in range(cols):
            dataset[i][j] = (dataset[i][j] - arr_min[j]) / (arr_max[j] - arr_min[j])

    return dataset, classes

def determineCentroids(dataset, classes):
    rows, cols = np.shape(dataset)

    stats = {}

    for i, row in enumerate(dataset):
        class_id = str(classes[i])
        if class_id in stats:
            stats[class_id].append(row)
        else:
            stats[class_id] = Centroid(classes[i], row)

    centroids = {}
    for key in stats:
        centroids[key] = stats[key].getCentroid()

    return stats, centroids


def euclidianDistance(a, b):    
    diff_sqrt = [(x - y)**2 for x, y in zip(a, b)]

    return np.sqrt(np.sum(diff_sqrt))

def costFunction(dataset, classes, cl, centroid):
    
    distances_sum = 0
    count = 0
    for i, d in enumerate(dataset):
        if str(classes[i]) == cl: 
            distances_sum += euclidianDistance(d, centroids[cl])
            count += 1

    return distances_sum / count

def fitnessFunction(costs):
    fitness = costs.copy()
    for key in fitness:
        fitness[key] = 1/(1 + costs[key])

    return fitness

def rouletteWheelFunction(P):
    p_sorted_asc = sorted(P.items(), key = operator.itemgetter(1))
    p_sorted_desc = dict(reversed(p_sorted_asc))

    pick = np.random.uniform(0, 1)
    current = 0
    for key in p_sorted_desc:
        current += p_sorted_desc[key]
        if current > pick:
            return key


t0=time.clock()

def ABC(dataset, classes, centroids, a_limit, max_iter):
    n_data, n_attr = np.shape(dataset) 
    n_bees = len(centroids) 
    var_min = 0 
    var_max = 1 

    keys = [key for key in centroids] 
    C = centroids.copy()
    for key in C:
        C[key] = 0

    
    costs = centroids.copy()
    for cl in costs:
        costs[cl] = costFunction(dataset, classes, cl, centroids[cl])

    best_solution = 99999999
    best_solutions = np.zeros(max_iter)

    for it in range(max_iter):
        
        for cl in centroids:
            _keys = keys.copy() 
            index = _keys.index(cl)
            del _keys[index]
            k = random.choice(_keys) 
            phi = np.random.uniform(-1, 1, n_attr)

            new_solution = centroids[cl] + phi * (centroids[cl] - centroids[k])

            
            new_solution_cost = costFunction(dataset, classes, cl, new_solution)

            
            if new_solution_cost <= costs[cl]:
                centroids[cl] = new_solution
                costs[cl] = new_solution_cost
                C[cl] = 0
            else: 
                
                C[cl] += 1

        F = fitnessFunction(costs) 
        f_sum_arr = [F[key] for key in F]
        f_sum = np.sum(f_sum_arr)
        P = {} 
        for key in F:
            P[key] = F[key]/f_sum

        
        for cl_o in centroids:
            selected_key = rouletteWheelFunction(P)

            _keys = keys.copy() 
            index = _keys.index(selected_key)
            del _keys[index]
            k = random.choice(_keys) 

            
            phi = np.random.uniform(-1, 1, n_attr)

            
            new_solution = centroids[selected_key] + phi * (centroids[selected_key] - centroids[k])

            
            new_solution_cost = costFunction(dataset, classes, selected_key, new_solution)

            
            if new_solution_cost <= costs[selected_key]:
                centroids[selected_key] = new_solution
                costs[selected_key] = new_solution_cost
                C[selected_key] = 0
            else: 
                
                C[selected_key] += 1

        
        for cl_s in centroids:
            if C[cl_s] > a_limit:
                random_solution = np.random.uniform(0, 1, n_attr)
                random_solution_cost = costFunction(dataset, classes, cl_s, random_solution)

                centroids[cl_s] = new_solution
                costs[cl_s] = random_solution_cost
                C[cl_s] = 0

        
        best_solution = 1
        for cl in centroids:
            if costs[cl] < best_solution:
                best_solution = costs[cl]

        best_solutions[it] = best_solution
        
           

        print('Iteration: {it}; Best cost: {best_solution}'.format(it = "%03d" % it, best_solution = best_solution))

    return best_solutions, centroids

def nearestCentroidClassifier(data, centroids):
    distances = centroids.copy()
    for key in centroids:
        distances[key] = euclidianDistance(data, centroids[key])

    distances_sorted = sorted(distances.items(), key = operator.itemgetter(1))
    nearest_class, nearest_centroid = distances_sorted[0]

    return nearest_class

def getSets(dataset, classes):
    size = len(dataset)

    trainning_set = dataset[:round(size * 0.75), :]
    trainning_set_classes = classes[:round(size * 0.75)]

    test_set = dataset[round(size * 0.75):, :]
    test_set_classes = classes[round(size * 0.75):]

    return trainning_set, test_set, trainning_set_classes, test_set_classes



databases = [{ 'filename': 'flight.data', 'has_id': True, 'class_position': 'last' }]

for database in databases:
    d, c = readDatabase(database['filename'], database['has_id'], database['class_position'])
    trainning_set, test_set, trainning_set_classes, test_set_classes = getSets(d.copy(), c.copy())

    stats, centroids = determineCentroids(trainning_set, trainning_set_classes)


    
    limits = [1000]
    for limit in limits:
        best_soltions, new_centroids = ABC(trainning_set, trainning_set_classes, centroids.copy(), a_limit = limit, max_iter = 825)
        print('\n\n## DATABASE: {filename}'.format(filename = database['filename']))

    
        count = 0
        
        for i, val in enumerate(test_set):
            cl = nearestCentroidClassifier(test_set[i], centroids)
            if cl != str(test_set_classes[i]):
                
                count += 1
        
        count = 0
        
        for i, val in enumerate(test_set):
            cl = nearestCentroidClassifier(test_set[i], new_centroids)
            if cl != str(test_set_classes[i]):
                count += 1
        
t1=time.clock() - t0
print("time elapsed is : ",(t1-t0))

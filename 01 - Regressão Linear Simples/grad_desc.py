from numpy import *
import math

RSS_points = []
gradient_size = []
iterations = []

verbose = True

def compute_error_for_given_points(b, m, points):
    total_error = 0
    
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]

        total_error += (y - (m * x + b)) ** 2
    
    return total_error / float(len(points))

def step_gradient(b_current, m_current, points, learning_rate):
    # Gradient Descent
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]

        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) *  x * (y - ((m_current * x) + b_current))

    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)

    RSS = compute_error_for_given_points(new_b, new_m, points)
    RSS_points.append(RSS)
    
    if(verbose):
        print("For values of:\nb = {0}\nm = {1}\nThe RSS is {2}"
            .format(new_b, new_m, RSS))

    return [new_b, new_m, b_gradient, m_gradient]

def compute_gradient_size(b, m):
    return math.sqrt(b ** 2 + m ** 2)

def step_gradient_norm(b_current, m_current, points, learning_rate, gradient_threshold):
    # Gradient Descent
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))

    
    while True:
        for i in range(0, len(points)):
            x = points[i, 0]
            y = points[i, 1]

            b_gradient += -(2/N) * (y - (b_current + (m_current * x)))
            m_gradient += -(2/N) * ((y - (b_current + (m_current * x))) * x)

        new_b = b_current - (learning_rate * b_gradient)
        new_m = m_current - (learning_rate * m_gradient)

        grad_sz = compute_gradient_size(b_gradient, m_gradient)
        gradient_size.append(grad_sz)
        print(linalg.norm([b_gradient, m_gradient]))
        #print(new_b, new_m)
        #print(b_gradient, m_gradient)
        #print(grad_sz, gradient_threshold)
        if(grad_sz < gradient_threshold):
            break
        
    RSS = compute_error_for_given_points(new_b, new_m, points)
    RSS_points.append(RSS)
    
    if(verbose):
        print("For values of:\nb = {0}\nm = {1}\nThe RSS is {2}"
            .format(new_b, new_m, RSS))

    return [new_b, new_m]

def gradient_desc_norm(points, starting_b, starting_m, learning_rate, gradient_threshold):
    b = starting_b
    m = starting_m
    
    gradient_norms = []
        
    iteration = 1
        
    while True:
        b, m, b_gradient, m_gradient = step_gradient(b, m, array(points), learning_rate)
        gradient_norm = linalg.norm([b_gradient, m_gradient])
        
        gradient_norms.append(gradient_norm)
        
        if gradient_norm < gradient_threshold:
            break
        
        iteration += 1
            
    return [b, m, gradient_norms, iteration]

def normal_equations(points):
    mean_x, mean_y = points.mean(0)
    
    # m
    num = 0
    den = 0
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        
        num += (x - mean_x)*(y - mean_y)
        den += (x - mean_x)**2
        
    new_m = num/den
    
    # b = mean(y) - m * mean(x)
    new_b = mean_y - new_m * mean_x
    
    return [new_b, new_m, 0, 0]
    

def gradient_desc_runner(points, starting_b, starting_m, learning_rate, num_iterations, stopping_criteria, gradient_threshold):
    b = starting_b
    m = starting_m
    
    i = 0

    if(stopping_criteria == 1):
        while True:
            b, m = step_gradient(b, m, array(points), learning_rate)

            grad_sz = compute_gradient_size(b, m)
            gradient_size.append(grad_sz)
            iterations.append(i)

            if(verbose):
                print("Gradient size: {0}".format(grad_sz))

            if(stopping_criteria == 1):
                if(i >= num_iterations):
                    break
            elif(stopping_criteria == 2):
                if(grad_sz >= gradient_threshold):
                    break

            i += 1
    elif(stopping_criteria == 2):
        results = gradient_desc_norm(array(points), b, m, learning_rate, gradient_threshold)
    elif(stopping_criteria == 3):
        results = normal_equations(array(points))

    return results




# Stopping Criteria takes one of two values
#   1 - Number of iterations
#   2 - Gradient Size (Euclidian Norm) - A gradient threshold must be provided
def run(learning_rate = False, num_iterations = False, messages = True, stopping_criteria = 1, gradient_threshold = 0):
    if(not messages):
        global verbose
        verbose = False
    
    global RSS_points, gradient_size, iterations
    RSS_points = []
    gradient_size = []
    iterations = []
    
    points = genfromtxt('income.csv', delimiter=',')

    # Hyperparameters
    if not learning_rate:
        learning_rate = 0.0001

    # y = b + mx
    initial_b = 0
    initial_m = 0

    if not num_iterations:
        num_iterations = 1000

    if(verbose):
        print("Starting gradient descent at b = {0}, m = {1}, error = {2}"
            .format(initial_b, initial_m, compute_error_for_given_points(initial_b, initial_m, points)))
        print("Running...")

    b, m, gradient_norms, n_iterations = gradient_desc_runner(points, initial_b, initial_m, learning_rate, num_iterations, stopping_criteria, gradient_threshold)

    if(verbose):
        print("After {0} iterations b = {1}, m = {2}, error = {3}"
            .format(num_iterations, b, m, compute_error_for_given_points(b, m, points)))

    return {
        "b": b,
        "m": m,
        "RSS": RSS_points,
        "gradient_norms": gradient_norms,
        "gradient_size": gradient_size,
        "n_iterations": n_iterations}
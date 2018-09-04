from numpy import *
import math

RSS_points = []
gradient_size = []

verbose = True

def compute_error_for_given_points(b, m, points):
    total_error = 0
    
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]

        total_error += (y - (m * x + b)) ** 2
    
    return total_error / float(len(points))

def step_gradient(points, b_current, m_current, learning_rate):
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

def gradient_desc_iter(points, b, m, learning_rate, num_iterations):
    for i in range(num_iterations):
        results = step_gradient(array(points), b, m, learning_rate)
        b = results[0]
        m = results[1]

    return results[:2]

def gradient_desc_norm(points, b, m, learning_rate, gradient_threshold):
    gradient_norms = []
    iteration = 1
        
    while True:
        b, m, b_gradient, m_gradient = step_gradient(array(points), b, m, learning_rate)
        gradient_norm = linalg.norm([b_gradient, m_gradient])
        
        gradient_norms.append(gradient_norm)
        
        if gradient_norm < gradient_threshold:
            break
        
        iteration += 1
            
    return [b, m, gradient_norms, iteration]

def normal_equations(points):
    mean_x, mean_y = points.mean(0)
    
    # m = ...
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
    
def gradient_desc_runner(points, b, m, learning_rate, num_iterations, running_mode, gradient_threshold):
    if(running_mode == 1):
        print("Running Gradient Descent using the number of iterations as a stoping condition")
        return gradient_desc_iter(array(points), b, m, learning_rate, num_iterations)
    elif(running_mode == 2):
        print("Running Gradient Descent using the norm's size as a stoping condition")
        return gradient_desc_norm(array(points), b, m, learning_rate, gradient_threshold)
    elif(running_mode == 3):
        print("Running Gradient Descent using the normal equations")
        return normal_equations(array(points))

# Running Mode takes one of two values
#   1 - Number of iterations
#   2 - Gradient Size (Euclidian Norm) - A gradient threshold must be provided
#   3 - Normal Equations
def run(learning_rate = False, num_iterations = False, messages = True, gradient_threshold = 1, running_mode = 1):
    if(not messages):
        global verbose
        verbose = False
    
    global RSS_points, gradient_size, iterations
    RSS_points = []
    gradient_norms = []
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

    results = gradient_desc_runner(points, initial_b, initial_m, learning_rate, num_iterations, running_mode, gradient_threshold)

    b = results[0]
    m = results[1]
    if(len(results) > 2):
        gradient_norms = results[2]
    if(len(results) > 3):
        num_iterations = results[3]

    if(verbose):
        print("After {0} iterations b = {1}, m = {2}, error = {3}"
            .format(num_iterations, b, m, compute_error_for_given_points(b, m, points)))

    return {
        "b": b,
        "m": m,
        "RSS": RSS_points,
        "gradient_norms": gradient_norms,
        "n_iterations": num_iterations
        }
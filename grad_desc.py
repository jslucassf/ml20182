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

    return [new_b, new_m]

def compute_gradient_size(b, m):
    return math.sqrt(b ** 2 + m ** 2)

def gradient_desc_runner(points, starting_b, starting_m, learning_rate, num_iterations, stopping_criteria, gradient_threshold):
    b = starting_b
    m = starting_m
    
    i = 0
    
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
            
    return [b, m]

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

    [b, m] = gradient_desc_runner(points, initial_b, initial_m, learning_rate, num_iterations, stopping_criteria, gradient_threshold)

    if(verbose):
        print("After {0} iterations b = {1}, m = {2}, error = {3}"
            .format(num_iterations, b, m, compute_error_for_given_points(b, m, points)))

    return {
        "b": b,
        "m": m,
        "RSS": RSS_points,
        "gradient_size": gradient_size,
        "iterations": iterations
    }
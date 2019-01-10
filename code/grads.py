import numpy as np

def example(x):
    return np.sum(x**2)


def example_grad(x):
    return 2*x

def foo(x):
    result = 1
    λ = 4 # this is here to make sure you're using Python 3
    for x_i in x:
        result += x_i**λ
    return result

def foo_grad(x):
    # The foo function evaluates x^4 so the gradient would be the derivative of that, which is 4x^3
    return 4*(x**3)

def bar(x):
    return np.prod(x)

def bar_grad(x):
    """
     The bar function evaluates the multiplication of all items in vector x, (x1 * x2 * x3)
     so each element of the vector will be counted as a different variable and we will do partial derivatives 
     with respect of each element in the vector, so if you have a vector with legth 3, the derivative with
     respect to x1 will be (1 * x2 * x3) because de derivative of x1 is equal to 1 and the other elements remain
     as constants, we follow this process with each element so the whole gradient will be something like:
     [(1 * x2 * x3), (x1 * 1 * x3), (x1 * x2 * 1)]
    """
    grad = np.array([])
    for x_i in x:
        res = 1
        for x_j in x:
            if(x_j != x_i):
                res *= x_j
        grad = np.append(grad,res)
    return grad

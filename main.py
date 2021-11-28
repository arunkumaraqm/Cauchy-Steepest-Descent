"""
Cauchy's steepest descent or gradient descent for minimization
Inputs: 
1. Objective function f such that f is convex and grad f is Lipschitz continuous
2. Starting Point X1
Outputs:
X such that X is a local minimum
"""
from sympy import diff, Eq, solveset, symbols, sympify
from math import isclose


MAX_ITERATIONS = 20
x, y, z, a = symbols('x y z a') 


def gradient_descent(function, initial_pt, precision=4):
    isequals = lambda a, b: isclose(a, b, abs_tol=10 ** (-precision))
    def listequals(alist, blist):
        res = True
        for a, b in zip(alist, blist):
            res = res and isequals(a, b)
        return res

    try: xi, yi, zi = initial_pt
    except: xi, yi, zi = *initial_pt, 0

    grad = []    # To store the gradient
    s = []       # To store the direction of the slope (negative of grad)
    temp = []    # x_(i+1) in terms of alpha
    initial = [] # To store the initial answer to check with the final
    result = []  # To store the final answer
    n = 0 

    df_dx = diff(function, x)
    df_dy = diff(function, y)
    df_dz = diff(function, z)


    while True:
        n += 1
        x_previous, y_previous, z_previous = xi, yi, zi

        for df_dc in [df_dx, df_dy, df_dz]:
            grad.append(df_dc.subs([(x, xi), (y, yi), (z, zi)])) #To push the gradient wrt x in the list
            s.append(-grad[-1]) #To push the x-coord of S-direction in the list

        for idx, icoord in enumerate([xi, yi, zi]):
            temp.append(icoord + (a * s[idx])) # To push the x-coord of xi equation in the list

        alphaEq = function.subs([(x, temp[0]), (y, temp[1]), (z, temp[2])]) 

        # Substituting and differentiating the eq with unknown alpha and equating with 0
        tempEq = Eq(diff(alphaEq, a), 0) 
        alpha = list(solveset(tempEq, a))[0] # Solving for alpha 

        xi = round(xi + (alpha * s[0]), precision) #Calculating the new x-coords
        yi = round(yi + (alpha * s[1]), precision) #Calculating the new y-coords
        zi = round(zi + (alpha * s[2]), precision) #Calculating the new z-coords

        for df_dc in [df_dx, df_dy, df_dz]:
            result.append(df_dc.subs([(x, xi), (y, yi), (z, zi)]))
        
        print(result)
        print("#{4} gradient, a, [x, y, z]: {5}, {3}, [{0}, {1}, {2}]".format(xi, yi, zi, round(alpha, precision), n, grad))
        
        if listequals(result, [0, 0, 0]):
            print("Terminated because result is zero") #To let the user know which condition caused it to stop
            break

        elif listequals([x_previous, y_previous, z_previous], [xi, yi, zi]):
            print("Terminated because new x's were equal to old x's") 
            break   

        elif n > MAX_ITERATIONS: 
            break   

        grad.clear(); result.clear(); s.clear(); temp.clear()

def main():
    equation = input("Enter the expression: ") 
    equation = sympify(equation) 
    print("Interpreting as", equation)

    initial_pt = input("Enter initial coordinates: ")
    initial_pt = tuple(map(float, initial_pt.split()))

    try:    gradient_descent(equation, initial_pt)
    except IndexError: print('ERROR. Please check whether your objective function satisfies input conditions.')

if __name__ == '__main__':
    main()

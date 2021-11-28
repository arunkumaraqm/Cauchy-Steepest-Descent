from sympy import *

MAX_ITERATIONS = 20
x, y, z, a = symbols('x y z a') 


def gradient_descent(function, initial_pt, precision=6):

    try: xi, yi, zi = initial_pt
    except: xi, yi, zi = *initial_pt, 0

    grad = []    # To store the gradient
    s = []       # To store the direction of the slope
    temp = []   
    initial = [] # To store the initial answer to check with the final
    result = []  # To store the final answer
    n = 0 


    while True:
        n += 1
        x_previous, y_previous, z_previous = xi, yi, zi

        for coord in [x, y, z]:
         grad.append(diff(function, coord).subs([(x,xi),(y,yi),(z,zi)])) #To push the gradient wrt x in the list
         s.append(-grad[-1]) #To push the x-coord of S-direction in the list

        for idx, icoord in enumerate([xi, yi, zi]):
            temp.append(icoord + (a * s[idx])) #To push the x-coord of xi equation in the list

        alphaEq = function.subs([(x,temp[0]),(y,temp[1]),(z,temp[2])]) #For brevity
        # Substituting and differentiating the eq with unknown alpha and equating with 0
        tempEq = Eq(diff(alphaEq, a), 0) 
        alpha = list(solveset(tempEq, a)) #Solving for alpha 
        alpha[0]
        s[0]

        xi = round(xi + (alpha[0] * s[0]), precision) #Calculating the new x-coords
        yi = round(yi + (alpha[0] * s[1]), precision) #Calculating the new y-coords
        zi = round(zi + (alpha[0] * s[2]), precision) #Calculating the new z-coords

        for coord in [x, y, z]:
            result.append(diff(function, coord).subs([(x, xi), (y, yi), (z, zi)]))
        
        print(result)
        print("#{4} gradient, a, [x, y, z]: {5}, {3}, [{0}, {1}, {2}]".format(xi,yi,zi,round(alpha[0],precision),n,grad))
        
        if result == [0, 0, 0]:
            print("Terminated because result is zero") #To let the user know which condition caused it to stop
            break

        elif [x_previous, y_previous, z_previous] == [xi, yi, zi]:
            print("Terminated because new x's were equal to old x's") 
            break   

        elif(n > MAX_ITERATIONS): break   

        grad.clear(); result.clear(); s.clear(); temp.clear()

def main():
    equation = input("Enter the expression: ") 
    equation = sympify(equation) 
    print("Interpreting as", equation)

    initial_pt = input("Enter initial coordinates: ")
    initial_pt = tuple(map(float, initial_pt.split()))

    gradient_descent(equation, initial_pt)

if __name__ == '__main__':
    main()

import numpy as np
from time import clock
import matplotlib.pyplot as plt

a = -1.0
b = 5.0
max = 30000

def integrate(a,b):
    return ((1/9)*np.sin(3*b) - (1/3)*b*np.cos(3*b) - ((1/9)*np.sin(3*a) - (1/3)*a*np.cos(3*a)))

actual = integrate(-1,5)


def f(x):
    return x*np.sin(3*x)

def findN(est, tol, func1):
    a = est
    b = a/10
    while (a - b) >= 2:
        c = (a + b) / 2
        c = int(np.ceil(c))
        fc = func1(-1, 5, f, c)
        if abs(actual - fc) < tol:
            a = c          
        else:
            b = c
    while abs(actual - fc) > tol:
        c += 1
        fc = func1(-1, 5, f, c)
    return c

def int_trap(a,b,f,n):
    h = (b-a)/float(n)
    I = f(a) + f(b)
    for k in range(1,n):
        I += 2*f(a+k*h)
    I *= h/2.0
    return I

def int_mid(a,b,f,n):
    h = (b-a)/float(n)
    m = a+h/2.0
    I = 0
    for k in range(0,n):
        I += f(m+k*h)
    I *= h
    return I

def int_simp(a,b,f,n):
    h = (b-a)/float(2*n)
    I = f(a) + f(b)
    for k in range(1,n):
        I += 4*f(a+(2*k-1)*h) + 2*f(a+2*k*h)
    I += 4*f(a+(2*n-1)*h)
    I *= h/3.0
    return I

def get_time_trap(n):
    iters = int(np.ceil(max / n))
    if n > 2861:
        iters = 10
    start = clock()
    for i in range(iters):
        int_trap(a,b,f,n)
    end = clock()
    elapsed = (end - start) / iters
    return elapsed

def get_time_mid(n):
    iters = int(np.ceil(max / n))
    if n > 2025:
        iters = 10
    start = clock()
    for i in range(iters):
        int_mid(a,b,f,n)
    end = clock()
    elapsed = (end - start) / iters
    return elapsed

def get_time_simp(n):
    iters = int(np.ceil(max / n))
    if n > 61:
        iters = 10
    start = clock()
    for i in range(iters):
        int_simp(a,b,f,n)
    end = clock()
    elapsed = (end - start) / iters
    return elapsed

def leastSquares(x, y):
    xySum = 0
    xSum = 0
    ySum = 0
    x2Sum = 0
    n = len(x)
    for i in range(0, n):
        xySum += x[i]*y[i]
        xSum += x[i]
        ySum += y[i]
        x2Sum += x[i]**2.0
    b1 = (xySum - ((xSum*ySum)/float(n)))
    b1 /= x2Sum - (xSum)**2/float(n)
    b0 = (ySum - b1*xSum) / float(n)
    return b0, b1

def leastSquares2(x,y,N):
    n = len(x)
    B = np.matrix(y)
    B = B.T
    A = np.matrix(np.ones((n,N+1)))
    for i in range(0,n):
        A[i,1] = x[i]
        A[i,2] = np.multiply(x[i],x[i])
    c = np.linalg.solve((A[0:n,0:N+1].T*A[0:n,0:N+1]), (A[0:n,0:N+1].T * B[0:n,0]))
    print(A*c)
    return A*c

start = clock()

midErr = [abs(actual - int_mid(a,b,f,21)), abs(actual - int_mid(a,b,f,64)), abs(actual - int_mid(a,b,f,202)), abs(actual - int_mid(a,b,f,639)), abs(actual - int_mid(a,b,f,2018)), abs(actual - int_mid(a,b,f,findN(88331, 0.5*10**-7, int_mid))), abs(actual - int_mid(a,b,f,findN(883316, 0.5*10**-9, int_mid))), abs(actual - int_mid(a,b,f,findN(8833154, 0.5*10**-11, int_mid)))] 
midTimes = [get_time_mid(21), get_time_mid(64), get_time_mid(202), get_time_mid(639), get_time_mid(2018), get_time_mid(findN(88331, 0.5*10**-7, int_mid)), get_time_mid(findN(883316, 0.5*10**-9, int_mid)), get_time_mid(findN(8833154, 0.5*10**-11, int_mid))]

trapErr = [abs(actual - int_trap(a,b,f,29)), abs(actual - int_trap(a,b,f,91)), abs(actual - int_trap(a,b,f,286)), abs(actual - int_trap(a,b,f,903)), abs(actual - int_trap(a,b,f,2854)), abs(actual - int_trap(a,b,f,findN(124920, 0.5*10**-7, int_trap))),abs(actual - int_trap(a,b,f,findN(1249197, 0.5*10**-9, int_trap))), abs(actual - int_trap(a,b,f,findN(12491966, 0.5*10**-11, int_trap)))]
trapTimes = [get_time_trap(29), get_time_trap(91), get_time_trap(286), get_time_trap(903), get_time_trap(2854), get_time_trap(findN(124920, 0.5*10**-7, int_trap)), get_time_trap(findN(1249197, 0.5*10**-9, int_trap)), get_time_trap(findN(12491966, 0.5*10**-11, int_trap))]

simpErr = [abs(actual - int_simp(a,b,f,6)), abs(actual - int_simp(a,b,f,11)), abs(actual - int_simp(a,b,f,18)), abs(actual - int_simp(a,b,f,32)), abs(actual - int_simp(a,b,f,56)), abs(actual - int_simp(a,b,f,findN(915, 0.5*10**-7, int_simp))), abs(actual - int_simp(a,b,f,findN(2892, 0.5*10**-9, int_simp))), abs(actual - int_simp(a,b,f,findN(9144, 0.5*10**-11, int_simp))), abs(actual - int_simp(a,b,f,findN(91438, 0.5*10**-15, int_simp)))]
simpTimes = [get_time_simp(6), get_time_simp(11), get_time_simp(18), get_time_simp(32), get_time_simp(56), get_time_simp(findN(915, 0.5*10**-7, int_simp)), get_time_simp(findN(2892, 0.5*10**-9, int_simp)), get_time_simp(findN(9144, 0.5*10**-11, int_simp)), get_time_simp(findN(91438, 0.5*10**-15, int_simp))]

tol = np.array([0.5*10**-1, 0.5*10**-2, 0.5*10**-3, 0.5*10**-4, 0.5*10**-5, 0.5*10**-7, 0.5*10**-9, 0.5*10**-11, 0.5*10**-14])

midY_int = np.array([leastSquares(np.log10(tol[0:4]), np.log10(midTimes[0:4]))[0]])
midSlope = np.array([leastSquares(np.log10(tol[0:4]), np.log10(midTimes[0:4]))[1]])

trapY_int = np.array([leastSquares(np.log10(tol[0:4]), np.log10(trapTimes[0:4]))[0]])
trapSlope = np.array([leastSquares(np.log10(tol[0:4]), np.log10(trapTimes[0:4]))[1]])

simpFit = leastSquares2(np.log10(tol), np.log10(simpTimes), 2)
simpFit = np.array(simpFit)

midFit = np.array(np.zeros(len(tol)))
trapFit = np.array(np.zeros(len(tol)))

for i in range(0, len(tol)):
    midFit[i] = midY_int + midSlope*np.log10(tol[i])
    trapFit[i] = trapY_int + trapSlope*np.log10(tol[i])

fig, ax = plt.subplots()
for i in range(0,5):
    mid, = ax.plot(tol[i], midTimes[i],'o', color = 'red')
    trap, = ax.plot(tol[i], trapTimes[i],'o', color = 'green')
    simp, = ax.plot(tol[i], simpTimes[i],'o', color = 'blue')
for i in range(5,len(tol) - 1):
    midT, = ax.plot(tol[i], midTimes[i],'^', color = 'red')
    trapT, = ax.plot(tol[i], trapTimes[i],'^', color = 'green')
    simpT, = ax.plot(tol[i], simpTimes[i],'o', color = 'blue')
    
simpT, = ax.plot(tol[8], simpTimes[8],'o', color = 'blue')

ax.set_title('Tolerance vs. Computational Cost')
ax.set_xscale('log', base = 10)
ax.set_yscale('log', base = 10)
ax.set_xlabel('Tolerance')
ax.set_ylabel('Computation Cost')

mFit, = plt.plot(tol, 10**midFit, color = 'red')
tFit, = plt.plot(tol, 10**trapFit, color = 'green')
sFit, = plt.plot(tol, 10**simpFit, color = 'blue')
plt.legend([mid, trap, simp, midT, trapT], ['Midpoint (Best Fit)', 'Trapezoidal (Best Fit)', 'Simpsons (Best Fit)', 'Midpoint', 'Trapezoid'], loc = 'upper right')
end = clock()
print(end - start)
plt.show()

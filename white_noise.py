from __future__ import division, print_function
import sympy
from sympy import (init_printing, Matrix, MatMul,
                   integrate, symbols)

init_printing()
dt, phi, u, x,y,z = symbols('dt \Phi_s, \mu, x,y,z')

F_k = Matrix([[1, 0,dt, 0,0],
              [0,1,0,dt,0 ],
              [0,0,1,0,0],
              [0,0,0,1,0],
              [0,0,0,0,1]])
Q_c = Matrix([[0, 0,0,0,0],
              [0, 0,0,0,0],
              [0, 0,1,0,0],
              [0,0,0,1,0],
              [0, 0, 0, 0,0]])*phi

Q = integrate(F_k * Q_c * F_k.T, (dt, 0, dt))

# factor phi out of the matrix to make it more readable
Q = Q / phi
MatMul(Q, phi)
print(Q)

print(Q*Q.T)
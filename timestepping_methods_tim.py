#timestepping options for averaged_sw_explicit.py
#rk2/rk4/heuns/ssprk3/leapfrog

def rk2(U, USlow_in, USlow_out, DU, V, W,
        expt, ensemble, cheby, cheby2, SlowSolver, wt, dt):
    #Average the nonlinearity
    cheby.apply(U, USlow_in, expt)
    SlowSolver.solve()
    cheby.apply(USlow_out, DU, -expt)
    DU *= wt
    ensemble.allreduce(DU, V)
    #Step forward V
    V.assign(U + 0.5*V)

    #transform forwards to U^{n+1/2}
    cheby2.apply(V, DU, dt/2)

    #Average the nonlinearity
    cheby.apply(DU, USlow_in, expt)
    SlowSolver.solve()
    cheby.apply(USlow_out, DU, -expt)
    DU *= wt
    ensemble.allreduce(DU, V)
    #Advance U
    cheby2.apply(U, DU, dt/2)
    V.assign(DU + V)

    #transform forwards to next timestep
    cheby2.apply(V, U, dt/2)


def rk4(U, Q, USlow_in, USlow_out, CSlow_out, U_shift, DU, Q1, Q2, Q3, QS, f0, f1, f2, f3, fa, fb, fc, C0, C1, C2, C3, P0, P1, P2, P3, M0, M1, M2, M3, C_in, y_out, y_in, P_out, P_in, M_out,
                expt, ensemble, cheby, cheby2, SlowSolver, SlowSolverC, y_Solver, P_Solver, M_Solver, wt, dt):
    
    ###############################################
    #Step 1 of 4:
    
    #Shift the q solution to e^sL q
    cheby.apply(Q, USlow_in, expt)
    U_shift.assign(USlow_in)
    
    #We already know the mean correction, C0, at the start of the time step
    
    ###
    #Compute P0, which requires solving 2 linear systems:
    C_in = C0 #Pass C_in for the solver
    
    y_Solver.solve()
    y_in.assign(y_out)
    
    P_Solver.solve()
    P0.assign(P_out)
    
    #Compute M0 = L*P0
    P_in.assign(P_out)
    M_Solver.solve()
    M0.assign(M_out)
    
    #Transform to u = e^sL q + P
    #U_shift += P0
    U_shift -= P0
    
    USlow_in.assign(U_shift) 
    
    #Compute the non-linear vector, N(u)
    SlowSolver.solve()
    
    #Form N + M
    #USlow_out += M0
    USlow_out -= M0
    
    #Apply the matrix exponential to form f_s = e^((-t+s)L)(N + M)
    cheby.apply(USlow_out, DU, -expt)
    
    #Apply the weightings to the gradient evaluations and combine them
    DU *= wt
    ensemble.allreduce(DU, f0)
    
    #Step forward to Q1 using the averaged gradient
    Q1.assign(Q + 0.5*f0)
    
    ################################################
    #Step 2 of 4:

    #Shift the q solution
    cheby2.apply(Q1, DU, dt/2)
    cheby.apply(DU, USlow_in, expt)
    U_shift.assign(USlow_in)
    
    #Compute the mean correction:
    SlowSolverC.solve()
    CSlow_out *= wt
    ensemble.allreduce(CSlow_out,C1)
    
    #Compute P1:
    C_in = C1 #Pass C_in for the solver
    
    y_Solver.solve()
    y_in.assign(y_out)
    
    P_Solver.solve()
    P1.assign(P_out)
    
    #Transform to u = e^-tL w + P
    #U_shift += P1
    U_shift -= P1
    
    USlow_in.assign(U_shift)
    
    #Compute the non-linear terms
    SlowSolver.solve()
    
    #Compute M0
    P_in.assign(P_out)
    M_Solver.solve()
    M1.assign(M_out)
    
    #Form N + M, M = LP
    #USlow_out += M1
    USlow_out -= M1
    
    #Apply the matrix exponential to N + C
    cheby.apply(USlow_out, DU, -expt)
    #Apply the weightings to the gradient evaluations and combine them
    DU *= wt
    ensemble.allreduce(DU, f1)
    
    #Compute e^(dt/2)*Q:
    cheby2.apply(Q, QS, dt/2)
    
    #Step forward Q2
    Q2.assign(QS + 0.5*f1)
    
    ################################################
    #Step 3 of 4:
    
    #Shift the q solution
    cheby.apply(Q2, USlow_in, expt)
    U_shift.assign(USlow_in)

    #Compute the mean correction:
    SlowSolverC.solve()
    CSlow_out *= wt
    ensemble.allreduce(CSlow_out,C2)
    
    #Compute P2:
    C_in = C2 #Pass C_in for the solver
    
    y_Solver.solve()
    y_in.assign(y_out)
    
    P_Solver.solve()
    P2.assign(P_out)
    
    #Transform to u = e^-tL w + P
    #U_shift += P2
    U_shift -= P2
    
    USlow_in.assign(U_shift)
    
    #Compute the non-linear terms
    SlowSolver.solve()
    
    #Compute M0
    P_in.assign(P_out)
    M_Solver.solve()
    M2.assign(M_out)
    
    #Form N + M, M = LP
    #USlow_out += M2
    USlow_out -= M2
    
    #Apply the matrix exponential to N + L*L_inv*C
    cheby.apply(USlow_out, DU, -expt)
    #Apply the weightings to the gradient evaluations and combine them
    DU *= wt
    ensemble.allreduce(DU, f2)

    #Step forward Q3
    Q3.assign(QS + f2)
    
    ################################################
    #Step 4 of 4:
    
    #Shift the q solution
    cheby2.apply(Q3, DU, dt/2)
    cheby.apply(DU, USlow_in, expt)
    U_shift.assign(USlow_in)
    
    #Compute the mean correction:
    SlowSolverC.solve()
    CSlow_out *= wt
    ensemble.allreduce(CSlow_out,C3)
    
    #Compute P3:
    C_in = C3 #Pass C_in for the solver
    
    y_Solver.solve()
    y_in.assign(y_out)
    
    P_Solver.solve()
    P3.assign(P_out)
    
    #Transform to u = e^-tL w + P
    #U_shift += P3
    U_shift -= P3
    
    USlow_in.assign(U_shift)
    
    #Compute the non-linear terms
    SlowSolver.solve()
    
    #Compute M0
    P_in.assign(P_out)
    M_Solver.solve()
    M3.assign(M_out)
    
    #Form N + M, M = LP
    #USlow_out += M3
    USlow_out -= M3

    #Apply the matrix exponential to N + L*L_inv*C
    cheby.apply(USlow_out, DU, -expt)
    #Apply the weightings to the gradient evaluations and combine them
    DU *= wt
    ensemble.allreduce(DU, f3)
    
    #Shift some of the gradient evaluations:
    cheby2.apply(f0, fa, dt)
    cheby2.apply(f1, fb, dt/2)
    cheby2.apply(f2, fc, dt/2)
    
    #Compute e^(dt)*Q:
    cheby2.apply(Q, QS, dt)

    #Compute Q at the end of the time step:
    Q.assign(QS + 1/6*(fa + 2*fb + 2*fc + f3))
    
    #Compute the mean correction at this final location:
    cheby.apply(Q, USlow_in, expt)
    SlowSolverC.solve()
    CSlow_out *= wt
    ensemble.allreduce(CSlow_out,C0)
    
    #Compute the solution, U.
    #This is actually only needed at checkpoint times
    C_in = C0 #Pass C_in for the solver
    
    y_Solver.solve()
    y_in.assign(y_out)
    
    P_Solver.solve()
    P0.assign(P_out)
    
    U.assign(Q - P0) 
    
    
def heuns(U, USlow_in, USlow_out, DU, U1, U2, W,
          expt, ensemble, cheby, cheby2, SlowSolver, wt, dt):
    #Average the nonlinearity
    cheby.apply(U, USlow_in, expt)
    SlowSolver.solve()
    cheby.apply(USlow_out, DU, -expt)
    DU *= wt
    ensemble.allreduce(DU, U1)
    #Step forward U1
    U1.assign(U + U1)

    #Average the nonlinearity
    cheby2.apply(U1, DU, dt)
    cheby.apply(DU, USlow_in, expt)
    SlowSolver.solve()
    cheby.apply(USlow_out, DU, -expt)
    DU *= wt
    ensemble.allreduce(DU, U2)
    #Step forward U2
    cheby2.apply(U, DU, dt)
    U2.assign(DU + U2)

    #transform forwards to next timestep
    cheby2.apply(U1, U, dt)
    U.assign(0.5*U + 0.5*U2)


def ssprk3(U, USlow_in, USlow_out, DU, U1, U2, W,
           expt, ensemble, cheby, cheby2, SlowSolver, wt, dt):
    #Average the nonlinearity
    cheby.apply(U, USlow_in, expt)
    SlowSolver.solve()
    cheby.apply(USlow_out, DU, -expt)
    DU *= wt
    ensemble.allreduce(DU, U1)
    #Step forward U1
    DU.assign(U + U1)
    cheby2.apply(DU, U1, dt)

    #Average the nonlinearity
    cheby.apply(U1, USlow_in, expt)
    SlowSolver.solve()
    cheby.apply(USlow_out, DU, -expt)
    DU *= wt
    ensemble.allreduce(DU, U2)
    #Step forward U2
    DU.assign(U1 + U2)
    cheby2.apply(DU, U2, -dt/2)
    cheby2.apply(U, U1, dt/2)
    U2.assign(0.75*U1 + 0.25*U2)

    #Average the nonlinearity
    cheby.apply(U2, USlow_in, expt)
    SlowSolver.solve()
    cheby.apply(USlow_out, DU, -expt)
    DU *= wt
    ensemble.allreduce(DU, U1)
    #Advance U
    DU.assign(U2 + U1)
    cheby2.apply(DU, U2, dt/2)
    cheby2.apply(U, U1, dt)
    U.assign(1/3*U1 + 2/3*U2)


def leapfrog(U, USlow_in, USlow_out, U_old, U_new, DU, U1, U2, V, W,
             expt, ensemble, cheby, cheby2, SlowSolver, wt, dt, asselin):
    #Average the nonlinearity
    cheby.apply(U, USlow_in, expt)
    SlowSolver.solve()
    cheby.apply(USlow_out, DU, -expt)
    DU *= wt
    ensemble.allreduce(DU, V)
    #Step forward V
    cheby2.apply(U_old, DU, dt)
    V.assign(DU + 2*V)
    cheby2.apply(V, U_new, dt)
    #Asselin filter
    cheby2.apply(U_old, U1, dt)
    cheby2.apply(U_new, U2, -dt)
    V.assign((U1+U2)*0.5 - U)
    U_old.assign(U + asselin*V)
    #Advance U
    U.assign(U_new)

from cheby_exp import *
from firedrake import *
from firedrake.petsc import PETSc
from math import pi
from math import ceil
from timestepping_methods_tim import *
from latlon import *
import numpy as np
import argparse

#get command arguments
parser = argparse.ArgumentParser(description='Williamson 5 testcase for averaged propagator.')
parser.add_argument('--ref_level', type=int, default=4, help='Refinement level of icosahedral grid. Default 4.')
parser.add_argument('--space_parallel', type=int, default=4, help='Default 4.')
parser.add_argument('--tmax', type=float, default=360, help='Final time in hours. Default 24x15=360.')
parser.add_argument('--dumpt', type=float, default=24, help='Dump time in hours. Default 24.')
parser.add_argument('--checkt', type=float, default=6, help='Create checkpointing file every checkt hours. Default 6.')
parser.add_argument('--dt', type=float, default=0.5, help='Timestep for the averaged model in hours. Default 0.5.')
parser.add_argument('--rho', type=float, default=1, help='Averaging window width as a multiple of dt. Default 1.')
parser.add_argument('--Mbar', action='store_true', dest='get_Mbar', help='Compute suitable Mbar, print it and exit.')
parser.add_argument('--ppp', type=float, default=4, help='Points per time-period for averaging.')
parser.add_argument('--timestepping', type=str, default='rk4', choices=['rk2', 'rk4', 'heuns', 'ssprk3', 'leapfrog'], help='Choose a time steeping method. Default rk4.')
parser.add_argument('--asselin', type=float, default=0.3, help='Asselin Filter coefficient for leapfrog. Default 0.3.')
parser.add_argument('--filename', type=str, default='explicit_tim')
parser.add_argument('--pickup', action='store_true', help='Pickup the result from the checkpoint.')
parser.add_argument('--pickup_from', type=str, default='explicit')
args = parser.parse_known_args()
args = args[0]
timestepping = args.timestepping
asselin = args.asselin
ref_level = args.ref_level
filename = args.filename
space_parallel = args.space_parallel
print(args)

#ensemble communicator
ensemble = Ensemble(COMM_WORLD, space_parallel)

#parameters
R0 = 6371220.
R = Constant(R0)
H = Constant(5960.)
Omega = Constant(7.292e-5)  # rotation rate
g = Constant(9.8)  # Gravitational constant
mesh = IcosahedralSphereMesh(radius=R0,
                             refinement_level=ref_level, degree=3,
                             comm = ensemble.comm)
x = SpatialCoordinate(mesh)
global_normal = as_vector([x[0], x[1], x[2]])
mesh.init_cell_orientations(global_normal)
outward_normals = CellNormal(mesh)
perp = lambda u: cross(outward_normals, u)

#Define the function spaces for the problem
V1 = FunctionSpace(mesh, "BDM", 2)
V2 = FunctionSpace(mesh, "DG", 1)
W = MixedFunctionSpace((V1, V2))#

f_expr = 2 * Omega * x[2] / R
Vf = FunctionSpace(mesh, "CG", 3)
f = Function(Vf).interpolate(f_expr)    # Coriolis frequency
u_0 = 20.0  # maximum amplitude of the zonal wind [m/s]
u_max = Constant(u_0)
u_expr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
eta_expr = -((R*Omega*u_max+u_max*u_max/2.0)*(x[2]*x[2]/(R*R)))/g

#topography (D = H + eta - b)
rl = pi/9.0
lambda_x = atan_2(x[1]/R0, x[0]/R0)
lambda_c = -pi/2.0
phi_x = asin(x[2]/R0)
phi_c = pi/6.0
minarg = Min(pow(rl, 2), pow(phi_x - phi_c, 2) + pow(lambda_x - lambda_c, 2))
bexpr = 2000.0*(1 - sqrt(minarg)/rl)
b = Function(V2, name="Topography")
b.interpolate(bexpr)

#checking cheby parameters based on ref_level
eigs = [0.003465, 0.007274, 0.014955] #maximum frequency
min_time_period = 2*pi/eigs[ref_level-3]
hours = args.dt
dt = 60*60*hours
rho = args.rho #averaging window is rho*dt
L = eigs[ref_level-3]*dt*rho
ppp = args.ppp #points per (minimum) time period
               #rho*dt/min_time_period = number of min_time_periods that fit in rho*dt
               # we want at least ppp times this number of sample points
Mbar = ceil(ppp*rho*dt*eigs[ref_level-3]/2/pi)
if args.get_Mbar:
    print("Mbar="+str(Mbar))
    import sys; sys.exit()

svals = np.arange(0.5, Mbar)/Mbar #tvals goes from -rho*dt/2 to rho*dt/2
weights = np.exp(-1.0/svals/(1.0-svals))
weights = weights/np.sum(weights)
print(weights)
svals -= 0.5

#parameters for timestepping
t = 0.
tmax = 60.*60.*args.tmax
dumpt = args.dumpt*60.*60.
checkt = args.checkt*60.*60.
tdump = 0.
tcheck = 0.

#print out settings
print = PETSc.Sys.Print
assert Mbar*space_parallel==COMM_WORLD.size, str(Mbar)+' '+str(COMM_WORLD.size)
print('averaging window', rho*dt, 'sample width', rho*dt/Mbar)
print('Mbar', Mbar, 'samples per min time period', min_time_period/(rho*dt/Mbar))
print(args)

#pickup the result
if args.pickup:
    chkfile = DumbCheckpoint(args.pickup_from, mode=FILE_READ, comm=ensemble.comm)
    un = Function(V1, name="Velocity")
    etan = Function(V2, name="Elevation")
    chkfile.load(un, name="Velocity")
    chkfile.load(etan, name="Elevation")
    t = chkfile.read_attribute("/", "time")
    tdump = chkfile.read_attribute("/", "tdump")
    tcheck = chkfile.read_attribute("/", "tcheck")
    chkfile.close()
else:
    un = Function(V1, name="Velocity").project(u_expr)
    etan = Function(V2, name="Elevation").interpolate(eta_expr)

#calculate norms for debug
uini = Function(V1, name="Velocity0").project(u_expr)
etaini = Function(V2, name="Elevation0").interpolate(eta_expr)
etanorm = errornorm(etan, etaini)/norm(etaini)
unorm = errornorm(un, uini, norm_type="Hdiv")/norm(uini, norm_type="Hdiv")
print('etanorm', etanorm, 'unorm', unorm)

##############################################################################
# Set up the exponential operator
##############################################################################
operator_in = Function(W)
u_in, eta_in = split(operator_in)
u, eta = TrialFunctions(W)
v, phi = TestFunctions(W)

F = (
    - inner(f*perp(u_in),v)*dx
    +g*eta_in*div(v)*dx
    - H*div(u_in)*phi*dx
)

a = inner(v,u)*dx + phi*eta*dx

operator_out = Function(W)

params = {
    'ksp_type': 'preonly',
    'pc_type': 'fieldsplit',
    'fieldsplit_0_ksp_type':'cg',
    'fieldsplit_0_pc_type':'bjacobi',
    'fieldsplit_0_sub_pc_type':'ilu',
    'fieldsplit_1_ksp_type':'preonly',
    'fieldsplit_1_pc_type':'bjacobi',
    'fieldsplit_1_sub_pc_type':'ilu'
}

Prob = LinearVariationalProblem(a, F, operator_out)
OperatorSolver = LinearVariationalSolver(Prob, solver_parameters=params)

ncheb = 10000

cheby = cheby_exp(OperatorSolver, operator_in, operator_out,
                  ncheb, tol=1.0e-8, L=L)

cheby2 = cheby_exp(OperatorSolver, operator_in, operator_out,
                   ncheb, tol=1.0e-8, L=L)

#############################################################################
#Construct an analogous solver to compute P = L_inv*C
#which is used in the new transformation, as u = e^-tL w + P
#I will implement Colin's two step approach to find a regularised version of the 
#psuedo-inverse.
#We want to solve (alpha*I - L)(alpha*I + L)P = L^* C
#we do this in two steps:
#1. (alpha*I - L)y = L^* C, let A1 = (alpha*I - L)
#2. (alpha*I + L)P = y, let A2 = (alpha*I + L)

#Firstly, define the regularisation level, alpha:
alpha = Constant(5e-8)

#Problem 1:
C_in = Function(W)
C_u, C_eta = split(C_in)
y_u, y_eta = TrialFunctions(W)
test_a, test_b = TestFunctions(W)

left_y = (
    inner(f*perp(y_u),test_a)*dx
    - g*y_eta*div(test_a)*dx
    + H*div(y_u)*test_b*dx
    + alpha*(inner(y_u,test_a)+y_eta*test_b)*dx
)

right_y = (
    inner(f*perp(C_u),test_a)*dx
    - g*C_eta*div(test_a)*dx
    + H*div(C_u)*test_b*dx
)

y_out = Function(W)

Prob_y = LinearVariationalProblem(left_y, right_y, y_out)
y_Solver = LinearVariationalSolver(Prob_y, solver_parameters=params)

#Problem 2:
y_in = Function(W)
y_u, y_eta = split(y_in)
P_u, P_eta = TrialFunctions(W)

left_P = (
    -inner(f*perp(P_u),test_a)*dx
    + g*P_eta*div(test_a)*dx
    - H*div(P_u)*test_b*dx
    + alpha*(inner(P_u,test_a)+P_eta*test_b)*dx
)

right_P = inner(y_u,test_a)*dx + y_eta*test_b*dx

P_out = Function(W)

Prob_P = LinearVariationalProblem(left_P, right_P, P_out)
P_Solver = LinearVariationalSolver(Prob_P)#, solver_parameters=params)

####################################################################################
#Now, set up a solver to compute LP = L*L_inv*C = M
P_in = Function(W)
Pc_u, Pc_eta = split(P_in)
M_u, M_eta = TrialFunctions(W)

left_M = inner(test_a,M_u)*dx + test_b*M_eta*dx

right_M = (
    - inner(f*perp(Pc_u),test_a)*dx
    + g*Pc_eta*div(test_a)*dx
    - H*div(Pc_u)*test_b*dx
)

M_out = Function(W)

#Note to self, need to put trial function stuff as first argument to the solver
Prob_M = LinearVariationalProblem(left_M, right_M, M_out)
M_solver = LinearVariationalSolver(Prob_M)

##############################################################################
# Set up solvers for the slow part
##############################################################################
USlow_in = Function(W) #value at previous timestep
USlow_out = Function(W) #value at RK stage
u0, eta0 = split(USlow_in)

CSlow_out = Function(W)

#RHS for Forward Euler step
gradperp = lambda f: perp(grad(f))
n = FacetNormal(mesh)
Upwind = 0.5 * (sign(dot(u0, n)) + 1)
both = lambda u: 2*avg(u)
K = 0.5*inner(u0, u0)
uup = 0.5 * (dot(u0, n) + abs(dot(u0, n)))

dT = Constant(dt)

vector_invariant = True
if vector_invariant:
    L = (
        dT*inner(perp(grad(inner(v, perp(u0)))), u0)*dx
        - dT*inner(both(perp(n)*inner(v, perp(u0))),
                   both(Upwind*u0))*dS
        + dT*div(v)*K*dx
        + dT*inner(grad(phi), u0*(eta0-b))*dx
        - dT*jump(phi)*(uup('+')*(eta0('+')-b('+'))
                        - uup('-')*(eta0('-') - b('-')))*dS
        )
else:
    L = (
        dT*inner(div(outer(u0, v)), u0)*dx
        - dT*inner(both(inner(n, u0)*v), both(Upwind*u0))*dS
        + dT*inner(grad(phi), u0*(eta0-b))*dx
        - dT*jump(phi)*(uup('+')*(eta0('+')-b('+'))
                        - uup('-')*(eta0('-') - b('-')))*dS
        )

#Define slow solvers, which compute the non-linear terms
SlowProb = LinearVariationalProblem(a, L, USlow_out)
SlowSolver = LinearVariationalSolver(SlowProb,
                                     solver_parameters = params)
                                     
#Construct a second slow solver for the mean correction to make it easier for me to follow
#I could just use the first problem once am more familiar with the code
SlowProbC = LinearVariationalProblem(a, L, CSlow_out)
SlowSolverC = LinearVariationalSolver(SlowProbC,
                                     solver_parameters = params)

##############################################################################
# Time loop
##############################################################################
#Define functions for standard and modulation variables
U = Function(W)
DU = Function(W)
U1 = Function(W)
U2 = Function(W)
U3 = Function(W)
X1 = Function(W)
X2 = Function(W)
X3 = Function(W)
V = Function(W)
U_shift = Function(W)

#Functions to store the 'working variable', q, in:
Q = Function(W)
Q1 = Function(W)
Q2 = Function(W)
Q3 = Function(W)
QS = Function(W)

#Create new functions for the mean corrections:
C0 = Function(W, name="Mean Correction")
C1 = Function(W)
C2 = Function(W)
C3 = Function(W)

#Create functions to store P = L_inv*C:
P0 = Function(W, name='L_inv_C')
P1 = Function(W)
P2 = Function(W)
P3 = Function(W)

#Create functions to store M = L*L_inv*C:
M0 = Function(W)
M1 = Function(W)
M2 = Function(W)
M3 = Function(W)

#Functions to store the gradient evaluations:
f0 = Function(W)
f1 = Function(W)
f2 = Function(W)
f3 = Function(W)
fa = Function(W)
fb = Function(W)
fc = Function(W)


#set weights
rank = ensemble.ensemble_comm.rank
expt = rho*dt*svals[rank]
wt = weights[rank]
print(wt, "weight", expt)
print("svals", svals)

#Compute the initial condition for the solution variable
#This usually requires iteration - initially just set q_0 == u_0 
#Or q_0 = u_0 + P(u_0)
# THIS IS SOMETHING I CAN LOOK INTO THE SENSITIVITY OF.
U_u, U_eta = U.split()
U_u.assign(un)
U_eta.assign(etan)

#Apply the initial condition to the working variable, Q:
Q_u, Q_eta = Q.split()
Q_u.assign(un)
Q_eta.assign(etan)

#Compute the mean evaluation of the initial state.
#This will enable defining an initial condition for the new modulation variable.
cheby.apply(Q, USlow_in, expt)
SlowSolver.solve()
USlow_out *= wt
ensemble.allreduce(USlow_out,C0)

C0_u, C0_eta = C0.split()

#Check that C is non-zero. It should be something different to 1.
etanorm = errornorm(C0_eta, etan)/norm(etan)
unorm = errornorm(C0_u, un, norm_type="Hdiv")/norm(un, norm_type="Hdiv")
print('C_eta val', etanorm, 'C_u val', unorm, 'hmm',norm(C0_eta))

#Compute initial P:
C_in = C0 #Pass C_in for the solver

Cin_u, Cin_eta = C_in.split()

#Check that C is non-zero. It should be something different to 1.
etanorm = errornorm(Cin_eta, etan)/norm(etan)
unorm = errornorm(Cin_u, un, norm_type="Hdiv")/norm(un, norm_type="Hdiv")
print('C_eta val', etanorm, 'C_u val', unorm, 'hmm',norm(Cin_eta))

y_Solver.solve()
y_in.assign(y_out)

#Check that y_out is non-zero. It should be different to 1.
y0_u,y0_eta = y_out.split()
etanorm = errornorm(y0_eta, etan)/norm(etan)
unorm = errornorm(y_u, un, norm_type="Hdiv")/norm(un, norm_type="Hdiv")
print('y_eta_diff', etanorm, 'y_u_diff', unorm, 'y_eta norm',norm(y0_eta), 'y_u norm',norm(y0_u))

#This means that y is currently set to zero....

P_Solver.solve()
#P0.assign(P_out)
P0.assign(y_in)

#Try with adjusted initial conditions:
Q.assign(U + P0)

P0_u,P0_eta = P0.split()

#Print the difference between these initial conditions:
etanorm = errornorm(Q_eta, etaini)/norm(etaini)
unorm = errornorm(Q_u, uini, norm_type="Hdiv")/norm(uini, norm_type="Hdiv")
print('IC eta diff', etanorm, 'IC u diff', unorm)


if rank==0:
    #setup PV solver
    PV = Function(Vf, name="PotentialVorticity")
    gamma = TestFunction(Vf)
    q = TrialFunction(Vf)
    D = etan + H - b
    a = q*gamma*D*dx
    L = (- inner(perp(grad(gamma)), un))*dx + gamma*f*dx
    PVproblem = LinearVariationalProblem(a, L, PV)
    PVsolver = LinearVariationalSolver(PVproblem, solver_parameters={"ksp_type": "cg"})
    PVsolver.solve()

    #write out initial fields
    mesh_ll = get_latlon_mesh(mesh)
    file_sw = File(filename+'.pvd', comm=ensemble.comm, mode="a")
    field_un = Function(
        functionspaceimpl.WithGeometry.create(un.function_space(), mesh_ll),
        val=un.topological)
    field_etan = Function(
        functionspaceimpl.WithGeometry.create(etan.function_space(), mesh_ll),
        val=etan.topological)
    field_PV = Function(
        functionspaceimpl.WithGeometry.create(PV.function_space(), mesh_ll),
        val=PV.topological)
    field_b = Function(
        functionspaceimpl.WithGeometry.create(b.function_space(), mesh_ll),
        val=b.topological)
    #Now, record the mean corrections as well
    field_C_u = Function(
        functionspaceimpl.WithGeometry.create(C0_u.function_space(), mesh_ll),
        val=C0_u.topological)
    field_C_eta = Function(
        functionspaceimpl.WithGeometry.create(C0_eta.function_space(), mesh_ll),
        val=C0_eta.topological)
    #Record Ps for interest also:
    field_P_u = Function(
        functionspaceimpl.WithGeometry.create(C0_u.function_space(), mesh_ll),
        val=P0_u.topological)
    field_P_eta = Function(
        functionspaceimpl.WithGeometry.create(C0_eta.function_space(), mesh_ll),
        val=P0_eta.topological)
    if not args.pickup:
        file_sw.write(field_un, field_etan, field_PV, field_b, field_C_u, field_C_eta, field_P_u, field_P_eta)

#start time loop
print('tmax', tmax, 'dt', dt)
while t < tmax - 0.5*dt:
    print(t)
    t += dt
    tdump += dt
    tcheck += dt

    if t < dt*1.5 and timestepping == 'leapfrog':
        U_old = Function(W)
        U_new = Function(W)
        U_old.assign(U)
        rk2(U, USlow_in, USlow_out, DU, V, W,
            expt, ensemble, cheby, cheby2, SlowSolver, wt, dt)
    else:
        if timestepping == 'leapfrog':
            leapfrog(U, USlow_in, USlow_out, U_old, U_new, DU, U1, U2, V, W,
                     expt, ensemble, cheby, cheby2, SlowSolver, wt, dt, asselin)
        elif timestepping == 'ssprk3':
            ssprk3(U, USlow_in, USlow_out, DU, U1, U2, W,
                   expt, ensemble, cheby, cheby2, SlowSolver, wt, dt)
        elif timestepping == 'rk2':
            rk2(U, USlow_in, USlow_out, DU, V, W,
                expt, ensemble, cheby, cheby2, SlowSolver, wt, dt)
        elif timestepping == 'rk4':
            rk4(U, Q, USlow_in, USlow_out, CSlow_out, U_shift, DU, Q1, Q2, Q3, QS, f0, f1, f2, f3, fa, fb, fc, C0, C1, C2, C3, P0, P1, P2, P3, M0, M1, M2, M3,
             C_in, y_out, y_in, P_out, P_in, M_out,
                expt, ensemble, cheby, cheby2, SlowSolver, SlowSolverC, y_Solver, P_Solver, M_solver, wt, dt)
        elif timestepping == 'heuns':
            heuns(U, USlow_in, USlow_out, DU, U1, U2, W,
                  expt, ensemble, cheby, cheby2, SlowSolver, wt, dt)


    if rank == 0:
        un.assign(U_u)
        etan.assign(U_eta)

        #dumping results
        if tdump > dumpt - dt*0.5:
            #dump averaged results
            PVsolver.solve()
            file_sw.write(field_un, field_etan, field_PV, field_b, field_C_u, field_C_eta, field_P_u, field_P_eta)
            #update dumpt
            print("dumped at t =", t)
            tdump -= dumpt

        #create checkpointing file every tcheck hours
        if tcheck > checkt - dt*0.5:
            thours = int(t/3600)
            chk = DumbCheckpoint(filename+"_"+str(thours)+"h", mode=FILE_CREATE, comm = ensemble.comm)
            tcheck -= checkt
            chk.store(un)
            chk.store(etan)
            chk.write_attribute("/", "time", t)
            chk.write_attribute("/", "tdump", tdump)
            chk.write_attribute("/", "tcheck", tcheck)
            chk.close()
            print("checkpointed at t =", t)

            #calculate norms for debug
            etanorm = errornorm(etan, etaini)/norm(etaini)
            unorm = errornorm(un, uini, norm_type="Hdiv")/norm(uini, norm_type="Hdiv")
            print('etanorm', etanorm, 'unorm', unorm)
            
            #Calculate difference between U and Q for debug
            etanorm = errornorm(Q_eta, etan)/norm(etan)
            unorm = errornorm(Q_u, un, norm_type="Hdiv")/norm(un, norm_type="Hdiv")
            print('etanorm', etanorm, 'unorm', unorm)

        #Record the mean evaluations, C, as defined time points.
        

print("Completed calculation at t = ", t/3600, "hours")

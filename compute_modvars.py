#Compute the modulation variables from a saved set of simulation results
#Note, that this will become more costly the longer the simulation is!
from cheby_exp import *
from firedrake import *
from firedrake.petsc import PETSc
from math import pi
from math import ceil
from timestepping_methods import *
from latlon import *
import numpy as np
import argparse
import time

#saved_filename = 'results/aved/aved_reflev3_rho1_dt0.5_TT360_checkt6/explicit'
saved_filename = 'results/standard_reflev3_dt45_TT360_checkt6/standard'
filename = 'standard_dt45_checkt6'

# parameters
ref_level = 3
space_parallel = 1

#Specify the total time and checkpoint times
TT = 360
checkt = 6

#Give the time-stepping and averaging parameters:
dt = 1/80
rho=1

HOURS = np.arange(checkt,TT + checkt,checkt)
print(HOURS)

print(COMM_WORLD.size, COMM_WORLD.rank)
PETSc.Sys.Print("should happen once", COMM_WORLD.size)
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
V1 = FunctionSpace(mesh, "BDM", 2)
V2 = FunctionSpace(mesh, "DG", 1)
W = MixedFunctionSpace((V1, V2))
f_expr = 2 * Omega * x[2] / R
Vf = FunctionSpace(mesh, "CG", 3)
f = Function(Vf).interpolate(f_expr)    # Coriolis frequency
u_0 = 20.0  # maximum amplitude of the zonal wind [m/s]
u_max = Constant(u_0)
u_expr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
eta_expr = -((R*Omega*u_max+u_max*u_max/2.0)*(x[2]*x[2]/(R*R)))/g
h_expr = eta_expr + H

##For standard model:
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

eigs = [0.003465, 0.007274, 0.014955] #maximum frequency
min_time_period = 2*pi/eigs[ref_level-3]
L = eigs[ref_level-3]*dt*rho

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
#############################################################


U_modvar = Function(W)
un_modvar, etan_modvar = U_modvar.split()

#The initial modulation variable is the same as the standard solution:
un_0 = Function(V1, name="Velocity").project(u_expr)
etan_0 = Function(V2, name="Elevation").interpolate(eta_expr)

un_modvar.assign(un_0)
etan_modvar.assign(etan_0)

#write out initial fields
mesh_ll = get_latlon_mesh(mesh)
file_sw = File(filename+'.pvd', comm=ensemble.comm, mode="a")
field_un_modvar = Function(
        functionspaceimpl.WithGeometry.create(un_modvar.function_space(), mesh_ll),
        val=un_modvar.topological)
field_etan_modvar = Function(
        functionspaceimpl.WithGeometry.create(etan_modvar.function_space(), mesh_ll),
        val=etan_modvar.topological)
file_sw.write(field_un_modvar, field_etan_modvar)

V_temp = Function(W)
u_cur, eta_cur = V_temp.split()

h_cur = Function(V2)

for hour in HOURS:
    t = int(hour)
    print("Time is ", t)

    #read data from averaged scripts:
    #chkfile0 = DumbCheckpoint(saved_filename+"_"+str(t)+"h", mode=FILE_READ)
    #chkfile0.load(u_cur, name="Velocity")
    #chkfile0.load(eta_cur, name="Elevation")
    #chkfile0.close()
    
    #if reading from standard run, then the depth field is solved instead.
    chkfile0 = DumbCheckpoint(saved_filename+"_"+str(t)+"h", mode=FILE_READ)
    chkfile0.load(u_cur, name="Velocity")
    chkfile0.load(h_cur, name="Depth")
    eta_cur.assign(h_cur + b - H)
    chkfile0.close()
    
    #number of chebys to apply
    t_no = int(t/dt)
    print("Number of exponential operations to apply is, ", t_no)
    
    tic = time.time()
    
    #Compute the modulation variable
    for tc in np.arange(t_no):
      cheby.apply(V_temp,U_modvar,-dt)
      V_temp.assign(U_modvar)

    toc = time.time()
            
    print('Modvar computation took', toc-tic)
    
    #Write out modvar fields
    file_sw.write(field_un_modvar, field_etan_modvar)
    
    
    
    
    
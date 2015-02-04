#!/usr/bin/env python
# encoding: utf-8

r"""
Shallow water flow
==================

Solve the x-split two-dimensional shallow water equations:

.. math::
    h_t + (hu)_x & = 0 \\
    (hu)_t + (hu^2 + \frac{1}{2}gh^2)_x & = fhv - gh B_x \\
    (hv)_t + (huv)_x = -fhu.

Here h is the depth, u is the x-velocity, v is the y-velocity
and g is the gravitational constant.
If we scale x and y on L, the width of the domain, h and B on H
the typical water depth, u and v on c = sqrt(gH) the typical
wave speed and t on T = L/c, we obtain equations of a single
parameter K = fL/c:

.. math::
    h_t + (hu)_x & = 0 \\
    (hu)_t + (hu^2 + \frac{1}{2}h^2)_x & = Khv - h B_x \\
    (hv)_t + (huv)_x = -Khu.

"""

import os
import numpy as np
from clawpack import riemann

Resolution = 100

Solver = ''
Scenario = ''
Bathymetry = ''
T = 10.
K = 10.
U = 0.

if not os.path.isfile('pyclaw.data'):
    print 'Configuration file pyclaw.data not found.'
    exit()

with open('pyclaw.data') as config:
    lines = iter(filter(None, [line.strip() for line in config]))
    Solver = next(lines).upper()
    Scenario = next(lines).upper()
    Bathymetry = next(lines).upper()
    Resolution = int(next(lines))
    T = float(next(lines))
    K = float(next(lines))
    if Scenario == 'STEADY_FLOW':
        U = float(next(lines))

# Reserve variables for bathymetry
# TODO: Can we store this somewhere in the Clawpack state?
B = np.zeros(Resolution)
Bx = np.zeros(Resolution)
B_ghost = np.zeros(Resolution+4)
Bx_ghost = np.zeros(Resolution+4)

def qinit(state,x_min,x_max):
    xc = state.grid.x.centers

    if Scenario == 'STILL_LAKE':
        state.q[0,:] = 0 * xc + 1.0 - B
        # x-momentum
        state.q[1,:] = 0 * xc
        # y-momentum
        state.q[2,:] = 0 * xc

    elif Scenario == 'WAVE':
        h = 1 + 0.2 * (xc > -0.4) * (xc < -0.3)
        state.q[0,:] = h - B
        state.q[1,:] = 0 * xc
        state.q[2,:] = 0 * xc

    elif Scenario == 'ROSSBY':
        x0 = 0.

        # Edge state
        hl = 1.
        ul = 0.
        vl = 0.

        # Central state
        hr = 3.
        ur = 0.
        vr = 0.
        # Water depth
        state.q[0,:] = hl + (hr-hl) * (xc > -0.2) * (xc < 0.2)
        state.q[0,:] -= B # Adjust for bathymetry
        # x-momentum
        state.q[1,:] = hl*ul + (hr*ur-hl*ul) * (xc > -0.2) * (xc < 0.2)
        # y-momentum
        state.q[2,:] = hl*vl + (hr*vr-hl*vl) * (xc > -0.2) * (xc < 0.2)

    elif Scenario == 'GEOSTROPHIC':
        h = 1.0 + 0.5*np.exp(-128*xc*xc) - B
        hx = np.gradient(h, 1./Resolution)
        state.q[0,:] = h
        state.q[1,:] = 0 * xc
        state.q[2,:] = h*(hx + Bx) / K

    elif Scenario == 'STEADY_FLOW':
        h = 1.0 - B
        u = U

        state.q[0,:] = h
        state.q[1,:] = h*u
        state.q[2,:] = 0 * xc


def init_topo(state,x_min,x_max,dx):
    xc = state.grid.c_centers_with_ghost(2)[0]

    global B, Bx, B_ghost, Bx_ghost
    if Bathymetry == 'FLAT':
        B_ghost = 0*xc
    elif Bathymetry == 'SLOPE':
        B_ghost = 0.4 + 0.8*xc
    elif Bathymetry == 'GAUSSIAN':
        B_ghost = 0.5*np.exp(-128*xc*xc)
    elif Bathymetry == 'COSINE':
        B_ghost = 0.5*np.cos(np.pi*xc*4)**2 * (xc > -0.125) * (xc < 0.125)
    elif Bathymetry == 'PARABOLIC_HUMP':
        B_ghost = (0.5 - 32.0*xc*xc)*(xc > -0.125)*(xc < 0.125)
    elif Bathymetry == 'PARABOLIC_BOWL':
        B_ghost = 2.0 * xc*xc
    elif Bathymetry == 'CLIFF':
        B_ghost = (np.tanh(100*xc)+1)/4
    Bx_ghost = np.gradient(B_ghost, dx)
    B = B_ghost[2:-2]
    Bx = Bx_ghost[2:-2]

def step_source(solver,state,dt):
    """
    Source term due to a rotating frame and variable bathymetry.
    Integrated using a 2-stage, 2nd-order Runge-Kutta method.
    This is a Clawpack-style source term routine, which approximates
    the integral of the source terms over a step.
    Note that q[0,:] = h is unaffected by the source term.
    """
    dt2 = dt/2.

    q = state.q

    h    = q[0,:]
    hu   = q[1,:]
    hv   = q[2,:]

    qstar = np.empty(q.shape)

    X = state.c_centers

    v_balance = U * K

    qstar[1,:] = q[1,:] + dt2 * (hv * K - h * Bx)
    qstar[2,:] = q[2,:] - dt2 * (hu * K - h * v_balance)

    hu   = qstar[1,:]
    hv   = qstar[2,:]

    q[1,:] = q[1,:] + dt * (hv * K - h * Bx)
    q[2,:] = q[2,:] - dt * (hu * K - h * v_balance)


def qbc_source_split_lower(state,dim,t,qbc,auxbc,num_ghost):
    for i in range(num_ghost):
        qbc[:,i] = qbc[:,num_ghost]
        # Fix height individually
        qbc[0,i] += B[num_ghost] - B[i]


def qbc_source_split_upper(state,dim,t,qbc,auxbc,num_ghost):
    for i in range(num_ghost):
        qbc[:,-1-i] = qbc[:,-1-num_ghost]
        # Fix height individually
        qbc[0,-1-i] += B[-1-num_ghost] - B[-1-i]

def auxbc_bathymetry_lower(state,dim,t,qbc,auxbc,num_ghost):
    auxbc[0,:num_ghost] = Bx_ghost[:num_ghost]

def auxbc_bathymetry_upper(state,dim,t,qbc,auxbc,num_ghost):
    auxbc[0,-num_ghost:] = Bx_ghost[-num_ghost:]

def setaux(num_ghost,mx,xlower,dxc,maux,aux):
    #    aux[0,i]  = bathymetry gradient

    if "_aux" not in setaux.__dict__:
        setaux._aux = np.empty(aux.shape)

        setaux._aux[0,:] = Bx_ghost

    aux[:,:] = np.copy(setaux._aux)

def setup(use_petsc=False,kernel_language='Fortran',outdir='./_output',solver_type='classic'):
    from clawpack import pyclaw

    if Solver == "UNBALANCED":
        import shallow_roe_with_efix_unbalanced
        riemann_solver = shallow_roe_with_efix_unbalanced
    elif Solver == "LEVEQUE":
        import shallow_roe_with_efix_leveque
        riemann_solver = shallow_roe_with_efix_leveque

    solver = pyclaw.ClawSolver1D(riemann_solver)

    if Solver == "UNBALANCED":
        solver.step_source = step_source

    solver.limiters = pyclaw.limiters.tvd.vanleer
    solver.num_waves = 3
    solver.num_eqn = 3

    solver.kernel_language=kernel_language

    solver.bc_lower[0] = pyclaw.BC.custom
    solver.bc_upper[0] = pyclaw.BC.custom
    solver.user_bc_lower = qbc_source_split_lower
    solver.user_bc_upper = qbc_source_split_upper

    solver.aux_bc_lower[0] = pyclaw.BC.custom
    solver.aux_bc_upper[0] = pyclaw.BC.custom
    solver.user_aux_bc_lower = auxbc_bathymetry_lower
    solver.user_aux_bc_upper = auxbc_bathymetry_upper

    xlower = -0.5
    xupper = 0.5
    mx = Resolution
    num_ghost = 2

    x = pyclaw.Dimension('x',xlower,xupper,mx)
    domain = pyclaw.Domain(x)
    dx = domain.grid.delta[0]

    num_eqn = 3

    num_aux = {
        "UNBALANCED": 0,
        "LEVEQUE": 1,
    }[Solver]
    state = pyclaw.State(domain,num_eqn, num_aux)

    init_topo(state, xlower, xupper, dx)

    if num_aux > 0:
        auxtmp = np.ndarray(shape=(num_aux,mx+2*num_ghost), dtype=float, order='F')
        setaux(num_ghost,mx,xlower,dx,num_aux,auxtmp)
        state.aux[:,:] = auxtmp[:,num_ghost:-num_ghost]

    state.problem_data['grav'] = 1.0
    state.problem_data['k'] = K
    state.problem_data['u'] = U
    state.problem_data['dx'] = dx

    qinit(state, xlower, xupper)

    claw = pyclaw.Controller()
    claw.keep_copy = True
    claw.tfinal = T
    claw.solution = pyclaw.Solution(state,domain)
    claw.solver = solver
    claw.outdir = outdir
    claw.setplot = setplot

    return claw


#--------------------------
def setplot(plotdata):
#--------------------------
    """
    Specify what is to be plotted at each frame.
    Input:  plotdata, an instance of visclaw.data.ClawPlotData.
    Output: a modified version of plotdata.
    """
    plotdata.clearfigures()  # clear any old figures,axes,items data

    # Figure for Surface Level and Potential Vorticity
    plotfigure = plotdata.new_plotfigure(name='Surface level and PV', figno=0)

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = [-0.5,0.5]
    max_h = {
        "STILL_LAKE": 1.2,
        "WAVE": 1.2,
        "ROSSBY": 3.5,
        "GEOSTROPHIC": 1.7,
        "STEADY_FLOW": 1.7,
    }[Scenario]
    plotaxes.ylimits = [0.0,max_h]
    #plotaxes.ylimits = [0.99,1.02]
    plotaxes.title = 'Surface level'
    plotaxes.axescmd = 'subplot(211)'

    # Set up for items on these axes:
    def surface_level(current_data):
        h = current_data.q[0,:]
        return h + B
    plotitem = plotaxes.new_plotitem(plot_type='1d')
    plotitem.plot_var = surface_level
    plotitem.plotstyle = '-'
    plotitem.color = 'b'
    plotitem.kwargs = {'linewidth':3}

    def bathymetry(current_data):
        return B
    plotitem = plotaxes.new_plotitem(plot_type='1d')
    plotitem.plot_var = bathymetry
    plotitem.plotstyle = '-'
    plotitem.color = 'g'
    plotitem.kwargs = {'linewidth':3}

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = [-0.5,0.5]
    plotaxes.title = 'Potential vorticity'
    plotaxes.axescmd = 'subplot(212)'

    # Set up for items on these axes:
    def potential_vorticity(current_data):
        h = current_data.q[0,:]
        hu = current_data.q[1,:]
        hv = current_data.q[2,:]

        vx = np.gradient(hv/h, 1./Resolution)

        pv = (vx + K) / h

        return pv

    plotitem = plotaxes.new_plotitem(plot_type='1d')
    plotitem.plot_var = potential_vorticity
    plotitem.plotstyle = '-'
    plotitem.color = 'b'
    plotitem.kwargs = {'linewidth':3}

    # Figure for momentum components
    plotfigure = plotdata.new_plotfigure(name='Momentum', figno=1)

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.axescmd = 'subplot(211)'
    plotaxes.xlimits = [-0.5,0.5]
    plotaxes.title = 'x-Momentum'

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type='1d')
    plotitem.plot_var = 1
    plotitem.plotstyle = '-'
    plotitem.color = 'b'
    plotitem.kwargs = {'linewidth':3}

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.axescmd = 'subplot(212)'
    plotaxes.xlimits = [-0.5,0.5]
    plotaxes.title = 'y-Momentum'

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type='1d')
    plotitem.plot_var = 2
    plotitem.plotstyle = '-'
    plotitem.color = 'b'
    plotitem.kwargs = {'linewidth':3}

    return plotdata


if __name__=="__main__":
    from clawpack.pyclaw.util import run_app_from_main
    output = run_app_from_main(setup,setplot)

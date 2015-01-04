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
The default initial condition used here models a dam break.
"""

import numpy as np
from clawpack import riemann

f = 1.
# Initialise bathymetry
# TODO: Turn this into a proper function and store it somewhere in the Clawpack state.
B = np.zeros(500)
Bx = np.zeros(500)

def qinit(state,x_min,x_max):
    xc = state.grid.x.centers

    x0 = 0.

    # Left state
    hl = 1.
    ul = 0.
    vl = 0.

    # Right state
    hr = 3.
    ur = 0.
    vr = 0.
    # Water depth
    state.q[0,:] = (hl-hr) * (xc <= -2.0) + hr + (hl-hr) * (xc > 2.0)
    state.q[0,:] -= B # Adjust for bathymetry
    # x-momentum
    state.q[1,:] = (hl*ul-hr*ur) * (xc <= -2.0) + hr*ur + (hl*ul-hr*ur) * (xc > 2.0)
    # y-momentum
    state.q[2,:] = (hl*vl-hr*vr) * (xc <= -2.0) + hr*vr + (hl*vl-hr*vr) * (xc > 2.0)

def init_topo(state,x_min,x_max):
    xc = state.grid.x.centers

    global B, Bx
    B = 0*xc #np.exp(-xc*xc)
    Bx = np.gradient(B, 10./500)

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

    g = state.problem_data['grav']

    qstar = np.empty(q.shape)

    X = state.c_centers

    qstar[1,:] = q[1,:] + dt2 * hv * f - g * h * Bx
    qstar[2,:] = q[2,:] - dt2 * hu * f

    hu   = qstar[1,:]
    hv   = qstar[2,:]

    q[1,:] = q[1,:] + dt * hv * f - g * h * Bx
    q[2,:] = q[2,:] - dt * hu * f

def setup(use_petsc=False,kernel_language='Fortran',outdir='./_output',solver_type='classic'):
    from clawpack import pyclaw
    import shallow_roe_with_efix_split

    solver = pyclaw.ClawSolver1D(shallow_roe_with_efix_split)
    solver.step_source = step_source
    solver.limiters = pyclaw.limiters.tvd.vanleer
    solver.num_waves = 3
    solver.num_eqn = 3

    solver.kernel_language=kernel_language

    solver.bc_lower[0] = pyclaw.BC.extrap
    solver.bc_upper[0] = pyclaw.BC.extrap

    xlower = -5.0
    xupper = 5.0
    mx = 500
    x = pyclaw.Dimension('x',xlower,xupper,mx)
    domain = pyclaw.Domain(x)
    num_eqn = 3
    state = pyclaw.State(domain,num_eqn)

    # Gravitational constant
    state.problem_data['grav'] = 1.0

    init_topo(state, xlower, xupper)
    qinit(state, xlower, xupper)

    claw = pyclaw.Controller()
    claw.keep_copy = True
    claw.tfinal = 100.
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

    # Figure for q[0]
    plotfigure = plotdata.new_plotfigure(name='Surface level', figno=0)

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = [-5.0,5.0]
    plotaxes.ylimits = [-0.5,3.5]
    plotaxes.title = 'Surface level'
    plotaxes.axescmd = 'subplot(311)'

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

    # Figure for q[1]
    #plotfigure = plotdata.new_plotfigure(name='x-Momentum', figno=1)

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.axescmd = 'subplot(312)'
    plotaxes.xlimits = [-5.0,5.0]
    plotaxes.title = 'x-Momentum'

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type='1d')
    plotitem.plot_var = 1
    plotitem.plotstyle = '-'
    plotitem.color = 'b'
    plotitem.kwargs = {'linewidth':3}

    # Figure for q[2]
    #plotfigure = plotdata.new_plotfigure(name='y-Momentum', figno=2)

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.axescmd = 'subplot(313)'
    plotaxes.xlimits = [-5.0,5.0]
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

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

Scenario = ''
Bathymetry = ''
T = 10.
K = 10.

if not os.path.isfile('pyclaw.data'):
    print 'Configuration file pyclaw.data not found.'
    exit()

with open('pyclaw.data') as config:
    lines = iter(filter(None, [line.strip() for line in config]))
    Scenario = next(lines).upper()
    Bathymetry = next(lines).upper()
    Resolution = int(next(lines))
    T = float(next(lines))
    K = float(next(lines))

# Reserve variables for bathymetry
# TODO: Can we store this somewhere in the Clawpack state?
B = np.zeros(Resolution)
Bx = np.zeros(Resolution)

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


def init_topo(state,x_min,x_max):
    xc = state.grid.x.centers

    global B, Bx
    if Bathymetry == 'FLAT':
        B = 0*xc
    elif Bathymetry == 'SLOPE':
        B = 0.4 + 0.8*xc
    elif Bathymetry == 'GAUSSIAN':
        B = 0.5*np.exp(-128*xc*xc)
    elif Bathymetry == 'PARABOLIC_HUMP':
        B = (0.5 - 32.0*xc*xc)*(xc > -0.125)*(xc < 0.125)
    elif Bathymetry == 'PARABOLIC_BOWL':
        B = 2.0 * xc*xc
    elif Bathymetry == 'CLIFF':
        B = (np.tanh(100*xc)+1)/4
    Bx = np.gradient(B, 1./Resolution)

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

    qstar[1,:] = q[1,:] + dt2 * (hv * K - h * Bx)
    qstar[2,:] = q[2,:] - dt2 * hu * K

    hu   = qstar[1,:]
    hv   = qstar[2,:]

    q[1,:] = q[1,:] + dt * (hv * K - h * Bx)
    q[2,:] = q[2,:] - dt * hu * K

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

    xlower = -0.5
    xupper = 0.5
    mx = Resolution
    x = pyclaw.Dimension('x',xlower,xupper,mx)
    domain = pyclaw.Domain(x)
    num_eqn = 3
    state = pyclaw.State(domain,num_eqn)

    init_topo(state, xlower, xupper)
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

    # Figure for q[0]
    plotfigure = plotdata.new_plotfigure(name='Surface level', figno=0)

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = [-0.5,0.5]
    max_h = {
        "STILL_LAKE": 1.2,
        "WAVE": 1.2,
        "ROSSBY": 3.5,
        "GEOSTROPHIC": 1.7
    }[Scenario]
    plotaxes.ylimits = [0.0,max_h]
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
    plotaxes.xlimits = [-0.5,0.5]
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

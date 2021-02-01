import unittest

import matplotlib
import matplotlib.pyplot as plt
# plt.switch_backend('TkAgg')
plt.style.use('ggplot')

from dymos.utils.doc_utils import save_for_docs


# class TestBalancedFieldLengthForDocs(unittest.TestCase):
#
#     # @save_for_docs
#     def test_balanced_field_length_for_docs(self):
import matplotlib.pyplot as plt

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

import dymos as dm
from dymos.examples.balanced_field.ground_roll_ode import GroundRollODE
from dymos.examples.balanced_field.ground_roll_ode_comp import GroundRollODEComp
from dymos.examples.balanced_field.takeoff_ode import TakeoffODE
from dymos.examples.balanced_field.takeoff_climb_ode_comp import TakeoffClimbODEComp

#
# Instantiate the problem and configure the optimization driver
#
p = om.Problem(model=om.Group())

p.driver = om.pyOptSparseDriver()
# p.driver.options['optimizer'] = 'SNOPT'
# p.driver.opt_settings['iSumm'] = 6
# # p.driver.opt_settings['Verify level'] = 3
# # p.driver.opt_settings['Major step limit'] = 0.1
# p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-5
# p.driver.opt_settings['Major optimality tolerance'] = 1.0E-5
# p.driver.opt_settings['Major iterations limit'] = 500
# p.driver.opt_settings['Minor iterations limit'] = 100000
# # p.driver.opt_settings['Linesearch tolerance'] = 0.5


p.driver.options['optimizer'] = 'IPOPT'
p.driver.opt_settings['print_level'] = 5
p.driver.opt_settings['derivative_test'] = 'first-order'
# p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'

p.driver.declare_coloring()



#
# First Phase - Brake Release to V1
# Operating with two engines
# We don't know what V1 is a priori, we're going to use this model to determine it.
#

br_to_v1 = dm.Phase(ode_class=GroundRollODEComp, transcription=dm.Radau(num_segments=3))

#
# Set the options on the optimization variables
#
br_to_v1.set_time_options(fix_initial=True, duration_bounds=(1, 1000), duration_ref=10.0)

br_to_v1.add_state('r', fix_initial=True, lower=0, ref=1000.0, defect_ref=1000.0,
             rate_source='r_dot')

br_to_v1.add_state('v', fix_initial=True, lower=0.0001, ref=100.0, defect_ref=100.0, rate_source='v_dot')

br_to_v1.add_parameter('h', opt=False, units='m', dynamic=True)
br_to_v1.add_parameter('T', val=27000 * 2, opt=False, units='lbf', dynamic=False)
br_to_v1.add_parameter('m', val=174200, opt=False, units='lbm')
br_to_v1.add_parameter('alpha', val=0.0, opt=False, units='deg')
br_to_v1.add_parameter('mu_r', val=0.03, opt=False, units=None, dynamic=False)

br_to_v1.add_timeseries_output('*')

# Second Phase - V1 to Vr
# Operating with one engine
# Vr is taken to be 1.2 * the stall speed (v_stall)
#

v1_to_vr = dm.Phase(ode_class=GroundRollODEComp, transcription=dm.Radau(num_segments=3))

#
# Set the options on the optimization variables
#
v1_to_vr.set_time_options(fix_initial=False, duration_bounds=(1, 1000), duration_ref=1.0)

v1_to_vr.add_state('r', fix_initial=False, lower=0, ref=1000.0, defect_ref=1000.0,
             rate_source='r_dot')

v1_to_vr.add_state('v', fix_initial=False, lower=0.0001, ref=100.0, defect_ref=100.0,
             rate_source='v_dot')

v1_to_vr.add_parameter('h', val=0.0, opt=False, units='m', dynamic=True)
v1_to_vr.add_parameter('T', val=27000, opt=False, units='lbf', dynamic=False)
v1_to_vr.add_parameter('m', val=174200, opt=False, units='lbm')
v1_to_vr.add_parameter('mu_r', val=0.03, opt=False, units=None, dynamic=False)
v1_to_vr.add_parameter('alpha', val=0.0, opt=False, units='deg')
v1_to_vr.add_timeseries_output('*')


rto = dm.Phase(ode_class=GroundRollODEComp, transcription=dm.Radau(num_segments=3))


#
# Set the options on the optimization variables
#
rto.set_time_options(fix_initial=False, duration_bounds=(1, 1000), duration_ref=1.0)

rto.add_state('r', fix_initial=False, lower=0, ref=1000.0, defect_ref=1000.0,
             rate_source='r_dot')

rto.add_state('v', fix_initial=False, lower=0.0001, ref=100.0, defect_ref=100.0,
             rate_source='v_dot')

rto.add_parameter('h', val=0.0, opt=False, units='m', dynamic=True)
rto.add_parameter('T', val=0.0, opt=False, units='N', dynamic=False)
rto.add_parameter('m', val=174200, opt=False, units='lbm')
rto.add_parameter('mu_r', val=0.3, opt=False, units=None, dynamic=False)
rto.add_parameter('alpha', val=0.0, opt=False, units='deg')

rto.add_timeseries_output('*')


# Minimize range at the end of the phase
rto.add_objective('r', loc='final', ref=1000.0) #  ref0=2000.0, ref=3000.0)


# Fourth Phase - Rotate for single engine takeoff
# v_rotate to runway normal force = 0
#

rotate = dm.Phase(ode_class=GroundRollODEComp, transcription=dm.Radau(num_segments=3))

#
# Set the options on the optimization variables
#
rotate.set_time_options(fix_initial=False, duration_bounds=(1.0, 5), duration_ref=1.0)

rotate.add_state('r', fix_initial=False, lower=0, ref=1000.0, defect_ref=1000.0,
             rate_source='r_dot')

rotate.add_state('v', fix_initial=False, lower=0.0001, ref=100.0, defect_ref=100.0,
             rate_source='v_dot')

rotate.add_parameter('h', val=0.0, opt=False, units='m', dynamic=True)
rotate.add_parameter('T', val=27000, opt=False, units='lbf', dynamic=False)
rotate.add_parameter('m', val=174200, opt=False, units='lbm')
rotate.add_parameter('mu_r', val=0.05, opt=False, units=None, dynamic=False)

rotate.add_polynomial_control('alpha', order=1, opt=True, units='deg', lower=0, upper=10, ref=10)

# rotate.add_control('alpha', val=0.0, opt=True, lower=0, upper=10, units='deg')

rotate.add_timeseries_output('*')

# rotate.add_boundary_constraint('alpha', loc='final', equals=10, units='deg')

# rotate.add_path_constraint('alpha_rate', lower=0, upper=3, units='deg/s')

# rotate.add_objective('r', loc='final', ref=10000.0)

# Fourth Phase - Rotate for single engine takeoff
# liftoff until v2 (1.2 * v_stall) at 35 ft
#

climb = dm.Phase(ode_class=TakeoffODE, transcription=dm.Radau(num_segments=10))

#
# Set the options on the optimization variables
#
climb.set_time_options(fix_initial=False, duration_bounds=(1, 100), duration_ref=1.0)

climb.add_state('r', fix_initial=False, lower=0, ref=1000.0, defect_ref=1000.0)
climb.add_state('h', fix_initial=True, lower=0.0, ref=1.0, defect_ref=1.0)
climb.add_state('v', fix_initial=False, lower=0.0001, ref=100.0, defect_ref=100.0)
climb.add_state('gam', fix_initial=True, lower=0.0, ref=0.05, defect_ref=0.05)

climb.add_parameter('T', val=27000, opt=False, units='lbf', dynamic=True)
climb.add_parameter('m', val=174200, opt=False, units='lbm')

climb.add_control('alpha', opt=True, units='deg', lower=-10, upper=15, ref=10, rate_continuity=True, rate_continuity_scaler=1)

climb.add_timeseries_output('*')





# climb.add_objective('time', loc='final', ref=100.0)

p.model.linear_solver = om.DirectSolver()
# p.model.linear_solver = om.PETScKrylov()
# p.model.linear_solver.options['maxiter'] = 4
# # p.model.linear_solver.options['iprint'] = 2
# p.model.linear_solver.precon = om.DirectSolver()


#
# Instantiate the trajectory and add phases
#
traj = dm.Trajectory()

p.model.add_subsystem('traj', traj)

traj.add_phase('br_to_v1', br_to_v1)
traj.add_phase('v1_to_vr', v1_to_vr)
traj.add_phase('rto', rto)
traj.add_phase('rotate', rotate)
traj.add_phase('climb', climb)

#
# Link the phases
#

# Standard "end of first phase to beginning of second phase" linkages
traj.link_phases(['br_to_v1', 'v1_to_vr'], vars=['time', 'r', 'v'])
traj.link_phases(['v1_to_vr', 'rotate'], vars=['time', 'r', 'v', 'alpha'])
traj.link_phases(['rotate', 'climb'], vars=['time', 'r', 'v', 'alpha'])
traj.link_phases(['br_to_v1', 'rto'], vars=['time', 'r', 'v'])

# # Less common "final value of r must be the match at ends of two phases".
traj.add_linkage_constraint(phase_a='rto', var_a='r', loc_a='final',
                            phase_b='climb', var_b='r', loc_b='final',
                            ref=1000)

# Add the boundary constraints here

rto.add_boundary_constraint('v', loc='final', upper=0.001, ref=100, linear=True)

rotate.add_boundary_constraint('F_r', loc='final', equals=0, ref=100000)

climb.add_boundary_constraint('h', loc='final', equals=35, ref=35, units='ft', linear=True)
climb.add_boundary_constraint('gam', loc='final', equals=5, ref=5, units='deg', linear=True)
climb.add_path_constraint('gam', lower=0, upper=5, ref=5, units='deg')
climb.add_boundary_constraint('v_over_v_stall', loc='final', lower=1.2, ref=1.2)

#
# Setup the problem and set the initial guess
#
p.setup(check=True)

p.set_val('traj.br_to_v1.t_initial', 0)
p.set_val('traj.br_to_v1.t_duration', 35)

p.set_val('traj.br_to_v1.states:r', br_to_v1.interpolate(ys=[0, 2500.0], nodes='state_input'))
p.set_val('traj.br_to_v1.states:v', br_to_v1.interpolate(ys=[0.0001, 100.0], nodes='state_input'))

p.set_val('traj.br_to_v1.parameters:alpha', 0, units='deg')
p.set_val('traj.br_to_v1.parameters:h', 0.0)

#

p.set_val('traj.v1_to_vr.t_initial', 35)
p.set_val('traj.v1_to_vr.t_duration', 35)

p.set_val('traj.v1_to_vr.states:r', v1_to_vr.interpolate(ys=[2500, 300.0], nodes='state_input'))
p.set_val('traj.v1_to_vr.states:v', v1_to_vr.interpolate(ys=[100, 110.0], nodes='state_input'))

p.set_val('traj.v1_to_vr.parameters:alpha', 0.0, units='deg')

p.set_val('traj.v1_to_vr.parameters:h', 0.0)

#

p.set_val('traj.rto.t_initial', 35)
p.set_val('traj.rto.t_duration', 1)

p.set_val('traj.rto.states:r', rto.interpolate(ys=[2500, 5000.0], nodes='state_input'))
p.set_val('traj.rto.states:v', rto.interpolate(ys=[110, 0.0001], nodes='state_input'))

p.set_val('traj.rto.parameters:alpha', 0.0, units='deg')
p.set_val('traj.rto.parameters:h', 0.0)
p.set_val('traj.rto.parameters:T', 0.0)
p.set_val('traj.rto.parameters:mu_r', 0.3)

#

p.set_val('traj.rotate.t_initial', 35)
p.set_val('traj.rotate.t_duration', 5)

p.set_val('traj.rotate.states:r', rotate.interpolate(ys=[1750, 1800.0], nodes='state_input'))
p.set_val('traj.rotate.states:v', rotate.interpolate(ys=[80, 85.0], nodes='state_input'))

p.set_val('traj.rotate.polynomial_controls:alpha', 0.0, units='deg')

p.set_val('traj.rotate.parameters:h', 0.0)

#
p.set_val('traj.climb.t_initial', 30)
p.set_val('traj.climb.t_duration', 20)

p.set_val('traj.climb.states:r', climb.interpolate(ys=[5000, 5500.0], nodes='state_input'), units='ft')
p.set_val('traj.climb.states:v', climb.interpolate(ys=[160, 170.0], nodes='state_input'), units='kn')
p.set_val('traj.climb.states:h', climb.interpolate(ys=[0, 35.0], nodes='state_input'), units='ft')
p.set_val('traj.climb.states:gam', climb.interpolate(ys=[0, 5.0], nodes='state_input'), units='deg')

p.set_val('traj.climb.controls:alpha', 5.0, units='deg')
p.set_val('traj.climb.parameters:T', 27000.0, units='lbf')

#
# Solve for the optimal trajectory
#
# p.run_model()
# p.check_partials(compact_print=True)
#
# p.run_model()
#
# from openmdao.utils.sc

dm.run_problem(p, run_driver=True, simulate=False, make_plots=True) #, restart='dymos_simulation.db')
om.n2(p)
rotate.rhs_all.list_outputs(print_arrays=True, prom_name=True)


# rotate.rhs_all.list_outputs(print_arrays=True, prom_name=True)

# import numpy as np
# with np.printoptions(linewidth=1024):
#     for var in ['T', 'calpha', 'D', 'F_r', 'mu_r', 'm']:
#         try:
#             print(var)
#             print(p.get_val(f'traj.rotate.rhs_all.{var}'))
#             print(p.get_val(f'traj.rotate.rhs_all.aero_force_comp.{var}'))
#         except:
#             pass

# print(p.get_val('traj.rotate.rhs_all.aero_forces_comp.F_r'))
# print(p.get_val('traj.rotate.rhs_all.F_r'))

# p.driver.scaling_report()
#
# Get the explicitly simulated solution and plot the results
#
# exp_out = traj.simulate()

#
# fig, axes = plt.subplots(3, 1)
#
# for phase_name in ['br_to_v1', 'v1_to_vr', 'rotate', 'rto']: #, 'climb']:
#
#     axes[0].plot(p.get_val(f'traj.{phase_name}.timeseries.time'),
#                  p.get_val(f'traj.{phase_name}.timeseries.states:r', units='ft'), 'o')
#
#     axes[0].plot(exp_out.get_val(f'traj.{phase_name}.timeseries.time'),
#                  exp_out.get_val(f'traj.{phase_name}.timeseries.states:r', units='ft'), '-')
#
#     axes[1].plot(p.get_val(f'traj.{phase_name}.timeseries.time'),
#                  p.get_val(f'traj.{phase_name}.timeseries.states:v', units='kn'), 'o')
#
#     axes[1].plot(exp_out.get_val(f'traj.{phase_name}.timeseries.time'),
#                  exp_out.get_val(f'traj.{phase_name}.timeseries.states:v', units='kn'), '-')
#
#     try:
#         axes[2].plot(p.get_val(f'traj.{phase_name}.timeseries.time'),
#                      p.get_val(f'traj.{phase_name}.timeseries.F_r', units='N'), 'o')
#
#         axes[2].plot(exp_out.get_val(f'traj.{phase_name}.timeseries.time'),
#                      exp_out.get_val(f'traj.{phase_name}.timeseries.F_r', units='N'), '-')
#     except KeyError:
#         pass
#
# plt.show()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

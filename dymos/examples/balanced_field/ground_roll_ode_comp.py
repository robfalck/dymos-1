import numpy as np

import openmdao.api as om


class GroundRollODEComp(om.ExplicitComponent):
    """
    The ODE System for an aircraft rolling down a runway.

    Computes the acceleration of an aircraft on a runway, per Raymer Eq. 17.97 _[1]
    Computes the position and velocity equations of motion using a modification of the 2D flight
    path parameterization of states per equations 4.42 - 4.46 of _[1].  Flight path angle
    and altitude are static quantities during the ground roll and are not integrated as states.

    References
    ----------
    .. [1] Raymer, Daniel. Aircraft design: a conceptual approach. American Institute of
    Aeronautics and Astronautics, Inc., 2012.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('g', types=(float, int), default=9.80665, desc='gravitational acceleration in m/s**2)')

    def setup(self):
        nn = self.options['num_nodes']

        # Scalar (constant) inputs
        self.add_input('S', val=124.7, desc='aerodynamic reference area', units='m**2')
        self.add_input('CD0', val=0.03, desc='zero-lift drag coefficient', units=None)
        self.add_input('CL0', val=0.5, desc='zero-alpha lift coefficient', units=None)
        self.add_input('CL_max', val=2.0, desc='maximum lift coefficient for linear fit', units=None)
        self.add_input('alpha_max', val=10, desc='angle of attack at CL_max', units='deg')
        self.add_input('h_w', val=1.0, desc='height of the wing above the CG', units='m')
        self.add_input('AR', val=9.45, desc='wing aspect ratio', units=None)
        self.add_input('e', val=0.801, desc='Oswald span efficiency factor', units=None)
        self.add_input('span', val=35.7, desc='Wingspan', units='m')
        self.add_input('h', val=0.0, desc='altitude', units='m')
        self.add_input('T', val=1.0, desc='thrust', units='N')
        self.add_input('mu_r', val=0.03, desc='runway friction coefficient', units=None)

        # Dynamic inputs (can assume a different value at every node)
        self.add_input('m', shape=(nn,), desc='aircraft mass', units='kg')
        self.add_input('v', shape=(nn,), desc='aircraft true airspeed', units='m/s')
        self.add_input('alpha', shape=(nn,), desc='angle of attack', units='rad')

        # Outputs
        self.add_output('CL', shape=(nn,), desc='lift coefficient', units=None)
        self.add_output('q', shape=(nn,), desc='dynamic pressure', units='Pa')
        self.add_output('L', shape=(nn,), desc='lift force', units='N')
        self.add_output('D', shape=(nn,), desc='drag force', units='N')
        self.add_output('F_r', shape=(nn,), desc='runway normal force', units='N')
        self.add_output('v_dot', shape=(nn,), desc='rate of change of speed', units='m/s**2', tags=['state_rate_source:v'])
        self.add_output('r_dot', shape=(nn,), desc='rate of change of range', units='m/s', tags=['state_rate_source:r'])
        self.add_output('W', shape=(nn,), desc='aircraft weight', units='N')
        self.add_output('v_stall', shape=(nn,), desc='stall speed', units='m/s')
        self.add_output('v_over_v_stall', shape=(nn,), desc='stall speed ratio', units=None)

        self.declare_partials(of='*', wrt='*', method='cs')
        self.declare_coloring(wrt='*', method='cs', show_sparsity=True)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        g = 9.80665  # m/s**2  - gravitational acceleration

        # Atmospheric model (constant since the aircraft is always close to the ground
        rho = 1.225  # kg/m**3 - atmospheric density

        # Compute factor k to include ground effect on lift
        v = inputs['v']
        S = inputs['S']
        CD0 = inputs['CD0']
        m = inputs['m']
        T = inputs['T']
        mu_r = inputs['mu_r']
        h = inputs['h']
        h_w = inputs['h_w']
        span = inputs['span']
        AR = inputs['AR']
        CL0 = inputs['CL0']
        alpha = inputs['alpha']
        alpha_max = inputs['alpha_max']
        CL_max = inputs['CL_max']
        e = inputs['e']

        # Compute factor K to account for ground effect
        b = span / 2.0
        K_nom = 1.0 / (np.pi * AR * e)
        K = K_nom * 33 * ((h + h_w) / b)**1.5 / (1.0 + 33 * ((h + h_w) / b)**1.5)

        # Compute the lift coefficient
        CL = outputs['CL'] = CL0 + (alpha / alpha_max) * (CL_max - CL0)

        # Compute the dynamic pressure
        q = outputs['q'] = 0.5 * rho * v ** 2

        # Compute the lift and drag forces
        L = outputs['L'] = q * S * CL
        D = outputs['D'] = q * S * (CD0 + K * CL ** 2)

        # Compute the downward force on the landing gear
        calpha = np.cos(alpha)
        salpha = np.sin(alpha)
        F_r = outputs['F_r'] = m * g - L * calpha - T * salpha

        # Compute the dynamics
        outputs['v_dot'] = (T * calpha - D - F_r * mu_r) / m
        outputs['r_dot'] = v

        # Compute the weight of the aircraft
        W = outputs['W'] = m * g

        # Compute the ratio of the current velocity to the stall velocity
        v_stall = outputs['v_stall'] = np.sqrt(2 * W / rho / S / CL_max)
        outputs['v_over_v_stall'] = v / v_stall


if __name__ == "__main__":

    import openmdao.api as om
    p = om.Problem()
    p.model = GroundRollODEComp(num_nodes=20)

    p.setup(force_alloc_complex=True)
    p.run_model()

    with np.printoptions(linewidth=1024):
        p.check_partials(method='fd', compact_print=True, step=1.0E-7)

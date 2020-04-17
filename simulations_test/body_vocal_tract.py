import sys
import sympy as sy
sys.path.append('/home/victorw/anaconda/envs/dev_pyphs/workspace/lib_pyphs_dev/')
import pyphs

# ------------------------
#      VERSION DU 
#   24 OCTOBRE 2019
# Ajout de l'adimensionnement
# ------------------------

PPTY_PHY_PARAMS = {"positive": True, "real" : True }
PPTY_STATE_VAR  = {"real" : True }

class Body(pyphs.Core):
    """
    """
    pass


class SingleSpringBody(Body):
    def __init__(self , n=1, adim=False): # n is a label (int)
        label = 'spring_k'
        pyphs.Core.__init__(self, label='{0}_{1}'.format(label, n))
        
        # Storage
        x_k = self.symbols('x_k{0}'.format(n), **PPTY_STATE_VAR)
        k   = self.symbols('k_{0}'.format(n), **PPTY_PHY_PARAMS)
        if not adim:
            H = sy.Rational(1,2)*k*x_k**2
        else:
            x_k_scale = self.symbols('x_k{0}_s'.format(n), **PPTY_PHY_PARAMS)
            H = sy.Rational(1, 2)*k*x_k_scale**2*x_k**2
        self.add_storages(x_k, H)

        # Ports
        F_muscle, v_muscle, Fp, vp = self.symbols('F_m{0} \
                                                  v_m{0} F_p{0} \
                                                  v_p{0}'.format(n))
        self.add_ports([vp, v_muscle],[-Fp, -F_muscle])

        # Matrix
        Jxx = [[0]]
        if not adim:
            Jxy = [[-1,1]]
        else:
            Jxy = [[-1/x_k_scale,1/x_k_scale]]

        Jyy = [[0, 0],[0,0]]
        self.set_Jxx(Jxx) 
        self.set_Jxy(Jxy)
        self.set_Jyy(Jyy)


class DissipativeSingleSpringBody(Body):
    def __init__(self , n=1): # n is a label (int)
        label = 'dissSpring_k'
        pyphs.Core.__init__(self, label='{0}_{1}'.format(label, n))
        
        # Storage
        x_k= self.symbols('x_k{0}'.format(n), **PPTY_STATE_VAR)
        k = self.symbols('k_{0}'.format(n), **PPTY_PHY_PARAMS)
        H = sy.Rational(1,2)*k*x_k**2
        self.add_storages(x_k, H)
    
        # Dissipation
        v_k = self.symbols('v_k{0}'.format(n), **PPTY_STATE_VAR)
        r = self.symbols('r_{0}'.format(n), **PPTY_PHY_PARAMS)
        zr = r*v_k
        self.add_dissipations(v_k, zr)     
        
        # Ports
        F_muscle, v_muscle, Fp, vp = self.symbols('F_m{0} \
                                                  v_m{0} F_p{0} \
                                                  v_p{0}'.format(n))
        self.add_ports([vp, v_muscle],[-Fp, -F_muscle])

        # Matrix
        Jxy = [[-1,1]]
        Jwy = [[-1,1]]
        self.set_Jxy(Jxy)
        self.set_Jwy(Jwy)

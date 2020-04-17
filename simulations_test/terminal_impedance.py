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

class RadiationImpedance(pyphs.Core):
    """
    """
    pass


class RadiationImpedance(RadiationImpedance):
    def __init__(self): 
        label = 'Z_ray'
        pyphs.Core.__init__(self, label=label)
        
        # Dissipation
        q  = self.symbols('q')
        Zr = self.symbols('Z_ray', positive=True, real=True)
        zw = q*Zr
        self.add_dissipations(q, zw)

        # Ports 
        q_R, h_R = self.symbols('q_R h_R')
        self.add_ports([q_R,],[h_R,])

        self.set_Jwy([[1]])
 

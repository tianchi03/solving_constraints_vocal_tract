import sys
import os
import sympy as sy
#sys.path.append('/home/victorw/anaconda/envs/dev_pyphs/workspace/lib_pyphs_dev/')
import pyphs

from IPython.core.display import display
from LDL_solve import *
import pickle

PPTY_PHY_PARAMS = {"positive": True, "real" : True }
PPTY_STATE_VAR  = {"real" : True }

fcancel = lambda x: x.cancel()

VERBOSE = 2

# Configuration for naming pickled equations
PICKLE_FOLDER_PATH = '04_equations'
PICKLE_FILE_NAME   =  'equations_vocalTract_N='
    


################
#### VERSION DU 23 avril 2020
#### Édition spéciale du confinement 3
#### Ajout de la sérialisation/décompression si les équations ont déjà été calculées ou pas

def prtVerbose(msg, lvl):
    if VERBOSE >= lvl:
        print(msg)

class VocalTractBase(pyphs.Core):
    """
    """
    pass

class VocalTractLumpedParameter(VocalTractBase):
    """
        Vocal tract: basé sur le modèle contraint sous la forme () (voir doc28). 
        La loi d'état de la compression du fluide est linéarisée. Les variables d'état sont:
               X = [ nu_R, nu_L, Pi_y, rho, Vol ]
               
        Tous les calculs sont effectués dans une classe auxiliaire "VocalTractEquations"
        
        INPUT:
            - label (string): label attaché à l'objet
            - N (int): nombre de tronçons
            - **kwargs: dictionnaire des valeurs des paramètres
            
        OUTPUT:
            - core (pyphs.Core): objet core pyphs
    """    
    
    
    def __init__(self, label="vocal_tract_rho_V", N=2, **kwargs):
        # Instanciation du core
        # NOTE: super() permet d'hériter de la classe
        super().__init__(label=label)
        
        # Equations
        if self.pickle_exists(N):
            equations = self.open_jar(N)
        else:
            prtVerbose("Computing equations pour N = {}...".format(N), 2)
            equations = VocalTractEquations(N=N)
            prtVerbose("Done !", 2)

            prtVerbose("Pickling it...", 2)
            self.pickle_equations(N, equations)
            prtVerbose("Done", 2)
        
        # Composants stockants
        H = equations.Ham_cstrd
        self.add_storages(equations.X, H)
        
        # Observers
        self.add_observer(equations.observers)
        
        # Ports
        self.add_ports(equations.U, equations.Y)

        # Dissipations
        if False:
            for i, w in enumerate(equations.w):
                zw = w * equations.zw[i]
                self.add_dissipations(w, zw)
            self.set_Jwx(equations.Jwx)

        # Interconnexion matrix
        self.set_Jxx(equations.Jxx)
        self.set_Jyx(equations.Jyx)

        
        # Valeurs des subs
        self.subs.update(
            {self.symbols(k, **PPTY_PHY_PARAMS): v for k,v in kwargs.items()}
        )
                
        
        # self.pickle_equations()
        self.build_pickle_numbers()
        
    def pickle_exists(self, N):
        ''' 
        Returns True is the equations are pickled in a file aleady, False
        otherwise 
        '''
        N_list = self.build_pickle_numbers()
        return str(N) in N_list

    def build_pickle_numbers(self):
        ''' 
        Builds a list containing the number of vocal tracts corresponding
        to every existing pickled file.

        OUTPUTS:
            - list (int)
        '''
        filenames = []
        for (_, _, filename) in os.walk(PICKLE_FOLDER_PATH):
            filenames.extend(filename)
            break

        N_list = []
        for fn in filenames:
            for i, ch in enumerate(fn):
                if ch == '=':
                    N_list.append(fn[i+1:])

        return N_list

    def open_jar(self, N):
        ''' Decompress pickled equations for a given number of tube
        INPUT:
            - N (int): number of tubes for the tract model
        OUTPUT:
            - VocalTractEquations (obj): object containing equations
        '''
        PATH = os.path.join(PICKLE_FOLDER_PATH, PICKLE_FILE_NAME + str(N))
        with open(PATH, "rb") as FID:
            obj = pickle.load(FID)
            if VERBOSE >= 2:
                print('Object succesfully loaded')

        return obj
    
    def pickle_equations(self, N, obj):
        ''' Saves class into a file for later loading '''
        PATH = os.path.join(PICKLE_FOLDER_PATH, PICKLE_FILE_NAME + str(N))
        with open(PATH, 'wb') as FID:
            pickle.dump(obj, FID)
            if VERBOSE >= 2:
                print('Object succesfully pickled at "' + PATH + '"')
        
        


class VocalTractEquations():
    """ 
    Fait les différents calculs qui permettent d'initialiser 
    l'objet core de pyphs
    """
        
    def __init__(self, N=2):
        self.VERBOSE = True
        # --- Constantes --- #
        self.N        = N                  # Nombre de tronçons
        self.Nxi      = 5                  # Nombre d'état par tronçon
        self.N_lambda = self.N-1           # Nombre de contraintes
        self.Nx       = self.N * self.Nxi  # Nombre total d'état sys. non contraint
        self.observers = {}                # Dictionnaire des observateurs
        
        self.create_symbols()
        
        self.compute_Delta_update_eq()
        
        # Matrice d'interconnexion
        self.compute_Jxx()
        self.compute_Jyx()
        self.compute_Jwx()
        
        self.compute_cstrnd_Ham()
        
        # Stockants
        self.X = self.get_uncstr_states()
        self.H = self.Ham_cstrd
        
        # Observateurs
        self.O = sy.Matrix(self.delta_nm_vec)

        
        for i in range(self.N_lambda):
            self.observers[self.O[i]] = self.update_Delta_eq[i] 

        # Dissipations
        self.compute_w()
        self.compute_zw()
            
        # Ports
        self.Y = [-self.qL, self.qR, *[-1*symb for symb in self.vw_vec]]
        self.U = [self.psiL, self.psiR, *self.Fw_vec]

    def create_symbols(self):
        """
        Créé tous les symboles sympy.
        """
        # Vecteur d'état original
        self.nuL_vec  = sy.symbols('nu_L1:{}'.format(self.N+1), **PPTY_STATE_VAR)
        self.nuR_vec  = sy.symbols('nu_R1:{}'.format(self.N+1), **PPTY_STATE_VAR)
        self.Pi_y_vec = sy.symbols('Pi_y1:{}'.format(self.N+1), **PPTY_STATE_VAR)
        self.rho_vec  = sy.symbols('rho_1:{}'.format(self.N+1), **PPTY_PHY_PARAMS)
        self.vol_vec  = sy.symbols('V_1:{}'.format(self.N+1),   **PPTY_STATE_VAR)
        
        # Paramètres
        self.ell_vec  = sy.symbols('ell_1:{}'.format(self.N+1), **PPTY_PHY_PARAMS)
        self.L_vec    = sy.symbols('L_1:{}'.format(self.N+1),   **PPTY_PHY_PARAMS)
        self.V0_vec   = sy.symbols('V_0_1:{}'.format(self.N+1), **PPTY_PHY_PARAMS)
        self.rho_0    = sy.symbols('rho_0', **PPTY_PHY_PARAMS)
        self.gamma    = sy.symbols('gamma', **PPTY_PHY_PARAMS)
        self.P0       = sy.symbols('P_0', **PPTY_PHY_PARAMS)
        self.mu0      = sy.symbols('mu_0', **PPTY_PHY_PARAMS) # viscosité
        
        # ------ États contraints ------
        self.nu_nm_vec = []
        for i in range(self.N_lambda):
            tmp_symb = sy.symbols('nu_{0}{1}'.format(i+1,i+2), **PPTY_STATE_VAR)
            self.nu_nm_vec.append(tmp_symb)
            
        self.delta_nm_vec = []
        for i in range(self.N_lambda):
            tmp_symb = sy.symbols('Delta_{}'.format(10*(i+1) + (i+2)))
            self.delta_nm_vec.append(tmp_symb)
        
        # Dissipations fluide
        self.q_nm_vec   = []
        self.psi_nm_vec = []
        for i in range(self.N - 1):
            tmp_symb = sy.symbols('q_{0}{1}^mu'.format(i+1,i+2), **PPTY_STATE_VAR)
            self.q_nm_vec.append(tmp_symb)
            tmp_symb = sy.symbols('psi_{0}{1}^mu'.format(i+1,i+2), **PPTY_STATE_VAR)
            self.psi_nm_vec.append(tmp_symb)
        
        # Ports
        self.psiL, self.psiR = sy.symbols('Psi_L Psi_R', **PPTY_STATE_VAR)
        self.qL, self.qR  = sy.symbols('q_L q_R', **PPTY_STATE_VAR)
        
        self.vw_vec = sy.symbols('v_w1:{}'.format(self.N+1), **PPTY_STATE_VAR)
        self.Fw_vec = sy.symbols('F_w1:{}'.format(self.N+1), **PPTY_STATE_VAR)
            
            
    ''' =========================================== '''
    ''' =================== Hamiltonians ========== '''        
    def comp_axial_iner_Ham(self):
        self.Q_inertial = sy.zeros(2*self.N)
        oneHalf = sy.Rational(1,2)
        X_inertial = self.get_X1()[0:2*self.N,0]
        
        for i in range(self.N):
            mu_i = (self.m(i+1))/(self.ell(i+1)**2)
            Qi =  mu_i*sy.Matrix([[1,-oneHalf],[-oneHalf,1]])
            self.Q_inertial[2*i:2*i+2, 2*i:2*i+2] = Qi
        
        # Calcule
        self.Ham_iner_axial = X_inertial.T*self.Q_inertial*X_inertial
        
        # Factorisation par mu
        col_vec = [(self.m(i))/(self.ell(i)**2) for i in range(self.N)]
        self.Ham_iner_axial = self.Ham_iner_axial[0,0].collect(col_vec)
        return self.Ham_iner_axial
    
    def comp_trans_iner_Ham(self):
        self.Ham_iner_trans = 0
        coef = sy.Rational(3,2)
        for i in range(self.N):
            self.Ham_iner_trans += coef*self.Piy(i)**2/(self.rho(i)*self.vol(i))
        
        return self.Ham_iner_trans
    
    def comp_thermodynamical_Ham(self):
        self.Ham_comp = 0
        oneHalf = sy.Rational(1,2)
        for i in range(self.N):
            self.Ham_comp += oneHalf*self.gamma*self.P0*(self.rho_tilde(i)/self.rho_0)**2*self.vol(i)
        
        return self.Ham_comp
    
    def compute_cstrnd_Ham(self):
        ''' 
        Computing the constrained Hamiltonian function.
        1) we compute the non constrained form.
        2) We apply the inverse variable change on the constrained states
        3) We replace the old states by their expressions function of the constrained and
            unconstrained states
        '''
        self.Ham_unconstrained = self.comp_axial_iner_Ham() \
                       + self.comp_trans_iner_Ham() \
                       + self.comp_thermodynamical_Ham()
        
        if not hasattr(self, 'M'):
            self.compute_variableChange()
            
        Minv = self.M.inv()
        Minv_nu = Minv[0:2*self.N,:]
        
        # On récupère les états originaux
        X1 = self.get_X1()
        
        # On récupère les états non contraints
        X = self.get_uncstr_states()
        
        # On récupère les deltas
        X_Delta = self.delta_nm_vec
        X_total = sy.Matrix(list(X) + X_Delta)
        
        # On calcule les epxressions pour remplacer
        # Dans le Hamiltonien non contraint
        expr_subs = Minv_nu * X_total
        
        # On créé le subs
        subs = []
        for i in range(2*self.N):
            subs.append((X1[i], expr_subs[i]))
            
        # On substitue
        self.Ham_cstrd = self.Ham_unconstrained.subs(subs)  
        

    ''' =========================================== '''
    ''' ====== Matrice d'interconnexion ============ '''
    def compute_Jxx(self):
        self.Jxx = sy.SparseMatrix(sy.zeros(self.Nx))

        for i in range(self.N):
            self.Jxx[self.Nxi*i:self.Nxi*(i+1), self.Nxi*i:self.Nxi*(i+1)] = self.compute_Ji(i+1)
            
        self.compute_permutation_matrix()
        self.compute_variableChange()
        self.Jxx = self.M*self.P * self.Jxx * self.P.T * self.M.T
        self.Jxx = self.Jxx[0:self.Nx-self.N_lambda, 0:self.Nx-self.N_lambda]
        
    def compute_Ji(self, i):
        ''' sub method to compute the interconnexion matrix of the i-th tract'''
        # INIT
        Ji  = sy.SparseMatrix(sy.zeros(self.Nxi))

        Ji[0,2] = -self.Piy(i)/(self.m(i))
        Ji[0,4] = -1/self.vol(i)
        Ji[1,4] = 1/self.vol(i)
        Ji[1,2] = self.Piy(i)/(self.m(i))
        Ji[2,3] = -self.Sw(i)
        Ji[2,4] = self.Sw(i)*self.rho(i)/self.vol(i)

        # Antisymétrie
        Ji = Ji -Ji.T

        return Ji
    
    def compute_permutation_matrix(self):
        '''
        Creates a permutation matrix to rearrange the original J matrix
        into one that fits the format of document/article.
        '''
        self.P = sy.SparseMatrix(sy.zeros(self.Nx))

        for i in range(self.N):
            # nu_L/ nu_R
            self.P[2*i, self.Nxi*i]         = 1
            self.P[2*i+1, self.Nxi*i+1]     = 1
            # Piy
            self.P[2*self.N+i, self.Nxi*i+2] = 1
            # rho
            self.P[3*self.N+i, self.Nxi*i+3] = 1
            # vol
            self.P[4*self.N+i, self.Nxi*i+4] = 1
            
    def compute_Jyx(self):
        self.Jyx = sy.SparseMatrix(sy.zeros(self.N + 2, self.Nx-self.N_lambda))
        self.Jyx[0,0] = -1
        self.Jyx[1, 2*self.N-self.N_lambda-1] = 1
        self.Jyx[2::,2*self.N-self.N_lambda:3*self.N-self.N_lambda] = -1* sy.eye(self.N)
        
        #Jyx[1, 2*N-N_lambda-1] = 1
        #Jyx[2::,2*N-N_lambda:3*N-N_lambda]= -1* sy.eye(N)

    def compute_Jwx(self):
        self.Jwx = sy.SparseMatrix(sy.zeros(self.N+1,
                                            self.Nxi*self.N-self.N_lambda))
        self.Jwx[0:self.N+1, 0:self.N+1] = sy.SparseMatrix(sy.eye(self.N+1))

    ''' =========================================== '''
    ''' ======= Changement de variable ============ '''
    def compute_variableChange(self):
        ''' Goes from a DAE-PHS formulation to a constrainted pHs formulation'''
        self.compute_B()
        self.compute_B_left_annihilator()
        self.M = sy.SparseMatrix(sy.zeros(self.Nx, self.Nx))
        self.M[0:self.Nx- self.N_lambda,::]       = self.annul_b
        self.M[self.Nx- self.N_lambda::, ::] = (self.b.T*self.b).inv()*self.b.T
    
    def compute_B(self):
        ''' Computes the matrix that translates the constraintes
        in function of the gradient of the Hamiltonian'''
        bT = sy.SparseMatrix(sy.zeros(self.N_lambda, self.Nx))
        for i in range(self.N_lambda):
            bT[i, 1+2*i] = 1 
            bT[i, 2+i*2] = -1
        self.b = bT.T
        
    def compute_B_left_annihilator(self):
        self.annul_b = sy.SparseMatrix(sy.zeros(self.Nx-self.N_lambda, self.Nx))

        self.annul_b[0,0] = 1
        self.annul_b[self.N::, self.N + self.N_lambda::] = sy.eye(3*self.N+1)
        for i in range(1,self.N_lambda+1):
            self.annul_b[i, 2*i-1] = 1 
            self.annul_b[i, 2*i] = 1
            
    ''' ================================================= '''
    ''' ====== Building constrained states equations ===='''
    def compute_Delta_update_eq(self):
        self.compute_Q22()
        self.compute_Q12()
        X_nu_temp = sy.Matrix([self.nuL_vec[0]] + self.nu_nm_vec + [self.nuR_vec[-1]])
        #Q22inv = self.Q22.inv(method='LDL')
        self.update_Delta_eq = LDL_solve(self.Q22, -1*self.Q12*X_nu_temp).applyfunc(fcancel)
    
    def compute_Q22(self):
        self.Q22 = sy.SparseMatrix(sy.zeros(self.N_lambda))
        for i in range(self.N_lambda):
            for j in range(self.N_lambda):
                if i == j:
                    self.Q22[i, j] = self.mu_add_mu(i+1,i+2)
                if j == i-1:
                    self.Q22[i, j] = sy.Rational(1,2) * self.mu(i+1)
                if j == i+1:
                    self.Q22[i, j] = sy.Rational(1,2) *self.mu(i+2)
        return self.Q22
    
    def compute_Q12(self):
        self.Q12 = sy.zeros(self.N_lambda, self.N+1)
        for i in range(self.N_lambda):
            for j in range(self.N+1):
                if i == j:
                    self.Q12[i, j] = -sy.Rational(1,2) * self.mu(i+1)
                if j == i+1:
                    self.Q12[i, j] = sy.Rational(1,2) * self.mu_minus_mu(i+1,i+2)
                if j == i+2:
                    self.Q12[i, j] = sy.Rational(1,2) * self.mu(i+2)
        self.Q12 = sy.SparseMatrix(self.Q12)   
        return self.Q12
    
    ''' ==================================== '''
    ''' =========== Dissipation ============ '''
    def compute_w(self):
        self.w = [self.qL, *(self.q_nm_vec), self.qR]

    def compute_zw(self):
        self.zw = []
        coef_visq = 16*sy.pi*self.mu0

        # Premier tronçon, resistance gauche
        expr = coef_visq * self.ell(1)**2 / self.vol(1)
        self.zw.append(expr)

        # Resistance équivalente interne
        for i in range(self.N-1):
            tract_ind = i+1
            expr = coef_visq*(self.ell(i+1)**2/self.vol(i+1) + \
                              self.ell(i+2)**2/self.vol(i+2))
            self.zw.append(expr)

        # Dernière
        expr = coef_visq * self.ell(self.N)**2 / self.vol(self.N)
        self.zw.append(expr)
        
    ''' =========================================== '''
    ''' =================== Accesseurs ============ '''
    def get_Xi(self, i):
        """ Returns the state vector of the i-th tract"""
        Xi = [
              self.nuL_vec[i-1], 
              self.nuR_vec[i-1],
              self.Pi_y_vec[i-1],
              self.vol_vec[i-1],
              self.rho_vec[i-1]
             ]
        return sy.Matrix(Xi)
    
    def get_X1(self):
        """ Returns the state vector of original states"""
        X = []
        for i in range(self.N):
            X += [self.nuL_vec[i], self.nuR_vec[i]]
        for i in range(self.N):
            X += self.Pi_y_vec + self.vol_vec + self.rho_vec
        
        return sy.Matrix(X)
    
    def get_uncstr_states(self):
        ''' Returns the state vector of new pHs'''
        X = []
        X.append(self.nuL_vec[0])
        X += self.nu_nm_vec
        X.append(self.nuR_vec[-1])
        X += self.Pi_y_vec
        X += self.vol_vec
        X += self.rho_vec
        return X

    def nuL(self, i):
        return self.nuL_vec[i-1]
    
    def nuR(self, i):
        return self.nuR_vec[i-1]
    
    def Piy(self, i):
        return self.Pi_y_vec[i-1]
    
    def rho(self, i):
        """ Absolute value of rho"""
        return self.rho_0 + self.rho_vec[i-1]
    
    def rho_tilde(self, i):
        """ Fluctuation around rho_0"""
        return self.rho_vec[i-1]
    
    def vol(self, i):
        return self.V0_vec[i-1] + self.vol_vec[i-1]
    
    def ell(self,i):
        return self.ell_vec[i-1]
    
    def Sw(self, i):
        return 2*self.ell_vec[i-1]*self.L_vec[i-1]
    
    def m(self,i):
        return (self.rho(i) * self.vol(i))
    
    def mu(self,i):
        symb = self.phy_symb('mu', ind=i)
        if symb not in self.observers.keys():
            self.observers[symb] = (self.m(i)/self.ell(i)**2)
            
        return symb
    
    def mu_add_mu(self,ind1, ind2):
        ''' 
        Returns a symbols that looks like "mu_1 \oplus mu_2" 
        which will be calculated separetely. It is stored as an observer
        '''
        str_symb = '(mu_' + str(ind1) + '_add_mu_' + str(ind2) + ')'
        
        symb = sy.symbols(str_symb, **PPTY_STATE_VAR)
        
        if symb not in self.observers.keys():
            self.observers[symb] = self.mu(ind1) + self.mu(ind2)
            
        return symb
    
    def mu_minus_mu(self,ind1, ind2):
        ''' 
        Returns a symbols that looks like "mu_1 \oplus mu_2" 
        which will be calculated separetely. It is stored as an observer
        '''
        str_symb = '(mu_' + str(ind1) + 'minus_mu_' + str(ind2) + ')'
        
        symb = sy.symbols(str_symb, **PPTY_STATE_VAR)
        
        if symb not in self.observers.keys():
            self.observers[symb] = self.mu(ind1) - self.mu(ind2)
            
        return symb

    def qmu(self,i):
        return self.q_nm_vec[i-1]
    
    def psimu(self,i):
        return self.psi_nm_vec[i-1]
    
    def phy_symb(self, str_symb, ind=None):
        assert type(str_symb) is str, "First arg must be a string"

        if ind is None:
            symb = sy.symbols(str_symb, **PPTY_STATE_VAR)
        else:
            assert type(ind) is int, "keyarg must be int"
            symb = sy.symbols(str_symb + '_' + str(ind), **PPTY_STATE_VAR)
        return symb
   

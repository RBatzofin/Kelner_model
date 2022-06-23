import astropy.constants as const
import astropy.units as u
import numpy as np
from scipy.integrate import quad

class Kelner_Model:
    r"""Secondary particle (π mesons, gamma rays, electrons and neutrinos) 
    emission from a proton population based on the work by Kelner et al.
    doi: 10.1103/PhysRevD.74.034018. 
    URL https://link.aps.org/doi/10.1103/PhysRevD.74.034018

    Parameters
    ----------
    alpha : float
        The spectral index of the proton spectra

    beta : float
        The reference energy for normalisation

    E_0 : `~astropy.units.Quantity` float
        The cutoff energy of the proton spectra

    particle_type : str
        Select the secondary particle spectrum you want:
        gamma rays = 'gamma'
        electrons = 'electron'
        muonic neutrinos from direct decay = 'muon1'
        muonic neutrinos total spectrum = 'muon'
        for muonic neutrinos from the decay of the muon use 'electron' 
        since these are approximately equal. 

    nH : `~astropy.units.Quantity` float
        Number density of the target protons. Default is :math:`1
        \mathrm{cm}^{-3}`.

    amplitude : `~astropy.units.Quantity` float
        Give the normalisation of the model.
    """
    
    def __init__(
        self,
        alpha = 2,
        beta = 1, 
        E_0 = 1000*u.TeV, 
        particle_type = 'muon' , 
        nH = 1.0*u.cm**-3, 
        amplitude = None
        ):

        self.nH = nH
        self.beta = beta
        self.K_pi = 0.17
        self.m_p = (const.m_p*const.c**2).to(u.TeV)
        self.m_pi = 1.3497664e-4 *u.TeV
        self.alpha = alpha
        self.E0 = E_0
        self.A = self.get_A()
        self.amplitude = amplitude
        
        if particle_type == 'gamma':
            self.particle_function = self.F_gamma
        elif particle_type == 'electron':
            self.particle_function = self.F_electron
        elif particle_type == 'muon':
            self.particle_function = self.F_muon_neutrino
        elif particle_type == 'muon1':
            self.particle_function = self.F_muon_neutrino1
        
        self.nhat = (self.high_energy_spec(0.1*u.TeV)
                    /self._low_energy_spec(0.1*u.TeV))
        
    def F_gamma(self,x, Ep):
        """
        Total spectrum of gamma rays Kelner Eq. 58

        Parameters
        ----------
        x : float
            Energy of the gamma ray divided by energy of the proton

        Ep : float
            Incident proton energy

        Returns
        -------
        F_gamma : float
            Total spectrum of gamma rays.
        """
        L = np.log(Ep/u.TeV)
        B_gamma = 1.3 +0.14*L + 0.011*L**2
        beta_gamma = (1.79 +0.11*L + 0.008*L**2)**-1
        k_gamma = (0.801 +0.049*L + 0.014*L**2)**-1
        
        One = B_gamma * np.log(x) / x
        Two = (1-x**beta_gamma)/(1 + k_gamma*x**beta_gamma *(1-x**beta_gamma))
        
        Three = (
            1/np.log(x) - (4*beta_gamma*x**beta_gamma)/(1-x**beta_gamma) 
            - (4 * k_gamma * beta_gamma * x**beta_gamma*(1-2*x**beta_gamma))
            /(1 + k_gamma*x**beta_gamma * (1-x**beta_gamma))
            )

        return One * Two**4 * Three 

    def F_electron(self,x, Ep):
        """
        Total spectrum of electrons Kelner Eq. 62

        Parameters
        ----------
        x : float
            Energy of the electron divided by energy of the proton

        Ep : float
            Incident proton energy

        Returns
        -------
        F_electron : float
            Total spectrum of electrons.
        """
        L = np.log(Ep/u.TeV)
        Be = (69.5 + 2.65*L + 0.3*L**2)**-1
        beta_e = (0.201 +0.062*L + 0.00042*L**2)**-(1/4)
        k_e = (0.279 + 0.141*L + 0.0172*L**2)/(0.3 + (2.3+L)**2)
        
        numerator = 1 + k_e*(np.log(x))**2
        denominator = x*(1+ (0.3/x**beta_e))
        
        return Be * (numerator**3)/denominator *(-np.log(x))**5

    def F_muon_neutrino1(self,x,Ep):
        """
        Spectrum of muonic neutrinos from direct decay Kelner Eq. 66

        Parameters
        ----------
        x : float
            Energy of the muon divided by energy of the proton

        Ep : float
            Incident proton energy

        Returns
        -------
        F_muon_neutrino1 : float
            Spectrum of muonic neutrinos.
        """
        if x >= 0.427:
            return 0
        y = x/0.427
        L = np.log(Ep/u.TeV)
        B = 1.75 + 0.204*L + 0.01*L**2
        b = (1.67 + 0.111*L + 0.0038*L**2)**-1
        k = 1.07 - 0.086*L + 0.002*L**2
        
        One = B*(np.log(y)/y)
        Two = (1-y**b)/(1 + k*y**b *(1-y**b))
        Three = (
            1/np.log(y) - (4*b*y**b)/(1-y**b)
            - (4*k*b*y**b *(1-2*y**b))/(1+k*y**b *(1-y**b))
        )

        return One * Two**4 * Three
        
    def F_muon_neutrino(self,x,Ep):
        """
        Total spectrum of muonic neutrinos

        Parameters
        ----------
        x : float
            Energy of the muon divided by energy of the proton

        Ep : float
            Incident proton energy

        Returns
        -------
        F_muon_neutrino : float
            Total spectrum of muonic neutrinos.
        """
        return self.F_electron(x,Ep) + self.F_muon_neutrino1(x,Ep)	

    def sigma_inel(self,Ep):
        """
        Inelastic cross-section for p-p interaction. Kelner eq 73

        Parameters
        ----------
        Ep : float
            Incident proton energy

        Returns
        -------
        sigma_inel : float
            Inelastic cross-section

        """
        try:
            L = np.log(Ep/u.TeV)
        except:
            Ep *= u.TeV
            L = np.log(Ep/u.TeV)
        E_th = 1.22 *10**-3 *u.TeV
        sigma_approx =  (34.3 +1.88*L + 0.25*L**2)
        factor = 1-(E_th/Ep)**4
        
        if Ep <= E_th:
            factor = 0.0
        
        return sigma_approx * factor**2 *10**-27 *u.cm**2

    def get_A(self):
        """
        Constant determined from the condition in Kelner eq 80
        """
        integration = quad(lambda Ep: Ep/(Ep**self.alpha) 
        * np.exp(-((Ep*u.TeV)/self.E0)**self.beta),
        1, np.inf, epsabs=0)

        A = (1*u.erg *u.cm**-3)/(integration[0])
        
        return A.to(u.TeV /u.cm**3)
    
    def Jp(self,E):
        """
        Particle distribution Kelner eq 74

        Parameters
        ----------
        E : Incident proton energy

        Returns
        -------
        Jp : float
            Particle distribution
        """
        return (self.A/(E**self.alpha))*np.exp(-(E/self.E0)**self.beta)
        
    def int_for_high_energy(self,x,E):
        """
        Function to integrate in Kelner eq 71
        """
        try:
            
            return (self.sigma_inel(E/x)*self.Jp(E/x)
            *self.particle_function(x,E/x)/x).value
        except:
            
            E *= u.TeV
            
            return (self.sigma_inel(E/x)*self.Jp(E/x)
            *self.particle_function(x,E/x)/x).value

    def high_energy_spec(self, E):
        """
        Spectrum for energies > 0.1 TeV
        """
        spec =  quad(self.int_for_high_energy
        ,0.0,1.0,args=E,epsrel = 10**-13, epsabs=0)
        return const.c.cgs.value*spec[0] *u.cm**-3 /u.s /u.TeV
    
        
    def int_for_delta(self,E):
        """
        Function to integrate in Kelner eq 78
        """
        E *= u.TeV
        if E.unit != u.TeV:
            E /= u.TeV
        q_pi = ((const.c * self.nH /self.K_pi)
            *self.sigma_inel(self.m_p+E/self.K_pi)
            *self.Jp(self.m_p+E/self.K_pi))
        
        return ((q_pi/(np.sqrt(E**2 - self.m_pi**2)))
        .to(u.cm**-3 *u.TeV**-self.alpha /u.s)).value
        
    def _low_energy_spec(self,E):
        E_min = E + (self.m_pi**2)/(4*E)
        
        spec = quad(self.int_for_delta,E_min.value,np.inf, epsrel = 1e-13, epsabs=0)
        return 2*spec[0] *u.cm**-3 /u.TeV /u.s
        
    def low_energy_spec(self,E):
        """
        Spectrum for energies < 0.1 TeV
        """
        return self.nhat*self._low_energy_spec(E)
        
    def energy_spec_function(self, E):
        """
        Energy spectrum from p-p interactions
        Parameters
        ----------
        E : `~astropy.units.qunatity` float or array
        Energy range to calculate the spectrum

        Returns
        -------
        energy_spec_function : `~astropy.units.quantity` float or array
        """
        flux = []
        unit = u.TeV**-1 /u.s /u.cm**3
        try:
            for e in E:
                if e >= 0.1*u.TeV:
                    _flux = self.high_energy_spec(e)
                elif e < 0.1*u.TeV:
                    _flux = self.low_energy_spec(e)
                if self.amplitude:
                    unit = u.TeV**-1 /u.s /u.cm**2
                    _flux = (_flux/(4*np.pi*u.kpc**2) 
                            /self.A *self.amplitude.value 
                            *u.TeV *u.TeV/u.erg)

                    _flux = _flux.to(u.TeV**-1 /u.s /u.cm**2)
                flux.append(_flux.value)
        except:
            if E >= 0.1*u.TeV:
                _flux = self.high_energy_spec(E)
            elif E < 0.1*u.TeV:
                _flux = self.low_energy_spec(E)
            if self.amplitude:
                unit = u.TeV**-1 /u.s /u.cm**2
                _flux = (_flux/(4*np.pi*u.kpc**2) 
                        /self.A *self.amplitude.value 
                        *u.TeV *u.TeV/u.erg)
                        
                _flux = _flux.to(u.TeV**-1 /u.s /u.cm**2)
            flux = _flux.value
            
            
        return flux	* unit
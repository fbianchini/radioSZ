import numpy as np
import astropy.constants as const
import astropy.units as u

c_light = const.c.value
h_planck = const.h.value
k_B = const.k_B.value
jansky = 1.0e-23

nucm2nuGHz = 29.979246

SI2MJy = (u.Hz**3*u.s**3*u.J/u.m**2).to('MJy')


def j2k(nu, T_cmb=2.725):
	""" 
	Returns the conversion factor between Jansky units and CMB Kelvin. 
	Parameters
	----------
	nu : float
		Frequency in [GHz]
	"""
	x = h_planck*(nu*1e9)/(k_B*T_cmb)
	g = (np.exp(x) - 1.)**2 / x**2 / np.exp(x)
	return c_light**2 / (2. * (nu*1e9)**2 * k_B) * g * 1.e-26


def k2j(nu, T_cmb=2.725):
	""" 
	Returns the conversion factor between CMB Kelvin and Jansky units. 
	Parameters
	----------
	nu : float
		Frequency in [GHz]
	"""
	return 1.0 / j2k(nu, T_cmb=T_cmb)

def I2T(I, nu):
    """ 
    Returns the brightness temperature (in K) given the intensity I (in MJy/sr) at frequency \nu (in GHz). 
    """
    fact = c_light**2 / (2*k_B*(nu*1e9)**2)
    return fact * I / SI2MJy 


def T2I(T, nu):
    fact = (2*k_B*(nu*1e9)**2) / c_light**2
    return fact * T * SI2MJy

def B_nu(nu, T0=2.725, mu=0):
    """ 
    Returns the planck blackbody function (in MJy/sr) at frequency \nu (in GHz) for a blackbody with temperature T (in K). 
    Parameters
    ----------
    nu : float
        Frequency in GHz
    """
    x = h_planck*(nu*1e9)/(k_B*T0)
    # MJy/sr
    return 2*h_planck*(nu*1e9)**3 / c_light**2 / (np.exp(x+mu) - 1.) * SI2MJy


def dB_nu_dT(nu, T0=2.725, mu=0):
    """ 
    Returns the derivative of the planck blackbody function (in MJy/sr) at frequency \nu (in GHz) for a blackbody with temperature T (in K). 
    Parameters
    ----------
    nu : float
    Frequency in GHz
    """
    x = h_planck*(nu*1e9)/(k_B*T0)
    return x * np.exp(x)/(np.exp(x)-1) * B_nu(nu, T0=T0, mu=mu) / T0

def dI_I_tSZ(nu, y, T0=2.725):
    """ 
    Returns the intensity tSZ shift at frequency \nu (in GHz) for a blackbody with temperature T. 
    """
    x = h_planck*(nu*1e9)/(k_B*T0)
    return y * x*np.exp(x)/(np.exp(x)-1) * (x*(np.exp(x) + 1.) / (np.exp(x) - 1.) - 4.0)

def dI_I_R(y, gamma=3.59):
    """ 
    Returns the intensity radio shift at frequency \nu (in GHz). 
    """
    return y * gamma * (gamma-3)

def T_R(nu, z=0, nu0=0.31, T0=24.1, alpha=-2.59):
    """ 
    Returns the radio synchrotron background temperature (in K) at frequency \nu (in GHz) 
    """
    return T0 * (nu/nu0)**alpha * (1+z)

def dT_T_tSZ(y):
    """ 
    Returns the tSZ temperature shift at frequency \nu (in GHz) 
    """
    return -2*y
    
def dT_T_R(y, gamma=3.59, f_zc=1):
    """ 
    Returns the radio temperature shift at frequency \nu (in GHz). 
    """
    return y * f_zc * gamma * (gamma-3)

def dT_SZ(nu, y, z=0, f_zc=1, gamma=3.59, nu0=0.31, T_cmb=2.725, T0=24.1, alpha=-2.59):
    """ 
    Returns the sum of the radio SZ and tSZ. 
    """
    T_radio = T_R(nu, z=z, nu0=nu0, T0=T0, alpha=alpha)
    return T_cmb*dT_T_tSZ(y) + T_radio*dT_T_R(y, gamma=gamma, f_zc=f_zc)

def dT_kSZ(nu, tau, beta, z=0, f_zc=1, gamma=3.59, T_cmb=2.725, nu0=0.31, T0=24.1, alpha=-2.59):
    """ 
    Returns the radio kSZ and kSZ
    """
    T_radio = T_R(nu, z=z, nu0=nu0, T0=T0, alpha=alpha)
    return (T_cmb + T_radio*f_zc*gamma) * tau * beta


def dT_tot(nu, y, tau, beta, z=0, f_zc=1, gamma=3.59, nu0=0.31, T_cmb=2.725, T0=24.1, alpha=-2.59):
    """ 
    Returns the total SZ + kSZ 
    """
    return dT_SZ(nu,  y, z=z, f_zc=f_zc, gamma=gamma, T_cmb=T_cmb, nu0=nu0, T0=T0, alpha=alpha) + \
           dT_kSZ(nu, tau, beta, z=z, f_zc=f_zc, gamma=gamma, T_cmb=T_cmb, nu0=nu0, T0=T0, alpha=alpha)

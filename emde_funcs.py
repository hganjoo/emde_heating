#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 16:22:48 2023

@author: hganjoo
"""

import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.special import hyp2f1,gamma,digamma,psi,erfc
import pandas as pd
from scipy.special import lambertw as W
from scipy.integrate import trapz,cumtrapz
import warnings
warnings.filterwarnings('ignore')


# Constants and params



OmegaM = 0.3089
OmegaB = 0.04
h = 0.6727

rhocrit = 2.775 * h * h * 1e11 # Msun/Mpc^3

aeq = 4.15 / (h*h*OmegaM) * 1e-5

OmegaC = OmegaM - OmegaB

rhobar = rhocrit*OmegaM


st = -1 # Y particle stats
g = 1.
p = 1.
b = 2.7

if st==1:
    b = 3.15
    p = 7./8


dcrit = 1.686

# Fit func for delta_crit(a)

def fdc(x):
    A = 0.3549
    B = -0.2331
    C = 0.0533
    D = 0.4935
    E = -0.2092
    F = 0.09327
    G = 0.2683
    
    return A * (np.log(1 + x) - (x + B*x*x + C*x*x*x)/(1 + D*x + E*x*x + F*x*x*x)) / (1 + G*(np.log(1+x) -2*x/(2+x)))

def gdc(x,y):
    A = 135.2*(1 + 0.04734*y + 0.0006373*y*y)
    B = 1.093*(1 + 0.03256*y + 0.0005114*y*y)
    C = 17.87*(1 + 0.03501*y + 0.0003641*y*y)
    D = 3.187*(1 + 0.03283*y + 0.0005260*y*y)
    E = 0.05388*(1-0.738*y)
    return np.power(1 + np.power(A*x,B) + np.power(C*x,D),-1*E)

def hdc(x):
    return 0.07074*np.exp(-0.118*np.power(x + 0.4258,2))


def dc(a,Trh):
    arh = (1/1.02) * np.power(3.91/gstar(0.204*Trh),1./3) * (T0/Trh)
    
    return dcrit + fdc(a/arh)*gdc(a/aeq,np.log(arh/aeq)) - hdc(np.log(a/aeq))
    
    
# Helper funcs for bound fraction using kR = 2.5 sharp-k filter   


def get_sig(kvals,pk):
        
    rs = 2.5/kvals
    rs = rs[::-1]
    
    rs = np.log(rs)
    rs = rs[:-1] + 0.5*np.diff(rs)
    rs = np.exp(rs)
    
    
    sig =  np.sqrt(cumtrapz(pk,dx=np.diff(np.log(kvals))[0])[::-1])
    
    return rs*h,sig

def diffbound(pk,ks,a,trh):

    rs = 2.5/ks
    rs = rs[::-1]

    sig =  np.sqrt(cumtrapz(pk,dx=np.diff(np.log(ks))[0])[::-1])

    nu = dc(a,trh)/sig
    nu = nu[:-1] + 0.5*np.diff(nu)

    rs = rs[:-1] + 0.5*np.diff(rs)

    lnr = np.log(rs)

    dfdlnr = np.sqrt(2./np.pi) * np.abs( np.diff(np.log(sig))/np.diff(lnr) ) * nu * np.exp(-0.5*nu*nu)

    lnr = lnr[:-1] + 0.5*np.diff(lnr)[0]
    r = np.exp(lnr)

    return dfdlnr,r

def boundf(pk,ks,a,trh):

    dfdlnr,r = diffbound(pk,ks,a,trh)
    return trapz(dfdlnr,x=np.log(r))




# EMDE power spectrum functions

As = np.exp(3.089)*1e-10
ns = 0.965
k0 = 0.05 # Mpc ^(-1)

pi = np.pi

def arh_r(T): return (1/1.02) * np.power(3.91/gstar(0.204*T),1./3) * (T0/T)

def H_RD(T): return np.sqrt((8*pi*pi*pi/90)*gstar(T))*T*T/mpl

mpl = 1.221e19 #GeV
T0 = 2.348e-13 #GeV

G = 1./(mpl*mpl)

sec_inv_GeV = 6.58e-25
GeV_in_Mpc_inv = (1.97e-16 * 3.24078e-23)**-1.0

def k_rh(trh):
    return arh_r(trh)*H_RD(trh)*GeV_in_Mpc_inv

rho_conv = 34.238e56 # Convert Gev^4 to Msun / (Mpc)^3

gstar_file = np.loadtxt('gstar.dat')
temps = gstar_file[:,0] # Temps loaded in GeV
gstars = gstar_file[:,1]

gstar = interp1d(temps,gstars,fill_value=(gstars[0],gstars[-1]),bounds_error=False)

def npr(xi):
    if xi > 1:
        return 2.7
    else:
        if xi <= 0.1:
            return 2.2
        else:
            return 2.2 - 0.29*(xi - 0.1)


def kpk(ky,xi,b):
    
    if xi < 1:
        return ky * (2.06/b) / np.sqrt(1 + xi)
    
    if b == 2.7: a = 1.26
    if b == 3.15: a = 1.28
    
    if xi > 50:
        
        X = np.power(3*0.594*xi/1.43,2./3)
        return ky * (a/b)*np.real(np.power(W(0.67*X),-1.5))
    
    else:
        
        return ky * ((a+0.58)/b)*np.power(xi,-0.299)
    

def kc(ky,xi,b):
    
    if xi < 1:
        nn = npr(xi)
        return kpk(ky,xi,b)*np.power(0.5*nn,1./nn)
    
    else:
        if b == 2.7: a = 1.26
        if b == 3.15: a = 1.28
        n = 2.7
        
        if xi > 50:
        
            X = np.power(np.real(W(0.77*xi**(2./3))),-1.5)
            x = ((0.58+a)/b) * X * np.power(np.log(0.18*xi*X),1/n) * np.sqrt(1 + 1/xi)
            
            return ky * x
        
        else:
            
            return ky * ((a + 0.71)/b)*np.power(xi,-0.21)
    
def tk_ss(k,ky,xi,b):
    
    kcut = kc(ky,xi,b)
    ks = k/kcut
    
    return np.exp(-1*np.power(ks,npr(xi)))


keq = 0.073 * OmegaM * h * h # 1 / Mpc
Step = lambda y: .5*(np.tanh(.5*y)+1)
Afit = lambda y: np.exp(0.60907/(1 + 2.149951*(-1.51891 + np.log(y))**2)**1.3764)*(9.11*Step(5.02 - y) + 3./5*y**2*Step(y - 5.02))
Bfit = lambda y: np.exp(np.log(0.594)*Step(5.02 - y) + np.log(np.e/y**2)*Step(y - 5.02))
aeqOVERahor = lambda x,y: x*np.sqrt(2)*(1 + y**4.235)**(1/4.235)
  

a1 = (1-(1+24*OmegaC/OmegaM)**.5)/4.
a2 = (1+(1+24*OmegaC/OmegaM)**.5)/4.
B = 2*digamma(1)-digamma(a2)-digamma(a2+.5)
f2_over_f1 = B/np.log(4/np.e**3)

def Rt(xdec,x):
    return (Afit(x/(xdec))*np.log((4/np.exp(3))**f2_over_f1*Bfit(x/(xdec))*aeqOVERahor(x, x/(xdec))))/(9.11*np.log((4/np.exp(3))**f2_over_f1*0.594*x*np.sqrt(2)))
  

# Returns k, delta / Phi0 and kRH for given m,trh,eta at aRH
def getps(m,trh,xi):    
    
    krh = arh_r(trh)*H_RD(trh) * GeV_in_Mpc_inv
    k = np.logspace(-2,15,5000)
    xdec = krh/keq
    ky = b * np.power(g*p/gstar(trh),1./6) * np.power((m/b)/trh,2./3) * np.sqrt(1 + xi)
    x = k/keq
    y = k/krh
    tk = np.where(x<0.05*0.86*xdec,1,Rt(0.86*xdec,x)) 
    
    if xi > 1:
        kd = 1.414*np.power(p*g/gstar(trh),1./6)*np.power(m/(b*trh),2./3)*np.power(xi,-0.5)
        q = k/(kd*krh)
        dyc = tk * np.log(1 + 0.22*q) * np.power(1 + 1.11*q + (0.94*q)**2 + (0.63*q)**3 + (0.45*q)**4,-0.25) / (0.22*q)  
    else:
        
        dyc = tk
    
    dy = dyc * tk_ss(y,ky,xi,b)
    #prefac = As*np.power(k/k0,ns-1)*(4./9)
    
    #ps = a*a*prefac*dy*dy
    
    return k,dy,krh


# Returns (k,full dimensionless PS at a<aRH, kRH) for given m,trh,eta
def getps_arh(m,trh,xi,a=1.):    
    
    krh = arh_r(trh)*H_RD(trh) * GeV_in_Mpc_inv
    k = np.logspace(np.log10(5),6,5000)*krh
    ky = b * np.power(g*p/gstar(trh),1./6) * np.power((m/b)/trh,2./3) * np.sqrt(1 + xi)
    
    if xi > 1:
        kd = 1.414*np.power(p*g/gstar(trh),1./6)*np.power(m/(b*trh),2./3)*np.power(xi,-0.5)
        q = k/(kd*krh)
        dyc = 0.596 * kd*kd * q*q*np.log(1 + 0.22*q) * np.power(1 + 1.11*q + (0.94*q)**2 + (0.63*q)**3 + (0.45*q)**4,-0.25) / (0.22*q)    
    else:
        
        dyc = 0.56*np.power(k/krh,2.)
    
    dy = dyc * tk_ss(k/krh,ky,xi,b)
    prefac = As*np.power(k/k0,ns-1)*(4./9)
    
    ps = a*a*prefac*dy*dy
    
    return k,ps,krh

# Returns bound fraction at aRH
def bf(m,trh,xi,a=1.):
    # this is a = a/aRH
    
    k,p,krh = getps_arh(m,trh,xi)
    return boundf(p,k,arh_r(trh),trh)


# Truncated linear sigma_v at aRH, in natural units
def svds(k,p,krh,th=4.3):
    
    try:
        
        #in1 = np.where(pis[i]<=th,pis[i],np.sqrt(th*pis[i]))
        #in1 = np.where(p<=th,p,th)
        in1 = np.where(p<=th,p,(th*th)/p)
        
        return krh * np.sqrt(np.trapz(y=in1/(k*k),x=np.log(k)))
    
    except:
        return (np.sqrt(cumtrapz(y=p/(k*k),x=np.log(k))) * krh)[-1]


# The gravitational heating free-streaming scale in units of kRH, at scale factor "a"
def kfs_aeq(m,trh,xi,a=aeq):
    # kfs around a_eq, including matter-rad transition
    # a is scale factor
    
    k,p,krh = getps_arh(m,trh,xi)
    arh = arh_r(trh)
    #sv_tr = svds(k,p,krh)/0.44
    alpha = 2.23 / svds(k,p,krh)
    lfac = np.log((a/arh)*(2 / (1 + np.sqrt(1 + a/aeq)))**2)
    kfs = alpha / lfac
    return kfs


# Free-streaming cutoff function at scale factor "a"
def fsc(k,m,trh,xi,a):
    kf = kfs_aeq(m,trh,xi,a) * k_rh(trh)
    f = 1./(1 + np.power(k/kf,2.5))
    return f




# Load base Pk
    
x = np.loadtxt('planck15_500_ex2.dat')
klin = x[:,0]
plin = x[:,1]*(2*np.pi)**3
plin = klin*klin*klin*plin/(2*pi*pi)

basepk = interp1d(klin,plin)
afac = aeq/(1./(1 + 500))

# Growth factor

fb = 0.157
mu1 = 1.25*np.sqrt(1 - 24*fb/25) - 0.25
mu2 = -1.25*np.sqrt(1 - 24*fb/25) - 0.25
    
def D1(a):
      return (1+a/aeq)**mu1*hyp2f1(-mu1,.5-mu1,.5-2.*mu1,1./(1+a/aeq))
  
def D2(a):
      return (1+a/aeq)**mu2*hyp2f1(-mu2,.5-mu2,.5-2.*mu2,1./(1+a/aeq))
    
def gf(a,arh):
    mu1 = 5./4 * np.sqrt(1-24./25*fb) - 1./4
    mu2 = -5./4 * np.sqrt(1-24./25*fb) - 1./4
    A1 = aeq**mu1
    A2 = -A1
    A2 *= gamma(.5-2.*mu1)*gamma(.5-mu2)*gamma(-mu2)
    A2 /= gamma(.5-2.*mu2)*gamma(.5-mu1)*gamma(-mu1)
    A2 *= np.log(1.733*aeq/arh)-psi(.5-mu1)-psi(-mu1)+2*psi(1)
    A2 /= np.log(1.733*aeq/arh)-psi(.5-mu2)-psi(-mu2)+2*psi(1)
    
    return A1*D1(a) + A2*D2(a)

def af(a,arh):
    return (gf(a,arh) / gf((1./501),arh))**2


# Matter-Radiation Equality Power Spectra

# return lcdm, emde and freestreaming PS at scale factor "a"
def psaeq(m,trh,xi,a=aeq):
    
    
    k,p0,krh = getps(m,trh,xi)
    arh = arh_r(trh)
    #bfrh = bf(m,trh,xi)
    #base = afac*afac*basepk(k)
    base = af(a,arh) * basepk(k)
    pkk = p0 * p0 * base
    
    pcut =  pkk * fsc(k,m,trh,xi,a)**2
    
    return k,base,pkk,pcut
    


# Bound fraction of freestreaming and lcdm PS in matter-domination: a = av/aeq  
# this is av = a/a_eq
def bfaeq(m,trh,xi,av=1.):
    
    k1,b1,p1,pc1 = psaeq(m,trh,xi,av*aeq)
    
    return boundf(pc1,k1,av*aeq,trh),boundf(b1*np.exp(-1*np.power(k1/1e6,2)),k1,av*aeq,trh),boundf(p1,k1,av*aeq,trh)



def pki(k,p):
    return trapz(p,x=np.log(k))


def sigeq(m,trh,xi):
    k1,b1,p1,pc1 = psaeq(m,trh,xi)
    return np.sqrt(pki(k1,p1))




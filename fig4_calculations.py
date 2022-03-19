#!/usr/bin

'''
Script for fitting three datasets from Middleton et al. 2006 with XSPEC using randomly sampled distance, inclination, and mass values.
See Mills et al. 2021 for a description of the random sampling and Reid et al. 2014 for the relationships between distance, mass, and inclination.

This script starts from a middle distance of 8.6 kpc (Reid et al. 2014) and then branches out to lower and higher distances in order to fit them in XSPEC without causing fit errors (too big of a jump in parameter space will cause the XSPEC fit to not find a suitable fit without manual oversight).

RXTE Observation IDs (see Middleton et al. 2006, Table 2, labeled as (a), (b), and (c):
(a) 20402-01-45-03
(b) 10408-01-10-00
(c) 10408-01-38-00

Output files:
fits_best.txt - contains the best fit parameters for a given distance, inclination, and mass. Columns are: Mass (solar masses), distance (kpc), Luminosity of 10408-01-38-00 in L_Edd units, inclination (degrees), spin (a_* dimensionless parameter), and reduced chi for each XSPEC fit.


'''

import os
import numpy as np
import batch
from numpy.random import *
import argparse


def fit_data_set(outfile,params):

    
    ## Initial best-fit values for 8.6 kpc ~median
    L1=-0.328305
    L2=-0.0589132
    L3=-0.519325
    spin=0.862437
    smedgeE1=6.9
    smedgetau1=0.0
    simplgam1=4.19097
    simplfrac1=0.331872
    gaussE1=7.0
    gaussnorm1=0.0
    smedgeE2=9.0
    smedgetau2=0.0
    simplgam2=4.500
    simplfrac2=0.190056
    gaussE2=7.0
    gaussnorm2=0.0219774
    smedgeE3=6.9
    smedgetau3=0.0
    simplgam3=3.53311
    simplfrac3=0.220867
    gaussE3=7.0
    gaussnorm3=0.0

    mu_a = 0.
    mu_r = 0.
    npar = params[:,0].size
    outarr = np.zeros([npar,6])

    ## Fit the sorted params
    for i in range(npar):
        dist = params[i,0]
        mu = params[i,1]
        mass_log = params[i,2]
        print(dist,mu,mass_log)

        ## call the batch fitting routine
        chi, spin, mass_out, norm, L1, L2, L3, smedgeE1, smedgetau1, simplgam1, simplfrac1, gaussE1, gaussnorm1, smedgeE2, smedgetau2, simplgam2, simplfrac2, gaussE2, gaussnorm2, smedgeE3, smedgetau3, simplgam3, simplfrac3, gaussE3, gaussnorm3 = batch.fit_par(dist,mu,mass_log,L1,L2,L3,spin, smedgeE1, smedgetau1, simplgam1, simplfrac1, gaussE1, gaussnorm1, smedgeE2, smedgetau2, simplgam2, simplfrac2, gaussE2, gaussnorm2, smedgeE3, smedgetau3, simplgam3, simplfrac3, gaussE3, gaussnorm3)

        ## Redefine mass, dist, inc
        mass = 10.**mass_out
        dist = 10./np.sqrt(norm)
        inc_deg = (180./np.pi)*np.arccos(mu)

        ## Storing parameter values in an outfile
        outarr[i,0] = mass
        outarr[i,1] = dist
        outarr[i,2] = L2
        outarr[i,3] = inc_deg
        outarr[i,4] = spin
        outarr[i,5] = chi

        ## update starting parameters for next fit
        L1 = L1
        L2 = L2
        L3 = L3
        spin = spin
        smedgeE1=smedgeE1
        smedgetau1=smedgetau1
        simplgam1=simplgam1
        simplfrac1=simplfrac1
        gaussE1=gaussE1
        gaussnorm1=gaussnorm1
        smedgeE2=smedgeE2
        smedgetau2=smedgetau2
        simplgam2=simplgam2
        simplfrac2=simplfrac2
        gaussE2=gaussE2
        gaussnorm2=gaussnorm2
        smedgeE3=smedgeE3
        smedgetau3=smedgetau3
        simplgam3=simplgam3
        simplfrac3= simplfrac3
        gaussE3=gaussE3
        gaussnorm3=gaussnorm3

    return outarr

def sortfit(params,par,par_cut):
    """ Sorting parameters in order by distance """

    ## Sort the parameters
    params = params[params[:,par].argsort(),]

    ## Split the parameter array into 2 pieces, at the middle
    plow = params[np.where(params[:,par] < par_cut),:][0]
    plow = np.flip(plow,axis=0)
    phigh = params[np.where(params[:,par] >=  par_cut),:][0]

    ## create filenames for saving parameter information
    f_low = "fits_"+str(par)+"_low.txt"
    f_high = "fits_"+str(par)+"_high.txt"

    outlow = fit_data_set(f_low,plow)
    outhigh = fit_data_set(f_high,phigh)
    comb = np.concatenate((outlow,outhigh))
    ## alway return an array sorted by distance
    return comb[comb[:,1].argsort(),]


def draw_params():
    """ Draw parameters from a random distribution centered on the Reid et al. 2014 best-fit values."""

    inc_center = 0.50  ## cos(60) = 0.5
    inc_width = 0.078  ## +/- 5 degrees from 60 degrees
    inc_min = 0.30
    inc_max = 0.70

    mass_center = 1.09  ## log(12.4 solar masses) - new mass
    mass_width = 0.06  ## +/- 2.0, but in log space
    mass_min = 1.0091
    mass_max = 1.2556

    dist_center = 8.6 #kpc
    dist_width = 2.0 #kpc
    dist_min = 5.9
    dist_max = 12.0

    mu_a_center = 1.324e-12 #rad/s (23.6 mas/day)
    mu_a_width = 2.81e-14 # rad/s (0.5 mas/day)

    mu_r_center = 5.611e-13 #rad/s (10 mas/day)
    mu_r_width = 2.81e-14 # rad/s (0.5 mas/day)
    mass_function = 1.602e34 #grams (8.054 solar masses - see readme.txt)

    c = 2.99e10 #cm/s
    #mu_a = 1.324e-12 #rad/s (23.6 milli arcseconds/day)
    #mu_r = 5.611e-13 #rad/s (10 milli arcseconds/day)

    while True:
        dist_rand = np.random.normal(dist_center,dist_width)
        if dist_rand > dist_min and dist_rand < dist_max:
            dist = dist_rand
            break

    while True:
        mu_a = np.random.normal(mu_a_center, mu_a_width)
        mu_r = np.random.normal(mu_r_center, mu_r_width)
 
        #print('mu_a ===== ', mu_a)
        #print('mu_r ===== ', mu_r)

        ### Convert to cgs units
        dist_cgs = dist*3.086e21 #cm (1 kpc = 3.086e21 cm)
        inc_cgs = np.arctan(2*dist_cgs*mu_a*mu_r/(c*(mu_a-mu_r))) #radians
        inc_degrees = (180/np.pi)*inc_cgs
        mass_cgs = mass_function/(np.sin(inc_cgs)**3) #grams

        ### Converting from cgs units into solar masses, kpc, mu.
        dist = dist_cgs / 3.086e21 #kpc
        mu = np.cos(inc_cgs) 
        mass = mass_cgs / 1.99e33 #solar masses
        mass_log = np.log10(mass) #log(solar masses)

        if ((mu <= inc_max and mu >=  inc_min) and (mass_log <= mass_max and mass_log >= mass_min)):
            break

    return dist,mu,mass_log


# Main Function
def main(**kwargs):

    n = kwargs['nsamples']
    params = np.zeros([n,3]) ## create array of zeros for the 3 parameters
    
    i=0
    while i < n:
        params[i,:] = draw_params()
        i += 1

    distcut = sortfit(params,0,8.6)
    mucut = sortfit(params,1,0.5)
    masscut = sortfit(params,2,np.log10(12.4))
    
    n = params[:,0].size
    outarr = np.zeros([n,6])
    for i in range(n):
        if (distcut[i,5] <= mucut[i,5]):
            if (distcut[i,5] <= masscut[i,5]):
                outarr[i,:] = distcut[i,:]
            else:
                outarr[i,:] = masscut[i,:]
        else:
            if (mucut[i,5] <= masscut[i,5]):
                outarr[i,:] = mucut[i,:]
            else:
                outarr[i,:] = masscut[i,:]


    np.savetxt("fits_best.txt",outarr)


# Execute main function
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--nsamples',
                      type = int,
                      default = 10,
      help='Number of samples to generate for sampling distances, inclinations, and masses.')
  args = parser.parse_args()
  main(**vars(args))

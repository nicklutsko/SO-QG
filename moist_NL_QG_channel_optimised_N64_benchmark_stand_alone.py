"""
01/10/2021 added moisture flag, made it faster to run

09/26/2019 Model is stable!!

07/20/2010 Added LH03-type moisture, including omega-equation


"""
import numpy as np
import matplotlib.pylab as plt

from random import random
from numba import jit
import time

# from mpi4py_fft import fftw
import scipy.fft as sci_fft
#%matplotlib inline

#import os
#exec(open(os.environ['PYTHONSTARTUP']).read())
#exec(open(STARTUP_2020_moist_two_layer).read())
#xr.set_options(display_width=80, display_style='text')
#save_path =  mconfig['paths']['work']+'/experiments/optimzed_QG_N128_benchmark_scipy_workers/'
#MT.mkdirs_r(save_path)


#######################################################
#  Declare some parameters, arrays, etc.

opt = 3 # 1 = just the linear parts, 2 = just the nonlinear parts, 3 = full model

N = 64#128 #zonal size of spectral decomposition
N2 = 64#128 #meridional size of spectral decomposition
Lx = 72. #size of x -- stick to multiples of 10
Ly = 96. #size of y -- stick to multiples of 10


nu = pow( 10., -3. ) #viscous dissipation
tau_d = 100. #Newtonian relaxation time-scale for interface
tau_f = 15. #surface friction
beta = 0.196 #beta
sigma = 3.5
U_1 = 1.

g = 0.04 #leapfrog filter coefficient

moist = True
if moist:
	C = 2. #linearized Clausius-Clapeyron parameter
	L = 0.#3 #non-dimensional measure of the strength of latent heating
	E = 1. #Evaporation rate
	Gamma = beta - (1. + C * L) / (1. - L)
else:
	L=0.
	C =0.

count = 0 #for saving
d1 = 0 #for saving

x = np.linspace( -Lx / 2, Lx / 2, N )
y = np.linspace( -Ly / 2, Ly / 2, N2 )
dx= min([ np.diff(x)[0], np.diff(y)[0] ])#*110/700 # non-dimensional dx

#Wavenumbers:
kk = np.fft.rfftfreq( N, Lx / float(N) / 2. / np.pi ) #zonal wavenumbers
ll = np.fft.fftfreq( N2, Ly / float(N2) / 2. / np.pi ) #meridional wavenumbers

Lapl = -(np.expand_dims(ll, 1) ** 2 + np.expand_dims(kk, 0) ** 2)

tot_time = 1000  #00 #Length of run
dt 		 = 0.025 #Timestep
ts 		 = int(tot_time / dt ) #Total timesteps
lim  	 = int(300 // dt ) #int(ts / 4 ) #Start saving
st = int( 1 / dt ) #How often to save ../../data/moist_qg

fft = sci_fft # np.fft
nworker =10

#######################################################
#  Declare arrays

#Spectral arrays, only need 3 time-steps
psic_1 = np.zeros( ( ( 3 , N2 , N // 2 + 1 ) ) ).astype( complex )
psic_2 = np.zeros( ( ( 3 , N2 , N // 2 + 1 ) ) ).astype( complex )
qc_1 = np.zeros( ( ( 3 , N2, N // 2 + 1 ) ) ).astype( complex )
qc_2 = np.zeros( ( ( 3 , N2 , N // 2 + 1 ) ) ).astype( complex )
vorc_1 = np.zeros( ( ( 3 , N2, N // 2 + 1  ) ) ).astype( complex )
vorc_2 = np.zeros( ( ( 3 , N2 , N // 2 + 1 ) ) ).astype( complex )

print('comples spectral array shapes: ' + str(psic_1.shape))

# moisture variables
if moist:
	mc = np.zeros( ( ( 3, N2, N // 2 + 1 ) ) ).astype( complex ) #moisture
	M = np.zeros( 3 ) #Total moisture


#Real arrays, only need 3 time-steps
psi_1 = np.zeros( ( ( 3 , N2 , N ) ) )
psi_2 = np.zeros( ( ( 3 , N2 , N ) ) )
q_1 = np.zeros( ( ( 3 , N2, N ) ) )
q_2 = np.zeros( ( ( 3 , N2 , N ) ) )

print('real array shapes: ' + str(psi_1.shape))

#For saving:
u = np.zeros( ( ( int((ts - lim) / st) + 1, 2, N2 ) ) )
v = np.zeros( ( ( int((ts - lim) / st) + 1, 2, N2 ) ) )
tau = np.zeros( ( ( int((ts - lim) / st) + 1, 2, N2 ) ) )
precip = np.zeros( ( ( int((ts - lim) / st) + 1, 2, N2 ) ) )

CFL_store = np.zeros( ( ( int((ts - lim) / st) + 1 ) ) )
# z_emf = np.zeros( ( ( int((ts - lim) / st) + 1, 2, N2 ) ) )
# z_ehf = np.zeros( ( ( int((ts - lim) / st) + 1, 2, N2 ) ) )
# z_eke = np.zeros( ( ( int((ts - lim) / st) + 1, 2, N2 ) ) )

#######################################################
#  Define equilibrium interface height + sponge

sponge = np.zeros( N2)
u_eq = np.zeros( N2)

for i in range( N2 ):
	y1 = float( i - N2 /2) * (y[1] - y[0] )
	y2 = float(min(i, N2 -i - 1)) * (y[1] - y[0] )
	sponge[i] = U_1 / (np.cosh(abs(y2/sigma)))**2
	u_eq[i] = U_1 * ( 1. / (np.cosh(abs(y1/sigma)))**2 - 1. / (np.cosh(abs(y2/sigma)))**2  )

psi_Rc = -fft.fft(  u_eq ) / 1.j / ll
psi_Rc[0] = 0.
psi_R = fft.ifft(psi_Rc )

#plt.plot(psi_R)
#######################################################
# %% Spectral functions

@jit(nopython=True)
def ptq(l,k, ps1, ps2):

	"""
	Calculate PV
	in:
	meridional wavemnumber l, zonal wavenumber k, psi1(l,k), psi2(l,k)
	"""
	q1 = -(np.expand_dims(l, 1) ** 2 + np.expand_dims(k, 0) ** 2 ) * ps1 - (ps1 - ps2) # -(k^2 + l^2) * psi_1 -0.5*(psi_1-psi_2)
	q2 = -(np.expand_dims(l, 1) ** 2 + np.expand_dims(k, 0) ** 2 ) * ps2 + (ps1 - ps2) # -(k^2 + l^2) * psi_2 +0.5*(psi_1-psi_2)
	return q1, q2


@jit(nopython=True)
def qtp(kk, ll, q1_s, q2_s):
	"""Invert PV"""
	divider =  ( np.expand_dims(ll, 1) ** 2 + np.expand_dims(kk, 0) ** 2)  # (psi_1 + psi_2)/2
	divider[0, 0] = np.nan
	psi_bt = -(q1_s + q2_s) / divider / 2.0  # (psi_1 + psi_2)/2
	psi_bt[0, 0] = 0.

	psi_bc = -(q1_s - q2_s) / (np.expand_dims(ll, 1) ** 2 + np.expand_dims(kk, 0) ** 2 + 2. ) / 2.  # (psi_1 - psi_2)/2

	psi1 = psi_bt + psi_bc
	psi2 = psi_bt - psi_bc

	return psi1, psi2


@jit(nopython=True)
def qtp_3d(kk, ll, q1_s, q2_s):
	"""Invert PV"""

	divider =  ( np.expand_dims(ll, 1) ** 2 + np.expand_dims(kk, 0) ** 2)  # (psi_1 + psi_2)/2
	divider[0, 0] = np.nan
	psi_bt = -(q1_s + q2_s) / divider  /2.0 # (psi_1 + psi_2)/2
	psi_bt[:, 0, 0] = 0.

	psi_bc = -(q1_s - q2_s) / (np.expand_dims(ll, 1) ** 2 + np.expand_dims(kk, 0) ** 2 + 2. ) / 2.  # (psi_1 - psi_2)/2

	psi1 = psi_bt + psi_bc
	psi2 = psi_bt - psi_bc

	return psi1, psi2


@jit(nopython=True, parallel = True)
def exponential_cutoff( data, a, s, kcut ):
    d1, d2 = np.shape( data )
    F = np.ones( ( d1, d2 ) )
    for i in range( d1 ):
        for j in range( d2 ):
            if i > 9 and i <= d1 / 2:
                F[i, j] *= np.exp( -a * ((float(i - kcut)/float(d1 / 2 - 1 - kcut) )) ** s )
            elif i > d1 / 2 and i < (d1 - 10 ):
                k = d1 - i
                F[i, j] *= np.exp( -a * ((float(k - kcut)/float(d1 / 2 - 1 - kcut) )) ** s )
            if j > 9:
                F[i, j] *= np.exp( -a * ((float(j - kcut)/float(d2 - 1 - kcut) )) ** s )
    return F

def sdat(c, F):
    print("Saving in:", F)
    np.savez(F,u = c)
    return 0


# def qtp(kk, ll, q1_s, q2_s):
# 	"""Invert PV"""
# 	psi_bt = -(q1_s + q2_s) / (ll[:, np.newaxis] ** 2 + kk[np.newaxis, :] ** 2) / 2.  # (psi_1 + psi_2)/2
# 	psi_bc = -(q1_s - q2_s) / (ll[:, np.newaxis] ** 2 + kk[np.newaxis, :] ** 2 + 2. ) / 2.  # (psi_1 - psi_2)/2
# 	psi_bt[0, 0] = 0.
# 	psi1 = psi_bt + psi_bc
# 	psi2 = psi_bt - psi_bc
#
# 	return psi1, psi2
#


# %%
# not used in this code
# def moist_PV(ps1, ps2, m, L):
#     """Calculate moist PV"""
#     q2 = Lapl * ps2 + (ps1 - ps2 + L * m) / (1. - L) # -(k^2 + l^2) * psi_2 + (psi_1-psi_2 + Lm) / (1 - L)
#     return q2
#

#

# %%

#ll_ax1= np.expand_dims(ll, 1)
#kk_ax0= np.expand_dims(kk, 0)
@jit(nopython=True)
def div( field, kk, ll ): # former grad()

	d1, d2 = np.shape( field )
	div = np.zeros( ( d1, d2 ) ) + 0.j
	div[:, :] = 1.j * np.expand_dims(ll, 1) + 1.j * np.expand_dims(kk, 0)

	return div * field

@jit(nopython=True)
def Laplace( field, kk, ll ):

	d1, d2 = np.shape( field )
	Lapl = np.zeros( ( d1, d2 ) )
	Lapl[:, :] = -(np.expand_dims(ll, 1) ** 2 + np.expand_dims(kk, 0) ** 2)

	return Lapl * field

#######################################################
# %% Initial conditions:

psic_1[0] = [ [ random() for i in range(N // 2 + 1 ) ] for j in range(N2) ] # wn space
psic_2[0] = [ [ random() for i in range(N // 2 + 1 ) ] for j in range(N2) ]

#Transfer values:
psic_1[ 1 , : , : ] = psic_1[ 0 , : , : ]
psic_2[ 1 , : , : ] = psic_2[ 0 , : , : ]

#Calculate initial PV
# %%

for i in range( 2 ):
	vorc_1[i], vorc_2[i] = ptq(ll, kk, psic_1[i], psic_2[i]) # in and outputs in wave number space

q_1[0:2, :,: ] = fft.irfft2( vorc_1[0:2,:,:], workers=nworker ) + beta * y[:, np.newaxis] # to real
q_2[0:2, :,: ] = fft.irfft2( vorc_2[0:2,:,:], workers=nworker ) + beta * y[:, np.newaxis] # to real

qc_1[0:2,:,:] = fft.rfft2( q_1[0:2,:,:] , workers=nworker )  # to imag
qc_2[0:2,:,:] = fft.rfft2( q_2[0:2,:,:] , workers=nworker )  # to imag



###########################
psi1 = fft.irfft2( psic_1[1] , workers=nworker )
psi2 = fft.irfft2( psic_2[1] , workers=nworker )

if moist:
	#Start at uniform 50% saturation
	tot_m = C * (psi1 - psi2) / 2.
	M[0] = np.mean( np.mean( tot_m ) ) / (1. + C * L)
	M[1] = M[0]

	m = tot_m - M[0] * (1. + C * L)
	mc[0] = fft.rfft2( m.real , workers=nworker )
	mc[1] = fft.rfft2( m.real , workers=nworker )


#######################################################
#  Time-stepping functions

# %%
@jit(nopython=True)
def reshuffle_helper1(psi, qc):

	N2, N = np.shape( psi )
	ex = int(N *  3 / 2)# - 1
	ex2 = int(N2 * 3 / 2)# - 1
	temp1 = np.zeros( ( ex2, ex ) ) + 0.j
	temp2 = np.zeros( ( ex2, ex ) ) + 0.j
	temp4 = np.zeros( ( N2, N ) ) + 0.j	#Final array

	#Pad values:
	temp1[:N2//2, :N] = psi[:N2//2, :N]
	temp1[ex2-N2//2:, :N] = psi[N2//2:, :N]

	temp2[:N2//2, :N] = qc[:N2//2, :N]
	temp2[ex2-N2//2:, :N] = qc[N2//2:, :N]
	return temp1, temp2, temp4, N2, N, ex2

@jit(nopython=True)
def reshuffle_helper2(temp3, temp4, N2, N, ex2 ):

	temp4[:N2//2, :N] = temp3[:N2//2, :N]
	temp4[N2//2:, :N] = temp3[ex2-N2//2:, :N]
	return temp4

def calc_nl( psi, qc ):
	""""Calculate non-linear terms, with Orszag 3/2 de-aliasing"""

	# N2, N = np.shape( psi )
	# ex = int(N *  3 / 2)# - 1
	# ex2 = int(N2 * 3 / 2)# - 1
	# temp1 = np.zeros( ( ex2, ex ) ) + 0.j
	# temp2 = np.zeros( ( ex2, ex ) ) + 0.j
	# temp4 = np.zeros( ( N2, N ) ) + 0.j	#Final array
	#
	# #Pad values:
	# temp1[:N2//2, :N] = psi[:N2//2, :N]
	# temp1[ex2-N2//2:, :N] = psi[N2//2:, :N]
	#
	# temp2[:N2//2, :N] = qc[:N2//2, :N]
	# temp2[ex2-N2//2:, :N] = qc[N2//2:, :N]

	temp1, temp2, temp4 , N2, N, ex2 = reshuffle_helper1(psi, qc)
	#Fourier transform product, normalize, and filter:
	temp3 = fft.rfft2( fft.irfft2( temp1 , workers=nworker ) * fft.irfft2( temp2, workers=nworker ) , workers=nworker ) * 9. / 4.
	# temp4[:N2//2, :N] = temp3[:N2//2, :N]
	# temp4[N2//2:, :N] = temp3[ex2-N2//2:, :N]

	return reshuffle_helper2(temp3, temp4, N2, N, ex2)

@jit(nopython=True)
def jacobian_prep(kk, ll, psi, qc):
	#kk, ll, psi, qc = kk, ll, psic_1[1, :, :], vorc_1[1, :, :]

	kk2, ll2 =np.expand_dims(kk, 0), np.expand_dims(ll, 1)
	dpsi_dx = 1.j * kk2 * psi
	dpsi_dy = 1.j * ll2 * psi

	dq_dx = 1.j * kk2 * qc
	dq_dy = 1.j * ll2 * qc
	return dpsi_dx, dq_dy, dpsi_dy ,dq_dx

def nlterm(kk, ll, psi, qc):
	""""Calculate Jacobian"""
	#kk, ll, psi, qc = kk, ll, psic_1[1, :, :], vorc_1[1, :, :]

	# kk2, ll2 =np.expand_dims(kk, 0), np.expand_dims(ll, 1)
	# dpsi_dx = 1.j * kk2 * psi
	# dpsi_dy = 1.j * ll2 * psi
	#
	# dq_dx = 1.j * kk2 * qc
	# dq_dy = 1.j * ll2 * qc

	dpsi_dx, dq_dy, dpsi_dy ,dq_dx = jacobian_prep(kk, ll, psi, qc)  ######### here was an error
	#calc_nl( dpsi_dx, dq_dy ) - calc_nl( dpsi_dy, dq_dx )

	return  calc_nl( dpsi_dx, dq_dy ) - calc_nl( dpsi_dy, dq_dx )


@jit(nopython=True)
def fs(ovar, rhs, det, nu, kk, ll):
    """Forward Step: q^t-1 / ( 1 + 2. dt * nu * (k^4 + l^4 ) ) + RHS"""
    mult = det / ( 1. + det * nu * (np.expand_dims(kk, 0) ** 4 + np.expand_dims(ll, 1) ** 4) )

    return mult * (ovar / det + rhs)


@jit(nopython=True)
def lf(oovar, rhs, det, nu, kk, ll):
    """Leap frog timestepping: q^t-2 / ( 1 + 2. * dt * nu * (k^4 + l^4 ) ) + RHS"""
    mult = 2. * det / ( 1. + 2. * det * nu * (np.expand_dims(kk, 0) ** 4 + np.expand_dims(ll, 1) ** 4) )
    return mult * (oovar / det / 2. + rhs)

@jit(nopython=True)
def filt(var, ovar, nvar, g):
	"""Leapfrog filtering"""
	return var + g * (ovar - 2. * var + nvar )



# start = time.time()
# qc_1[1, :] = filt( qc_1[1, :], qc_1[0, :], qc_1[2, :], g)
# end = time.time()
# print("Elapsed (with compilation) = %s" % (end - start))
#
# # NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
# start = time.time()
# qc_1[1, :] = filt( qc_1[1, :], qc_1[0, :], qc_1[2, :], g)
# end = time.time()
# print("Elapsed (after compilation) = %s" % (end - start))


# %%
#######################################################
#  Main time-stepping loop

print("Timestep:", 0)

forc1 = np.zeros( ( N2, N ) )
forc2 = np.zeros( ( N2, N ) )
cforc1 = np.zeros( ( N2, N // 2 + 1 ) ).astype(complex)
cforc2 = np.zeros( ( N2, N // 2 + 1  ) ).astype(complex)

nl1 = np.zeros( ( N2, N // 2 + 1  ) ).astype(complex)
nl2 = np.zeros( ( N2, N // 2 + 1 ) ).astype(complex)

#mforc = np.zeros( ( N2, N // 2 + 1 ) ).astype(complex)

if moist:
	mnl = np.zeros( ( N2, N // 2 + 1 ) ).astype(complex)

F = exponential_cutoff( psic_1[0], np.log(1. + 400. * np.pi / float(N) ), 6, 7 )

norm = np.zeros( int(ts / 10) )
norm[0] = np.linalg.norm( psic_1[1] + psic_2[1] )

u = np.zeros( ( ( int((ts - lim) / 10), 2, N2 ) ) )
#v = np.zeros( ( ( ( int((ts - lim) / 10), 2, N2, N ) ) ) )
#z_emfs = np.zeros( ( ( 1000, 2, N2 ) ) )

if moist:
	tot_P = np.zeros( ( N2, N  ) )
	eddy_P = np.zeros( ( N2, N ) )
	eddy_Pc = np.zeros( ( N2, N // 2 + 1 ) ).astype(complex)

	tot_E = np.zeros( ( N2, N  ) )
	eddy_E = np.zeros( ( N2, N ) )
	eddy_Ec = np.zeros( ( N2, N // 2 + 1 ) ).astype(complex)



#Timestepping:
#i=1

for i in range( 1, ts ):
	start = time.time()

	#x 12ms
	#timeit
	# if i % 100 == 0:
	# 	print("Timestep:", i, ' / ', ts)
	# 	# plt.contourf(u[:, 0] )
	# 	# plt.show()


	if opt > 1:
		#NL terms -J(psi, qc) - beta * v
		nl1[:, :] = -nlterm( kk, ll, psic_1[1, :, :], vorc_1[1, :, :]) - beta * 1.j * np.expand_dims(kk, 0) * psic_1[1, :, :]
		nl2[:, :] = -nlterm( kk, ll, psic_2[1, :, :], vorc_2[1, :, :]) - beta * 1.j * np.expand_dims(kk, 0) * psic_2[1, :, :]

		if moist:
			mnl[:, :] = -nlterm( kk, ll, psic_2[1, :, :], mc[1, :, :])

	if opt != 2:
		#Linear terms
		#Relax interface
		forc1[:, :] = (psi_1[1] - psi_2[1] - psi_R[:, np.newaxis]) / tau_d
		forc2[:, :] = -(psi_1[1] - psi_2[1] - psi_R[:, np.newaxis]) / tau_d

		#Sponge
		forc1[:, :] -= sponge[:, np.newaxis] * (q_1[1] - np.mean( q_1[1], axis = 1)[:, np.newaxis] )
		forc2[:, :] -= sponge[:, np.newaxis] * (q_2[1] - np.mean( q_2[1], axis = 1)[:, np.newaxis] )

		#Convert to spectral space + add friction
		cforc1 = fft.rfft2( forc1  , workers =nworker)
		cforc2 = fft.rfft2( forc2  , workers =nworker) + ( np.expand_dims(kk, 0) ** 2  + np.expand_dims(ll, 1) ** 2 ) * psic_2[1] / tau_f

	#x 450nus
	#timeit
	rhs1 = nl1[:] + cforc1[:]
	rhs2 = nl2[:] + cforc2[:]
	if moist:
		mrhs = mnl[:]

	if i == 1:
		#Forward step
		qc_1[2, :] = fs(qc_1[1, :, :], rhs1[:], dt, nu, kk, ll)
		qc_2[2, :] = fs(qc_2[1, :, :], rhs2[:], dt, nu, kk, ll)
		if moist:
			mc[2, :] = fs(mc[1, :, :], mrhs[:], dt, nu, kk, ll)
	else:
		#Leapfrog step
		qc_1[2, :, :] = lf(qc_1[0, :, :], rhs1[:], dt, nu, kk, ll)
		qc_2[2, :, :] = lf(qc_2[0, :, :], rhs2[:], dt, nu, kk, ll)
		if moist:
			mc[2, :] = lf(mc[0, :, :], mrhs[:], dt, nu, kk, ll)

	#x 1.3ms
	#timeit
	q_1[1] = fft.irfft2( qc_1[2]  , workers =nworker)
	q_2[1] = fft.irfft2( qc_2[2]  , workers =nworker)

	#Subtract off beta and invert
	vorc_1[1] = fft.rfft2( q_1[1] - beta * y[:, np.newaxis] , workers =nworker)
	vorc_2[1] = fft.rfft2( q_2[1] - beta * y[:, np.newaxis] , workers =nworker)
	psic_1[1], psic_2[1] = qtp(kk,ll, vorc_1[1], vorc_2[1] )
	psi_1[1] = fft.irfft2( psic_1[1]  , workers =nworker)
	psi_2[1] = fft.irfft2( psic_2[1]  , workers =nworker)


	#############################################
	#Now calculate precip, then adjust fields
	#x 1.5 ms
	#timeit

	#Convert to real space
	if moist:
		m = fft.irfft2(mc[2] , workers =nworker)

		u2 = fft.irfft2( -1.j * np.expand_dims(ll, 1) * psic_2[1]  , workers =nworker)
		v2 = fft.irfft2( 1.j * np.expand_dims(kk, 0) * psic_2[1]  , workers =nworker)

		#Calculate precip
		sat_def = (C * (psi_1[1, :,:] - psi_2[1, :, :]) ) - ((1. + C * L) * M[1] + m)

		# E and P are here Evaporationa and precipition, not Potential/Kinetic energies or so ...
		sat_def_mask=sat_def < 0.
		tot_P = np.where(sat_def_mask, -sat_def , 0)
		tot_E = np.where(~sat_def_mask, E * np.sqrt(u2 ** 2 + v2 ** 2) * sat_def  , 0)


		mean_P = np.mean( tot_P)
		mean_E = np.mean( tot_E)

		eddy_P = tot_P - mean_P
		eddy_E = tot_E - mean_E

		eddy_Pc = fft.rfft2( eddy_P  , workers =nworker) * F #F is exponential cut-off filter
		eddy_Ec = fft.rfft2( eddy_P  , workers =nworker) * F #F is exponential cut-off filter

		#Adjust fields, time-step
		rhs1 -= L * eddy_Pc
		rhs2 += L * eddy_Pc
		mrhs -= eddy_Pc + eddy_Ec

		if i == 1:
			#Forward step
			qc_1[2, :] = fs(qc_1[1, :, :], rhs1[:], dt, nu, kk, ll)
			qc_2[2, :] = fs(qc_2[1, :, :], rhs2[:], dt, nu, kk, ll)
			if moist:
				mc[2, :] = fs(mc[1, :, :], mrhs[:], dt, nu, kk, ll)
				M[2] = M[1] + dt * (mean_E - mean_P)
		else:
			#Leapfrog step
			qc_1[2, :] = lf(qc_1[0, :, :], rhs1[:], dt, nu, kk, ll)
			qc_2[2, :] = lf(qc_2[0, :, :], rhs2[:], dt, nu, kk, ll)
			if moist:
				mc[2, :] = lf(mc[0, :, :], mrhs[:], dt, nu, kk, ll)
				M[2] = M[0] + 2. * dt * (mean_E - mean_P)

		u1 = -1.j * np.expand_dims(ll, 1) * psic_1[1, :, :]
		u2 = -1.j * np.expand_dims(ll, 1) * psic_2[1, :, :]

		div_u = div( (u1 + u2) / 2., kk, ll )
		div_nu = div( psic_1[1] - psic_2[1], kk, ll )

		div_ageo = calc_nl( div_u, div_nu )
		div_ageo = -2. * div( div_ageo, kk, ll ) + Laplace( L * eddy_Pc, kk, ll ) + Laplace( (psic_1[1] - psic_2[1] - psi_Rc[:, np.newaxis]) / tau_d, kk, ll )

		div_ageo /= (np.expand_dims(ll, 1) ** 2 + np.expand_dims(kk, 0) ** 2 + 2)

		#x
		mrhs -= div_ageo
		#x 3ms
		#timeit

		if i == 1:
			#Forward step
			mc[2, :] = fs(mc[1, :, :], mrhs[:], dt, nu, kk, ll)
		else:
			#Leapfrog step
			mc[2, :] = fs(mc[0, :, :], mrhs[:], dt, nu, kk, ll)

		if i > 1:
			mc[1, :] = filt( mc[1, :], mc[0, :], mc[2, :], g)
			M[1] = filt( M[1], M[0], M[2], g)

	if i > 1:
		#Leapfrog filter
		qc_1[1, :] = filt( qc_1[1, :], qc_1[0, :], qc_1[2, :], g)
		qc_2[1, :] = filt( qc_2[1, :], qc_2[0, :], qc_2[2, :], g)


	q_1[0:2] = fft.irfft2( qc_1[1:]  , workers =nworker)
	q_2[0:2] = fft.irfft2( qc_2[1:]  , workers =nworker)

	#Subtract off beta and invert
	vorc_1[0:2] = fft.rfft2( q_1[0:2] - beta * y[:, np.newaxis] , workers =nworker)
	vorc_2[0:2] = fft.rfft2( q_2[0:2] - beta * y[:, np.newaxis] , workers =nworker)
	psic_1[0:2], psic_2[0:2] = qtp_3d(kk, ll, vorc_1[0:2], vorc_2[0:2] )

	psi_1[0:2] = fft.irfft2( psic_1[0:2] , workers =nworker)
	psi_2[0:2] = fft.irfft2( psic_2[0:2] , workers =nworker)


	#Transfer values:
	qc_1[0:2, :, :] = qc_1[1:, :, :]
	qc_2[0:2, :, :] = qc_2[1:, :, :]

	if moist:
		mc[0:2, :, :] = mc[1:, :, :]
		M[0:2] = M[1:]

	if i > lim:

		if i % st == 0:

			temp_u = fft.irfft2( -1.j * np.expand_dims(ll, 1) * psic_1[1], workers =nworker)
			temp_u2 = fft.irfft2( -1.j * np.expand_dims(ll, 1) * psic_2[1], workers =nworker)
			temp_v = fft.irfft2( 1.j * np.expand_dims(kk, 0) * psic_1[1], workers =nworker)
			temp_v2 = fft.irfft2( 1.j * np.expand_dims(kk, 0) * psic_2[1], workers =nworker)
			temp_tau = fft.irfft2( psic_1[1] - psic_2[1], workers =nworker)
			if abs(L) > 0.:
			   temp_precip = fft.irfft2( Pc , workers =nworker)

			m_u = np.mean( temp_u, axis = 1)
			m_u2 = np.mean( temp_u2, axis = 1)
			m_v = np.mean( temp_v, axis = 1)
			m_v2 = np.mean( temp_v2, axis = 1)
			m_tau = np.mean( temp_tau, axis = 1)
			if abs(L) > 0.:
				m_precip = np.mean( temp_precip, axis = 1)

			e_u = temp_u - m_u[:, np.newaxis]
			e_u2 = temp_u2 - m_u2[:, np.newaxis]
			e_v = temp_v - m_v[:, np.newaxis]
			e_v2 = temp_v2 - m_v2[:, np.newaxis]
			e_tau = temp_tau - m_tau[:, np.newaxis]

			u[d1 + (i - lim) // st, 0] = m_u[:]
			u[d1 + (i - lim) // st, 1] = m_u2[:]
			v[d1 + (i - lim) // st, 0] = m_v[:]
			v[d1 + (i - lim) // st, 1] = m_v2[:]
			tau[d1 + (i - lim) // st, 0] = m_tau[:]
			CFL_store[d1] = CFL
			if abs(L) > 0.:
				precip[d1 + (i - lim) // st, 0] = m_precip[:]

			# z_emf[d1 + (i - lim) // st, 0] = np.mean( e_u * e_v, axis = 1 )
			# z_emf[d1 + (i - lim) // st, 1] = np.mean( e_u2 * e_v2, axis = 1 )
			# z_ehf[d1 + (i - lim) // st, 0] = np.mean( e_v * e_tau, axis = 1 )
			# z_ehf[d1 + (i - lim) // st, 1] = np.mean( e_v2 * e_tau, axis = 1 )
			# z_eke[d1 + (i - lim) // st, 0] = np.mean( e_u ** 2 + e_v ** 2, axis = 1 )
			# z_eke[d1 + (i - lim) // st, 1] = np.mean( e_u2 ** 2 + e_v2 ** 2, axis = 1 )
			#
		if i % int(100 // dt) == 0:
			#Save stuff to make it easier to restart
			#sdat( u[:d1 + (i - lim) // st], save_path+"/zu_N" + str(N2) + "_" + str(L) + "_" + str(C) + "_" + str(U_1) + ".dat.npz" )
			#sdat( v[:d1 + (i - lim) // st], save_path+"/zv_N" + str(N2) + "_" + str(L) + "_" + str(C) + "_" + str(U_1) + ".dat.npz" )
			#sdat( tau[:d1 + (i - lim) // st], save_path+"/ztau_N" + str(N2) + "_" + str(L) + "_" + str(C) + "_" + str(U_1) + ".dat.npz" )
			#if abs(L) > 0.:
			#	sdat( precip[:d1 + (i - lim) // st], save_path+"/zprecip_N" + str(N2) + "_" + str(L) + "_" + str(C) + "_" + str(U_1) + ".dat.npz" )
			#sdat( z_emf[:d1 + (i - lim) // st], save_path+"/zemf_N" + str(N2) + "_" + str(L) + "_" + str(C) + "_" + str(U_1) + ".dat.npz" )
			#sdat( z_ehf[:d1 + (i - lim) // st], save_path+"/zehf_N" + str(N2) + "_" + str(L) + "_" + str(C) + "_" + str(U_1) + ".dat.npz" )
			#sdat( z_eke[:d1 + (i - lim) // st], save_path+"/zeke_N" + str(N2) + "_" + str(L) + "_" + str(C) + "_" + str(U_1) + ".dat.npz" )
			#sdat( psic_1, save_path+"/psic1_N" + str(N2) + "_" + str(L) + "_" + str(C) + "_" + str(U_1) + ".dat.npz" )
			#sdat( qc_1, save_path+"/qc_1N" + str(N2) + "_" + str(L) + "_" + str(C) + "_" + str(U_1) + ".dat.npz" )
			#sdat( psic_2, save_path+"/psic2_N" + str(N2) + "_" + str(L) + "_" + str(C) + "_" + str(U_1) + ".dat.npz" )
			#sdat( qc_2, save_path+"/qc_2_N" + str(N2) + "_" + str(L) + "_" + str(C) + "_" + str(U_1) + ".dat.npz" )
			#if moist:
			#	sdat( mc, save_path+"/mc_N" + str(N2) + "_" + str(L) + "_" + str(C) + "_" + str(U_1) + ".dat.npz" )
			#sdat( count, save_path+"/count_N" + str(N2) + "_" + str(L) + "_" + str(C) + "_" + str(U_1) + ".dat.npz" )
			#sdat( CFL_store[:d1], save_path+"/CFL_N" + str(N2) + "_" + str(L) + "_" + str(C) + "_" + str(U_1) + ".dat.npz" )

			count += 1
			if count == 1001:
				break



	# if i %10 == 0:
	# 	norm[i // 10] = np.linalg.norm( psic_1[1] + psic_2[1] )

	#x
	end = time.time()
	if i % 1000 == 0:
		print("Timestep:", i, " / ", ts)
		temp_u = np.fft.irfft2( -1.j * np.expand_dims(ll, 1) * psic_1[1])
		temp_v = np.fft.irfft2( 1.j * np.expand_dims(kk, 0) * psic_1[1])
		CFL  = np.max([abs(temp_u).max(), abs(temp_v).max()]) * dt / dx
		#CFL  = np.max([abs(v).max(), abs(u).max()]) * dt / dx

		delt = (end - start)
		time_left = delt * float(ts - i)
		print("1 iteration = %s" % delt)
		print("Estimated time left: %0.1f" % time_left)
		print("CFL: %s" %CFL)



print('done')
#def sdat(c, F):
   # print("Saving in:", F)
   # np.savez(F,u = c)
   # return 0

#sdat( u, "data/zu_N" + str(N2) + "_" + str(L) + "_" + str(C) + "_" + str(E) + ".dat")

# %%
%matplotlib inline
plt.subplot(3, 1, 1)
plt.contourf(  u[:, 0] )
plt.colorbar()
plt.subplot(3, 1, 2)
plt.contourf( u[:, 1] )
plt.colorbar()
plt.subplot(3, 1, 3 )
plt.plot( np.mean( u[:, 0], axis = 0) )
plt.plot( np.mean( u[:, 1], axis = 0) )
plt.grid()
plt.show()
#
#
# plt.contourf(np.fft.irfft2(psic_1[0,:,:]))
# plt.colorbar()
# plt.show()
#
# u_snapshot =np.gradient(np.fft.irfft2(psic_1[0,:,:]))
# plt.contourf(-u_snapshot[0])
# plt.colorbar()
# plt.show()
#
# u_snapshot =np.gradient(np.fft.irfft2(psic_2[0,:,:]))
# plt.contourf(-u_snapshot[0])
# plt.colorbar()
# plt.show()

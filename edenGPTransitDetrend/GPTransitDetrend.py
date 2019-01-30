from mpl_toolkits.axes_grid.inset_locator import inset_axes
import exotoolbox
import batman
import seaborn as sns
import argparse
import matplotlib
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
import utils
import os
import dynesty
import time as clocking_time
import contextlib
plt.switch_backend('agg')

parser = argparse.ArgumentParser()
# This reads the lightcurve file. First column is time, second column is flux:
parser.add_argument('-lcfile', default=None)
# This reads the external parameters to fit (assumed to go in the columns):
parser.add_argument('-eparamfile', default=None)
# This defines which of the external parameters you want to use, separated by commas.
# Default is all:
parser.add_argument('-eparamtouse', default='all')
# This reads the external parameters to fit (assumed to go in the columns):
parser.add_argument('-compfile', default=None)
# This defines which comparison stars, if any, you want to use, separated by commas.
# Default is all:
parser.add_argument('-comptouse', default='all')
# This reads an output folder:
parser.add_argument('-ofolder', default='')
# This defines the limb-darkening to be used:
parser.add_argument('-ldlaw', default='quadratic')

# Transit priors. First t0:
parser.add_argument('-t0mean', default=None)
# This reads the standard deviation:
parser.add_argument('-t0sd', default=None)

# Period:
parser.add_argument('-Pmean', default=None)
# This reads the standard deviation:
parser.add_argument('-Psd', default=None)

# Rp/Rs:
parser.add_argument('-pmean', default=None)
# This reads the standard deviation:
parser.add_argument('-psd', default=None)

# a/Rs:
parser.add_argument('-amean', default=None)
# This reads the standard deviation:
parser.add_argument('-asd', default=None)

# Impact parameter:
parser.add_argument('-bmean', default=None)
# This reads the standard deviation:
parser.add_argument('-bsd', default=None)

# ecc:
parser.add_argument('-eccmean', default=None)
# This reads the standard deviation:
parser.add_argument('-eccsd', default=None)

# omega:
parser.add_argument('-omegamean', default=None)
# This reads the standard deviation:
parser.add_argument('-omegasd', default=None)

# Define if it is a circular fit (ecc = 0, omega = 90)
parser.add_argument('--circular', dest='circular', action='store_true')
parser.set_defaults(circular=False)

# Number of live points:
parser.add_argument('-nlive', default=1000)
args = parser.parse_args()

# Is it a circular fit?
circular = args.circular
# Extract lightcurve and external parameters. When importing external parameters, 
# standarize them and save them on the matrix X:
lcfilename = args.lcfile
tall,fall,f_index = np.genfromtxt(lcfilename,unpack=True,usecols=(0,1,2))
# Float the times (batman doesn't like non-float 64):
tall = tall.astype('float64')

idx = np.where(f_index == 0)[0]
t,f = tall[idx],fall[idx]
out_folder = args.ofolder

# find separate nights, here assumed to be more then 0.4 days apart
splitp=np.where(np.ediff1d(t)>0.4)[0] + 1
tnights = len(splitp) + 1 # total number of nights

print('Detected '+str(tnights)+' different nights of data.')

eparamfilename = args.eparamfile
eparams = args.eparamtouse
data = np.genfromtxt(eparamfilename,unpack=True)
for i in range(len(data)):
    x = (data[i] - np.mean(data[i]))/np.sqrt(np.var(data[i]))
    if i == 0:
        X = x
    else:
        X = np.vstack((X,x))
if eparams != 'all':
    idx_params = np.array(eparams.split(',')).astype('int')
    X = X[idx_params,:]

compfilename = args.compfile
if compfilename is not None:
    comps = args.comptouse
    data = np.genfromtxt(compfilename,unpack=True)
    if len(np.shape(data))==1:
        print('\n Only 1 comparison... duplicating \n')
        data = np.array([data,data])
    for i in range(len(data)):
        x = (data[i] - np.mean(data[i]))/np.sqrt(np.var(data[i]))
        if i == 0:
            Xc = x
        else:
            Xc = np.vstack((Xc,x))
    if comps != 'all':
        idx_params = np.array(comps.split(',')).astype('int')
        Xc = Xc[idx_params,:]

nights = {}
for i in range(tnights):
    # put data for separate nights into dictionary for individual access
    # nights[0] is time, nights[1] is flux, nights[2] is eparams, nights[3] is comparisons
    if i == 0:
        nights['n'+str(i)] = np.array([t[:splitp[i]],f[:splitp[i]],X[:,:splitp[i]],Xc[:,:splitp[i]]])
    elif i == tnights - 1:
        nights['n'+str(i)] = np.array([t[splitp[i-1]:],f[splitp[i-1]:],X[:,splitp[i-1]:],Xc[:,splitp[i-1]:]])
    else:
        nights['n'+str(i)] = np.array([t[splitp[i-1]:splitp[i]],f[splitp[i-1]:splitp[i]],X[:,splitp[i-1]:splitp[i]],Xc[:,splitp[i-1]:splitp[i]]])

# Extract limb-darkening law:
ld_law = args.ldlaw

# Transit parameter priors if any:
t0mean = args.t0mean
if t0mean is not None:
    t0mean = np.double(t0mean)
    t0sd = np.double(args.t0sd)

Pmean = args.Pmean
if Pmean is not None:
    Pmean = np.double(Pmean)
    Psd = np.double(args.Psd)

pmean = args.pmean
if pmean is not None:
    pmean = np.double(pmean)
    psd = np.double(args.psd)

amean = args.amean
if amean is not None:
    amean = np.double(amean)
    asd = np.double(args.asd)

bmean = args.bmean
if bmean is not None:
    bmean = np.double(bmean)
    bsd = np.double(args.bsd)

if not circular:
    eccmean = args.eccmean
    omegamean = args.omegamean
    if eccmean is not None:
        eccmean = np.double(args.eccmean)
        eccsd = np.double(args.eccsd)
    if omegamean is not None:
        omegamean = np.double(args.omegamean)
        omegasd = np.double(args.omegasd)
# Other inputs:
n_live_points = int(args.nlive)

# Cook the george kernel:
import george

# Dimensions in number of parameters
# Cook jitter term
jitter = george.modeling.ConstantModel(np.log((200.*1e-6)**2.))

# create dictionaries to store kernels and gp classes for individual nights
kernels = {}
gps = {}

for key in nights:
    kernels[key] = np.var(nights[key][1])*george.kernels.ExpSquaredKernel(np.ones(nights[key][2].shape[0]),ndim=(nights[key][2].shape[0]),axes=range(nights[key][2].shape[0]))
    # Wrap GP object to compute likelihood
    gps[key] = george.GP(kernels[key], mean=0.0,fit_mean=False,white_noise=jitter,fit_white_noise=True)
    gps[key].compute(nights[key][2].T)

# Define transit-related functions:
def reverse_ld_coeffs(ld_law, q1, q2):
    if ld_law == 'quadratic':
        coeff1 = 2.*np.sqrt(q1)*q2
        coeff2 = np.sqrt(q1)*(1.-2.*q2)
    elif ld_law=='squareroot':
        coeff1 = np.sqrt(q1)*(1.-2.*q2)
        coeff2 = 2.*np.sqrt(q1)*q2
    elif ld_law=='logarithmic':
        coeff1 = 1.-np.sqrt(q1)*q2
        coeff2 = 1.-np.sqrt(q1)
    elif ld_law == 'linear':
        return q1,q2
    return coeff1,coeff2

def init_batman(t,law):
    """  
    This function initializes the batman code.
    """
    params = batman.TransitParams()
    params.t0 = 0.
    params.per = 1.
    params.rp = 0.1
    params.a = 15.
    params.inc = 87.
    params.ecc = 0.
    params.w = 90.
    if law == 'linear':
        params.u = [0.5]
    else:
        params.u = [0.1,0.3]
    params.limb_dark = law
    m = batman.TransitModel(params,t)
    return params,m

def get_transit_model(t,t0,P,p,a,inc,q1,q2,ld_law):
    params,m = init_batman(t,law=ld_law)
    coeff1,coeff2 = reverse_ld_coeffs(ld_law, q1, q2)
    params.t0 = t0
    params.per = P
    params.rp = p
    params.a = a
    params.inc = inc
    if ld_law == 'linear':
        params.u = [coeff1]
    else:
                params.u = [coeff1,coeff2]
    return m.light_curve(params)

# Initialize batman:
params,m = init_batman(t,law=ld_law)

# Now define MultiNest priors and log-likelihood:
def prior(cube):
    # Prior on "median flux" is uniform:
    cube[0] = utils.transform_uniform(cube[0],-2.,2.)

    # Pior on the log-jitter term (note this is the log VARIANCE, not sigma); from 0.01 to 100 ppm:
    cube[1] = utils.transform_uniform(cube[1],np.log((0.01e-3)**2),np.log((100e-3)**2))

    # Prior on t0:
    if t0mean is None:
        cube[2] = utils.transform_uniform(cube[2],np.min(t),np.max(t))
    else:
        cube[2] = utils.transform_normal(cube[2],t0mean,t0sd)

    # Prior on Period:
    if Pmean is None:
        cube[3] = utils.transform_loguniform(cube[3],0.1,1000.)
    else:
        cube[3] = utils.transform_normal(cube[3],Pmean,Psd)

    # Prior on planet-to-star radius ratio:
    if pmean is None:
        cube[4] = utils.transform_uniform(cube[4],0,1)
    else:
        cube[4] = utils.transform_truncated_normal(cube[4],pmean,psd)

    # Prior on a/Rs:
    if amean is None:
        cube[5] = utils.transform_uniform(cube[5],0.1,300.)
    else:
        cube[5] = utils.transform_normal(cube[5],amean,asd)

    # Prior on impact parameter:
    if bmean is None:
        cube[6] = utils.transform_uniform(cube[6],0,2.)
    else:
        cube[6] = utils.transform_truncated_normal(cube[6],bmean,bsd,a=0.,b=2.)

    # Prior either on the linear LD or the transformed first two-parameter law LD (q1):
    cube[7] = utils.transform_uniform(cube[7],0,1.)
 
    pcounter = 8
    # (Transformed) limb-darkening coefficient for two-parameter laws (q2):
    if ld_law  != 'linear':
        cube[pcounter] = utils.transform_uniform(cube[pcounter],0,1.)
        pcounter += 1

    if not circular:
        if eccmean is None:
            cube[pcounter] = utils.transform_uniform(cube[pcounter],0,1.)
        else:
            cube[pcounter] = utils.transform_truncated_normal(cube[pcounter],eccmean,eccsd,a=0.,b=1.)
        pcounter += 1
        if omegamean is None:
            cube[pcounter] = utils.transform_uniform(cube[pcounter],0,360.)
        else:
            cube[pcounter] = utils.transform_truncated_normal(cube[pcounter],omegamean,omegasd,a=0.,b=360.)
        pcounter += 1

    # Prior on coefficients of comparison stars:
    if compfilename is not None:
        for key in nights:
            for i in range(nights[key][3].shape[0]):
                cube[pcounter] = utils.transform_uniform(cube[pcounter],-10,10)
                pcounter += 1

    # Prior on kernel maximum variance; from 0.01 to 100 mmag: 
    cube[pcounter] = utils.transform_loguniform(cube[pcounter],(0.01*1e-3)**2,(100*1e-3)**2)
    pcounter = pcounter + 1

    # Now priors on the alphas = 1/lambdas; gamma(1,1) = exponential, same as Gibson+:
    for key in nights:
        for i in range(nights[key][2].shape[0]):
            cube[pcounter] = utils.transform_exponential(cube[pcounter])
            pcounter += 1
    return cube

def loglike(cube):
    # Evaluate the log-likelihood. For this, first extract all inputs:
    mmean, ljitter,t0, P, p, a, b, q1  = cube[0],cube[1],cube[2],cube[3],cube[4],cube[5],cube[6],cube[7]
    pcounter = 8
    if ld_law != 'linear':
        q2 = cube[pcounter]
        coeff1,coeff2 = reverse_ld_coeffs(ld_law,q1,q2)
        params.u = [coeff1,coeff2]
        pcounter += 1
    else:
        params.u = [q1]

    if not circular:
        ecc = cube[pcounter] 
        pcounter += 1
        omega = cube[pcounter]
        pcounter += 1
        ecc_factor = (1. + ecc*np.sin(omega * np.pi/180.))/(1. - ecc**2)
    else:
        ecc = 0.0
        omega = 90.
        ecc_factor = 1.

    inc_inv_factor = (b/a)*ecc_factor
    # Check that b and b/aR are in physically meaningful ranges:
    if b>1.+p or inc_inv_factor >=1.:
        lcmodel = np.ones(len(t))
    else:
        # Compute inclination of the orbit:
        inc = np.arccos(inc_inv_factor)*180./np.pi

        # Evaluate transit model:
        params.t0 = t0
        params.per = P
        params.rp = p
        params.a = a
        params.inc = inc
        params.ecc = ecc
        params.w = omega
        lcmodel = m.light_curve(params)

    model = mmean - 2.51*np.log10(lcmodel)
    models = {}
    lcmodels = {}
    for i in range(tnights):
        # put data for separate nights into dictionary for individual access
        if i == 0:
            models['n'+str(i)] = model[:splitp[i]]
            lcmodels['n'+str(i)] = lcmodel[:splitp[i]]
        elif i == tnights - 1:
            models['n'+str(i)] = model[splitp[i-1]:]
            lcmodels['n'+str(i)] = lcmodel[splitp[i-1]:]
        else:
            models['n'+str(i)] = model[splitp[i-1]:splitp[i]]
            lcmodels['n'+str(i)] = lcmodel[splitp[i-1]:splitp[i]]
    #models = {'n1': model[:splitp], 'n2': model[splitp:]} # split models into different nights
    
    if compfilename is not None:
        for key in nights:
            for i in range(nights[key][3].shape[0]):
                models[key] = models[key] + cube[pcounter]*nights[key][3][i]
                pcounter += 1
    max_var = cube[pcounter]
    pcounter = pcounter + 1
    
    # create dictionaries for separate nights
    alphas = {}
    gp_vector = {}
    residuals = {}
    logl = 0
    
    for key in nights:
        alphas[key] = np.zeros(nights[key][2].shape[0])
        for i in range(nights[key][2].shape[0]):
            alphas[key][i] = cube[pcounter]
            pcounter = pcounter + 1
        gp_vector[key] = np.append(np.append(ljitter,np.log(max_var)),np.log(1./alphas[key]))

        # Evaluate model:
        residuals[key] = nights[key][1] - models[key]

        gps[key].set_parameter_vector(gp_vector[key])
        logl = logl + gps[key].log_likelihood(residuals[key])
    #print(logl)
    return logl 

#              v neparams   v max variance
n_params = 8 + X.shape[0] * tnights + 1 # multiply for number of nights
if compfilename is not None:
    n_params +=  Xc.shape[0] * tnights
if ld_law != 'linear':
    n_params += 1
if not circular:
    n_params += 2

print('Number of external parameters:',X.shape[0]*tnights)
print('Number of comparison stars:',Xc.shape[0]*tnights)
print('Number of counted parameters:',n_params)
out_file = out_folder+'out_multinest_trend_george_'

import pickle
# If not ran already, run Dynesty, save posterior samples and evidences to pickle file:
if not os.path.exists(out_folder+'transit_posteriors_trend_george.pkl'):
    # Run Dynesty:
    from multiprocessing import Pool
    nthreads = 4
    with contextlib.closing(Pool(processes=nthreads-1)) as executor:
        tic = clocking_time.time()
        sampler = dynesty.DynamicNestedSampler(loglike, prior, n_params, bound='single', sample='rwalk', nlive=500, pool=executor, queue_size=nthreads)
        sampler.run_nested()
        toc = clocking_time.time()
        print('\t Sampling took {:1.2f} hours.'.format((toc - tic)/3600))
        # Get output:
        #output = pymultinest.Analyzer(outputfiles_basename=out_file, n_params = n_params)
        results = sampler.results
    
    posterior_samples = results['samples']
    
    # Get out parameters: this matrix has (samples,n_params+1):
    #posterior_samples = output.get_equal_weighted_posterior()[:,:-1]
    # Extract parameters:
    mmean, ljitter,t0, P, p, a, b, q1  = posterior_samples[:,0],posterior_samples[:,1],posterior_samples[:,2],posterior_samples[:,3],\
                                         posterior_samples[:,4],posterior_samples[:,5],posterior_samples[:,6],posterior_samples[:,7]
    a_lnZ = results['logz']
    out = {}
    out['posterior_samples'] = {}
    out['posterior_samples']['unnamed'] = posterior_samples
    out['posterior_samples']['mmean'] = mmean
    out['posterior_samples']['ljitter'] = ljitter
    out['posterior_samples']['t0'] = t0
    out['posterior_samples']['P'] = P
    out['posterior_samples']['p'] = p
    out['posterior_samples']['a'] = a
    out['posterior_samples']['b'] = b
    out['posterior_samples']['q1'] = q1

    pcounter = 8
    if ld_law != 'linear':
        q2 = posterior_samples[:,pcounter]
        out['posterior_samples']['q2'] = q2
        pcounter += 1

    if not circular:
        ecc = posterior_samples[:,pcounter]
        out['posterior_samples']['ecc'] = ecc
        pcounter += 1
        omega = posterior_samples[:,pcounter]
        out['posterior_samples']['omega'] = omega
        pcounter += 1

    xc_coeffs = {}
    if compfilename is not None:
        for key in nights: # splitting comparison and alpha saves into different nights
            out['posterior_samples']['comps_'+key] = {}
            xc_coeffs[key] = []
            for i in range(Xc.shape[0]):
                xc_coeffs[key].append(posterior_samples[:,pcounter])
                out['posterior_samples']['comps_'+key]['xc'+str(i)] = posterior_samples[:,pcounter]
                pcounter += 1
             
    max_var = posterior_samples[:,pcounter]
    out['posterior_samples']['max_var'] = max_var
    pcounter = pcounter + 1
    alphas = {}
    for key in nights:
        out['posterior_samples']['alphas_'+key] = {}
        alphas[key] = []
        for i in range(X.shape[0]):
            alphas[key].append(posterior_samples[:,pcounter])
            out['posterior_samples']['alphas_'+key]['alpha'+str(i)] = posterior_samples[:,pcounter]
            pcounter = pcounter + 1

    out['lnZ'] = a_lnZ
    pickle.dump(out,open(out_folder+'transit_posteriors_trend_george.pkl','wb'))
else:
    out = pickle.load(open(out_folder+'transit_posteriors_trend_george.pkl','rb'))
    posterior_samples = out['posterior_samples']['unnamed']
    
mmean,ljitter = np.median(out['posterior_samples']['mmean']),np.median(out['posterior_samples']['ljitter'])
max_var = np.median(out['posterior_samples']['max_var'])
alphas = {}
gp_vector = {}
for key in nights:
    alphas[key] = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        alphas[key][i] = np.median(out['posterior_samples']['alphas_'+key]['alpha'+str(i)])
    gp_vector[key] = np.append(np.append(ljitter,np.log(max_var)),np.log(1./alphas[key]))


# Evaluate LC:
t0, P, p, a, b, q1 = np.median(out['posterior_samples']['t0']),np.median(out['posterior_samples']['P']),\
                      np.median(out['posterior_samples']['p']),np.median(out['posterior_samples']['a']),np.median(out['posterior_samples']['b']),\
                      np.median(out['posterior_samples']['q1'])


if ld_law != 'linear':
        q2 = np.median(out['posterior_samples']['q1'])
        coeff1,coeff2 = reverse_ld_coeffs(ld_law,q1,q2)
        params.u = [coeff1,coeff2]
else:
        params.u = [q1]

if not circular:
        ecc = np.median(out['posterior_samples']['ecc'])
        omega = np.median(out['posterior_samples']['omega'])
        ecc_factor = (1. + ecc*np.sin(omega * np.pi/180.))/(1. - ecc**2)
else:
        ecc = 0.0
        omega = 90.
        ecc_factor = 1.

inc_inv_factor = (b/a)*ecc_factor
# Check that b and b/aR are in physically meaningful ranges:
if b>1.+p or inc_inv_factor >=1.:
        lcmodel = np.ones(len(t))
else:
        # Compute inclination of the orbit:
        inc = np.arccos(inc_inv_factor)*180./np.pi

        # Evaluate transit model:
        params.t0 = t0
        params.per = P
        params.rp = p
        params.a = a
        params.inc = inc
        params.ecc = ecc
        params.w = omega
        lcmodel = m.light_curve(params)

model = - 2.51*np.log10(lcmodel)
models = {}
lcmodels = {}
for i in range(tnights):
    # put data for separate nights into dictionary for individual access
    if i == 0:
        models['n'+str(i)] = model[:splitp[i]]
        lcmodels['n'+str(i)] = lcmodel[:splitp[i]]
    elif i == tnights - 1:
        models['n'+str(i)] = model[splitp[i-1]:]
        lcmodels['n'+str(i)] = lcmodel[splitp[i-1]:]
    else:
        models['n'+str(i)] = model[splitp[i-1]:splitp[i]]
        lcmodels['n'+str(i)] = lcmodel[splitp[i-1]:splitp[i]]
#models = {'n1': model[:splitp], 'n2': model[splitp:]} # added for comparing each night
comp_models = {}
if compfilename is not None:
    for key in nights:
        comp_model = mmean
        #print(mmean)
        for i in range(Xc.shape[0]):
            comp_model = comp_model + np.median(out['posterior_samples']['comps_'+key]['xc'+str(i)])*nights[key][3][i]#Xc[i,idx]
            comp_models[key] = comp_model # not sure if should be indented

# Evaluate model: 
residuals = {}
for key in nights:
    residuals[key] = nights[key][1] - models[key] - comp_models[key]
    gps[key].set_parameter_vector(gp_vector[key])
#gps['n1'].set_parameter_vector(gp_vector['n1'])
#gps['n2'].set_parameter_vector(gp_vector['n2'])

# Get prediction from GP:
pred_mean = {}
pred_var = {}
pred_std = {}
for key in nights:
    pred_mean[key], pred_var[key] = gps[key].predict(residuals[key], nights[key][2].T, return_var=True)
    pred_std[key] = np.sqrt(pred_var[key])
    
#if compfilename is not None:
#    for i in range(Xc.shape[0]):
#        model = model + cube[pcounter]*Xc[i,:]
#        pcounter += 1

# PLOTTING ---------------------------------------------------------------------------------------

# Plot 1 - Raw LC w/ GP model
    
# REMINDER:
# Nights[key][0] is time, [1] is flux

f, axarr = plt.subplots(tnights, sharey='row')

fout = {}
fout_err = {}
pred_mean_f = {}
plotcount = 0
for key in nights:
    fout[key],fout_err[key] = exotoolbox.utils.mag_to_flux(nights[key][1]-comp_models[key],np.ones(len(nights[key][0]))*np.sqrt(np.exp(ljitter)))
    #print(comp_models[key], nights[key][1])
    axarr[plotcount].errorbar(nights[key][0] - int(nights[key][0][0]),fout[key],yerr=fout_err[key],fmt='.')
    pred_mean_f[key],fout_err[key] = exotoolbox.utils.mag_to_flux(pred_mean[key],np.ones(len(nights[key][0]))*np.sqrt(np.exp(ljitter)))

    axarr[plotcount].plot(nights[key][0] - int(nights[key][0][0]),pred_mean_f[key])
    axarr[plotcount].set_xlabel('Time (BJD - '+str(int(nights[key][0][0]))+')')
    plotcount += 1

f.suptitle('Raw LCs with GP Model')
plt.savefig('raw_lc.png')
#plt.show()

# Plot 2 - Residuals

f, axarr = plt.subplots(tnights, sharey='row')
plotcount = 0
for key in nights:
    nights[key][1] = nights[key][1] - comp_models[key] - pred_mean[key]
    axarr[plotcount].errorbar(nights[key][0] - int(nights[key][0][0]),nights[key][1],yerr=np.ones(len(nights[key][0]))*np.sqrt(np.exp(ljitter)),fmt='.')
    axarr[plotcount].set_xlabel('Time (BJD - '+str(int(nights[key][0][0]))+')')
    plotcount +=1

f.suptitle('Residuals to the GP Model')
plt.savefig('residuals.png')
#plt.show()

# Plot 3 - Detrend LC w/ transit model

f, axarr = plt.subplots(tnights, sharey='row')
plotcount = 0
for key in nights:
    fout[key],fout_err[key] = exotoolbox.utils.mag_to_flux(nights[key][1],np.ones(len(nights[key][0]))*np.sqrt(np.exp(ljitter)))
    axarr[plotcount].errorbar(nights[key][0] - int(nights[key][0][0]),fout[key],yerr=fout_err[key],fmt='.')
    axarr[plotcount].plot(nights[key][0] - int(nights[key][0][0]),lcmodels[key],'b-')
    axarr[plotcount].set_xlabel('Time (BJD - '+str(int(nights[key][0][0]))+')')
    axarr[plotcount].set_ylim(0.98, 1.02)
    plotcount += 1
    
'''
fileout = open('detrended_lc.dat','w')
for i in range(len(tall)):
    fileout.write('{0:.10f} {1:.10f} {2:.10f} {3:.10f}\n'.format(tall[i],fout[i],fout_err[i],lcmodel[i]))
fileout.close()
'''


f.suptitle('Detrended LCs with Transit Model')
f.text(0.04, 0.5, 'Relative Flux', va='center', rotation='vertical')

plt.subplots_adjust(hspace=0.75)

#plt.show()
plt.savefig('detrended_transit_model.png')

print('\nEvidence: ', out['lnZ'])

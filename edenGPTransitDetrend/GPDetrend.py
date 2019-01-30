from mpl_toolkits.axes_grid.inset_locator import inset_axes
import exotoolbox
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
import george

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
# Number of live points:
parser.add_argument('-nlive', default=1000)
args = parser.parse_args()

# Extract lightcurve and external parameters. When importing external parameters, 
# standarize them and save them on the matrix X:
# ------------------------------------------------------------------------------
lcfilename = args.lcfile
# save telescope, target, and date to write in plot titles

tall,fall,f_index = np.genfromtxt(lcfilename,unpack=True,usecols=(0,1,2))
idx = np.where(f_index == 0)[0]
t,f = tall[idx],fall[idx]
out_folder = args.ofolder

# find separate nights, here assumed to be more then 0.4 days apart
splitp=np.where(np.ediff1d(t)>0.4)[0] + 1
tnights = len(splitp) + 1 # total number of nights

print('Detected '+str(tnights)+' different nights of data.')

eparamfilename = args.eparamfile
eparams = args.eparamtouse

# just load twice because can't seem to get names if unpack=True    
data = np.genfromtxt(eparamfilename,dtype=None, names=True, skip_header=0)
anames = np.asarray(data.dtype.names) # store the external parameter(alpha) names
data = np.genfromtxt(eparamfilename,unpack=True,dtype=None)
ebad = [] # will store indices of external parameters that won't work 
for i in range(len(data)):
    if np.var(data[i]) == 0 or None: # find external parameters which will not work
        ebad.append(i)
        print('\nExternal parameter -'+anames[i]+'- has zero variance, will remove automatically \n')
    x = (data[i] - np.mean(data[i]))/np.sqrt(np.var(data[i]))
    if i == 0:
        X = x
    else:
        X = np.vstack((X,x))
# remove bad eparams while keeping those selected by user
if eparams != 'all':
    idx_params = np.array(eparams.split(',')).astype('int')
    idx_params = np.delete(idx_params,np.where(np.isin(idx_params,ebad)==True))
    X = X[idx_params,:]
    anames = anames[idx_params]
else:
    X = np.delete(X, ebad, axis=0)
    anames = np.delete(anames, ebad)

compfilename = args.compfile
if compfilename is not None:
    comps = args.comptouse
    data = np.genfromtxt(compfilename,unpack=True)
    if len(np.shape(data))==1:
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
        
# ------------------------------------------------------------------
# Other inputs:
n_live_points = int(args.nlive)
jitter = george.modeling.ConstantModel(np.log((200.*1e-6)**2.))
# create dictionaries to store kernels and gp classes for individual nights
kernels = {}
gps = {}

for key in nights:
    kernels[key] = np.var(nights[key][1])*george.kernels.ExpSquaredKernel(np.ones(nights[key][2].shape[0]),ndim=(nights[key][2].shape[0]),axes=range(nights[key][2].shape[0]))
    # Wrap GP object to compute likelihood
    gps[key] = george.GP(kernels[key], mean=0.0,fit_mean=False,white_noise=jitter,fit_white_noise=True)
    gps[key].compute(nights[key][2].T)

# Now define MultiNest priors and log-likelihood:
def prior(cube):
    # Prior on "median flux" is uniform:
    cube[0] = utils.transform_uniform(cube[0],-2.,2.)
    # Pior on the log-jitter term (note this is the log VARIANCE, not sigma); from 0.01 to 100 ppm:
    cube[1] = utils.transform_uniform(cube[1],np.log((0.01e-3)**2),np.log((100e-3)**2))
    pcounter = 2
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
    mflux,ljitter = cube[0],cube[1]
    pcounter = 2
    model = mflux
    models = {}
    
    if compfilename is not None:
        for key in nights:
            for i in range(nights[key][3].shape[0]):
                models[key] = model + cube[pcounter]*nights[key][3][i]
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
        
    return logl 

n_params = 3 + X.shape[0] * tnights
if compfilename is not None:
    n_params +=  Xc.shape[0]*tnights

print('Number of external parameters:',X.shape[0]*tnights)
print('Number of comparison stars:',Xc.shape[0]*tnights)
print('Number of counted parameters:',n_params)
out_file = out_folder+'out_multinest_trend_george_'

import pickle
# If not ran already, run MultiNest, save posterior samples and evidences to pickle file:
if not os.path.exists(out_folder+'GP_posteriors_trend_george.pkl'):
    # Run Dynesty:
    from multiprocessing import Pool
    nthreads = 4
    with contextlib.closing(Pool(processes=nthreads-1)) as executor:
        tic = clocking_time.time()
        sampler = dynesty.DynamicNestedSampler(loglike, prior, n_params, bound='single', sample='rwalk', nlive=n_live_points, pool=executor, queue_size=nthreads)
        sampler.run_nested()
        toc = clocking_time.time()
        print('\t Sampling took {:1.2f} hours.'.format((toc - tic)/3600))
        # Get output:
        #output = pymultinest.Analyzer(outputfiles_basename=out_file, n_params = n_params)
        results = sampler.results
    mc_samples = results['samples']
    
    a_lnZ = results['logz']
    out = {}
    out['posterior_samples'] = {}
    out['lnZ'] = a_lnZ
    out['posterior_samples']['unnamed'] = mc_samples # for easy read of output
    out['posterior_samples']['mmean'] = mc_samples[:,0]
    out['posterior_samples']['ljitter'] = mc_samples[:,1]
  
    pcounter = 2
    xc_coeffs = {}
    if compfilename is not None:
        for key in nights: # splitting comparison and alpha saves into different nights
            out['posterior_samples']['comps_'+key] = {}
            xc_coeffs[key] = []
            for i in range(Xc.shape[0]):
                xc_coeffs[key].append(mc_samples[:,pcounter])
                out['posterior_samples']['comps_'+key]['xc'+str(i)] = mc_samples[:,pcounter]
                pcounter += 1
    out['posterior_samples']['max_var'] = mc_samples[:,pcounter]
    pcounter = pcounter + 1
    alphas = {}
    for key in nights:
        out['posterior_samples']['alphas_'+key] = {}
        alphas[key] = []
        for i in range(X.shape[0]):
            alphas[key].append(mc_samples[:,pcounter])
            out['posterior_samples']['alphas_'+key]['alpha'+str(i)] = mc_samples[:,pcounter]
            pcounter = pcounter + 1  

    pickle.dump(out,open(out_folder+'GP_posteriors_trend_george.pkl','wb'))
else:
    mc_samples = pickle.load(open(out_folder+'GP_posteriors_trend_george.pkl','rb'))['posterior_samples']['unnamed']
    out = pickle.load(open(out_folder+'GP_posteriors_trend_george.pkl','rb'))

# Extract posterior parameter vector:
cube = np.median(mc_samples,axis=0)
cube_var = np.var(mc_samples,axis=0)

mflux,ljitter = cube[0],cube[1]

models = {}
if compfilename is not None:
    for key in nights:
        model = mflux
        #print(mmean)
        for i in range(Xc.shape[0]):
            model = model + np.median(out['posterior_samples']['comps_'+key]['xc'+str(i)])*nights[key][3][i]#Xc[i,idx]
            models[key] = model # not sure if should be indented

max_var = np.median(out['posterior_samples']['max_var'])

alphas = {}
gp_vector = {}
for key in nights:
    alphas[key] = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        alphas[key][i] = np.median(out['posterior_samples']['alphas_'+key]['alpha'+str(i)])
    gp_vector[key] = np.append(np.append(ljitter,np.log(max_var)),np.log(1./alphas[key]))

# Evaluate model:     
residuals = {}
for key in nights:
    residuals[key] = nights[key][1] - models[key]
    gps[key].set_parameter_vector(gp_vector[key])

# Get prediction from GP:
pred_mean = {}
pred_var = {}
pred_std = {}
for key in nights:
    pred_mean[key], pred_var[key] = gps[key].predict(residuals[key], nights[key][2].T, return_var=True)
    pred_std[key] = np.sqrt(pred_var[key])
'''
models = {}
if compfilename is not None:
    for key in nights:
        model = mflux
        #print(mmean)
        for i in range(Xc.shape[0]):
            model = model + np.median(out['posterior_samples']['comps_'+key]['xc'+str(i)])*nights[key][3][i]#Xc[i,idx]
            models[key] = model # not sure if should be indented
'''
print('\nPLOTTING...... \n')
        
# PLOT 1 - The raw light curve with GP model

# REMINDER:
# Nights[key][0] is time, [1] is flux

f, axarr = plt.subplots(tnights, sharey='row')

fout = {}
fout_err = {}
pred_mean_f = {}
plotcount = 0
for key in nights:
    fout[key],fout_err[key] = exotoolbox.utils.mag_to_flux(nights[key][1]-models[key],np.ones(len(nights[key][0]))*np.sqrt(np.exp(ljitter)))
    #print(comp_models[key], nights[key][1])
    axarr[plotcount].errorbar(nights[key][0] - int(nights[key][0][0]),fout[key],yerr=fout_err[key],fmt='.')
    pred_mean_f[key],fout_err[key] = exotoolbox.utils.mag_to_flux(pred_mean[key],np.ones(len(nights[key][0]))*np.sqrt(np.exp(ljitter)))

    axarr[plotcount].plot(nights[key][0] - int(nights[key][0][0]),pred_mean_f[key])
    axarr[plotcount].set_xlabel('Time (BJD - '+str(int(nights[key][0][0]))+')')
    plotcount += 1

f.suptitle('Raw LCs with GP Model')
plt.show()

'''
fout,fout_err = exotoolbox.utils.mag_to_flux(fall-model,np.ones(len(tall))*np.sqrt(np.exp(ljitter)))
plt.errorbar(tall - int(tall[0]),fout,yerr=fout_err,fmt='.')
pred_mean_f,fout_err = exotoolbox.utils.mag_to_flux(pred_mean,np.ones(len(tall))*np.sqrt(np.exp(ljitter)))
plt.plot(tall - int(tall[0]),pred_mean_f)
plt.xlabel('Time (BJD - '+str(int(tall[0]))+')')
plt.ylabel('Relative flux')
plt.title(name+'\nRaw LC w/ GP Model')
plt.figtext(0.5, 0.15, 'Evidence: '+str(round(out['lnZ'],4)), horizontalalignment='center')
plt.savefig('raw_lc.png')
#plt.show()
plt.gcf().clear()

print('Raw light curve saved!')
'''
# PLOT 2 - Residuals to the GP model

f, axarr = plt.subplots(tnights, sharey='row')
plotcount = 0
for key in nights:
    nights[key][1] = nights[key][1] - models[key] - pred_mean[key]
    axarr[plotcount].errorbar(nights[key][0] - int(nights[key][0][0]),nights[key][1],yerr=np.ones(len(nights[key][0]))*np.sqrt(np.exp(ljitter)),fmt='.')
    axarr[plotcount].set_xlabel('Time (BJD - '+str(int(nights[key][0][0]))+')')
    plotcount +=1

f.suptitle('Residuals to the GP Model')
plt.show()
'''
fall = fall - model - pred_mean
#plt.errorbar(tall,fall,yerr=np.ones(len(tall))*np.sqrt(np.exp(ljitter)),fmt='.')
plt.errorbar(tall - int(tall[0]),fall,yerr=np.ones(len(tall))*np.sqrt(np.exp(ljitter)),fmt='.')
plt.title(name+'\nResiduals')
plt.savefig('residuals.png')
plt.gcf().clear()

print('Residuals saved!')
'''
# PLOT 3 - Detrended light curve

f, axarr = plt.subplots(tnights, sharey='row')
plotcount = 0
for key in nights:
    fout[key],fout_err[key] = exotoolbox.utils.mag_to_flux(nights[key][1],np.ones(len(nights[key][0]))*np.sqrt(np.exp(ljitter)))
    axarr[plotcount].errorbar(nights[key][0] - int(nights[key][0][0]),fout[key],yerr=fout_err[key],fmt='.')
    axarr[plotcount].set_xlabel('Time (BJD - '+str(int(nights[key][0][0]))+')')
    axarr[plotcount].set_ylim(0.98, 1.02)
    plotcount += 1
    
'''
fileout = open('detrended_lc.dat','w')
for i in range(len(tall)):
    fileout.write('{0:.10f} {1:.10f} {2:.10f} {3:.10f}\n'.format(tall[i],fout[i],fout_err[i],lcmodel[i]))
fileout.close()
'''


f.suptitle('Detrended LCs')
f.text(0.04, 0.5, 'Relative Flux', va='center', rotation='vertical')

plt.subplots_adjust(hspace=0.75)

plt.show()


'''
fout,fout_err = exotoolbox.utils.mag_to_flux(fall,np.ones(len(tall))*np.sqrt(np.exp(ljitter)))
fileout = open('detrended_lc.dat','w')
for i in range(len(tall)):
    fileout.write('{0:.10f} {1:.10f} {2:.10f}\n'.format(tall[i],fout[i],fout_err[i]))
fileout.close()
mdev = np.std(fall) * 1000 # magnitude
fdev = np.std(fout) * 100 #percent
plt.errorbar(tall - int(tall[0]),fout,yerr=fout_err,fmt='.')
plt.xlabel('Time (BJD - '+str(int(tall[0]))+')')
plt.ylabel('Relative flux')
plt.title(name+'\nGP Detrended LC')
plt.figtext(0.5, 0.15, '$\sigma_m$ = '+str(round(fdev,3))+'% = '+str(round(mdev,3))+' mmag', horizontalalignment='center')
plt.savefig('detrended_lc.png')
plt.gcf().clear()

print('Detrended light curve saved!')

# PLOT 4 - Alpha Posteriors

alist = [] # stores full alphas
amed = [] # stores median of alphas
acounter = 0 # find amount of alphas for plotting
for key in out['posterior_samples']:
    if 'alpha' in key:
        acounter += 1
        alist.append(out['posterior_samples'][key])
        amed.append(str(round(np.median(out['posterior_samples'][key]),4)))

f, axarr = plt.subplots(acounter, sharex='col')
text = 'Median Posteriors: \n' # to be put at bottom of figure
for i in range(len(amed)):
    axarr[i].hist(alist[i], bins=500, color='black')
    axarr[i].set_title(anames[i], fontsize=10)
    text = text + anames[i] + '- ' + amed[i] +' ' 

#plt.xlim(0, 4.0)
f.suptitle('Alpha Posteriors: \n'+name, fontsize='large')
plt.subplots_adjust(hspace=0.75,top=0.85)
plt.figtext(0.1, 0.01,text, fontsize=8)

plt.savefig('alpha_posteriors.png')

print('Alpha posteriors saved!')

# PLOT 5 - Comparison Posteriors
clist = []
cmed = []
ccounter = 0
for key in out['posterior_samples']:
    if 'xc' in key:
        ccounter += 1
        clist.append(out['posterior_samples'][key])
        cmed.append(str(round(np.median(out['posterior_samples'][key]),4)))

f, axarr = plt.subplots(acounter, sharex='col')
text = 'Median Posteriors: \n'
for i in range(len(cmed)):
    axarr[i].hist(clist[i], bins=500, color='black')
    axarr[i].set_title('xc'+str(i), fontsize=10)
    text = text + 'xc'+str(i) + '- ' + cmed[i] +' ' 

f.suptitle('Comparison Posteriors: \n'+name, fontsize='large')
plt.subplots_adjust(hspace=0.75,top=0.85)
plt.figtext(0.1, 0.01,text, fontsize=8)

plt.savefig('comps_posteriors.png')

print('Comparison posteriors saved!')
'''
print('\nEvidence: ', out['lnZ'])




import numpy as np
import readgadget,readfof #Pylians
import redshift_space_library as RSL #Pylians
import MAS_library as MASL  #Pylians
import smoothing_library as SL  #Pylians


def compute_PDF():
    #These parameters are specific to the Quijote simulations, so just determining where the data is read from.
    snapdir = '/data101/boyle/quijote_haloes/'
    snapnum = 4
    i = 0 #Index of the realisation to be read in
    cosmo = 'fiducial'

    z = 0.0 # Redshift of sample
    num_weighting = True # Number weighted PDF (mass weighting is also possible).
    zspace = True #Want to output redshift-space PDF.
    minp = 50 #Minimum number of particles for a halo to be included in the sample. Equivalent to a minimum mass.
    maxp = None
    Filter='Top-Hat' #Spherical top-hat.
    smoothing = 20.0 # Smoothing radius, in this case 20 Mpc/h
    grid = 1024 #Grid cells per dimension
    MAS = 'CIC' #Counts in Cells
    threads = 2
    fname_out = 'test_halo_pdf.dat'
    BoxSize = 1000 #Length of one side of the box in Mpc/h.

    Omega_m = 0.3175
    Omega_l = 1.-Omega_m
    Hubble = 100.0*np.sqrt(Omega_m*(1.0+z)**3+Omega_l)#km/s/(Mpc/h)


    FoF = readfof.FoF_catalog(snapdir+cosmo+'/'+str(i), snapnum, long_ids=False, swap=False, SFR=False, read_IDs=False)
    # Reading in halo catalogue
    mass = FoF.GroupMass*1.e10 #Halo masses (Msun/h)
    pos = FoF.GroupPos/1.e3 #Positions of haloes (Mpc/h)
    particles = FoF.GroupLen #Number of particles in haloes
    if zspace: #Redshift space
        velocities = FoF.GroupVel * (1 + z) #Peculiar velocities
        RSL.pos_redshift_space(pos, velocities, BoxSize, Hubble, z, 2) #Convert halo positions into redshift space
        # positions

    # Filtering out haloes below a certain mass / particle number
    filter = particles>minp
    mass = mass[filter]
    pos = pos[filter]
    particles = particles[filter]

    # Filtering out haloes above a certain mass / particle number
    if maxp is not None:
        filter2 = particles<maxp
        mass = mass[filter2]
        pos = pos[filter2]

    # Creating a grid on which to deposit the haloes
    N = np.zeros((grid, grid, grid), dtype=np.float32)
    # Putting the haloes on the grid
    if num_weighting:
        MASL.MA(pos, N, BoxSize, MAS)
    else:
        MASL.MA(pos, N, BoxSize, MAS, W=mass)

    #Defining the smoothing filter.
    W_k = SL.FT_filter(BoxSize, smoothing, grid, Filter, threads)
    #Applying the smoothing filter.
    N_smoothed = SL.field_smoothing(N, W_k, threads)

    # The field needs to be renormalised by the smoothing volume after smoothing:
    v = (4. / 3.) * np.pi * (smoothing*(grid/BoxSize)) ** 3.
    N_smoothed *= v
    N_smoothed_mean = np.mean(N_smoothed) # Measuring the mean halo number density.
    delta_smoothed = (N_smoothed/N_smoothed_mean)-1. # Generating the same field in terms of delta_halo (so N/mean(
    # N)-1).
    np.savetxt(fname_out[:-4]+'_Nmean.dat', np.array([N_smoothed_mean])) # Saving the mean number density (a good
    # check).

    # Measuring and outputting the halo number density PDF: P(N).
    binedges = np.arange(-0.5, 100, 1.0) #This way, the bins correspond to having their centres at 1,2,3... haloes
    # per sphere (which gives a smooth PDF).
    bins = (binedges[1:]+binedges[:-1])/2
    pdf, binedges = np.histogram(N_smoothed, bins=binedges, density=True)
    np.savetxt(fname_out[:-4]+'_N.dat', np.transpose([bins, pdf]), delimiter='\t')

    # Measuring and outputting the halo overdensity PDF: P(\delta_h). The 'get_binedges' function below just converts
    # the bins used for N above into equivalent bins in delta.
    binedges = get_binedges(binedges, smoothing, snapnum, minp, maxp, cosmo=cosmo)
    bins = (binedges[1:]+binedges[:-1])/2
    pdf, binedges = np.histogram(delta_smoothed, bins=binedges, density=True)
    np.savetxt(fname_out[:-4]+'_delta.dat', np.transpose([bins, pdf]))

    return

def get_binedges(binedges_N, smoothing, snapnum, minp, maxp, cosmo=None):
    Nmeans = 50
    if not cosmo:
        cosmo = 'fiducial/'
    N_mean = 0
    V = 1.e9
    v = (4. / 3.) * np.pi * smoothing ** 3.
    for i in range(Nmeans):
        fof = readfof.FoF_catalog('/data101/boyle/quijote_haloes/'+cosmo+str(i), snapnum, long_ids=False, swap=False,
                                  SFR=False, read_IDs=False)
        L = fof.GroupLen

        if maxp is not None:
            N = len(L[(L>minp)&(L<maxp)])
        else:
            N = len(L[L>minp])
        N_mean += N*v/V
    N_mean = N_mean/Nmeans
    binedges = (binedges_N/N_mean)-1.
    return binedges
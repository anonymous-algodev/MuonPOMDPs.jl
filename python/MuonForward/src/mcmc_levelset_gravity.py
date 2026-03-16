import numpy as np
from scipy.stats import norm, uniform
from scipy import ndimage
from scipy.spatial import procrustes
from scipy.ndimage import affine_transform
import skfmm
from src.affine_trans_2D import affine_trans_2D

def mcmc_levelset_grav(initial_model, iter_num, realism_idxes,
                       vel_range_x, vel_range_y, vel_range_z, anisotropy_ang,
                       step_size=0.75, sigma_coeff = 0.05):

    '''
    This is the main function for sampling intrusion leveset models using McMC from gravity inversion with the Mineral-X demo case. 
    Reference for this protype: https://www.dropbox.com/scl/fi/3su22pseoio21nt5038dc/Ideon_project_01252024.pptx?rlkey=br14uslpf022d35fczhsn5uew&dl=0
    Paper reference:https://www.sciencedirect.com/science/article/pii/S0098300423001231 
    
    Input paratmers
    * initial_model: initial intrusion model, 2D or 3D array. initial intrusion can be any model, i.e. sphere or ellipsoid
    * iter_num: mcmc iteration number
    * realism_idxes: section index of 3D model corresponding to the geological sections. See the input example of this case. 
    * vel_range_x, vel_range_y, vel_range_z, anisotropy_ang: parameter bounds for the velocity model
    * step_size: step size of velocity purtbuation, 0-1. suggested: 0.75
    * sigma_coeff: hyperparameter of the target loss distribution, float, 0-1. 
                The smaller, the closer to loss=0. Suggested: 0.05, 
                
    Ouput 
    * model_catche: accepted intrusion models (levesets).
    * loss_array: loss function values over McMC iterations.
    * grav_cache: forward modeled gravity of each accepted intrusion models.
    '''
    
    dim = len(initial_model.shape)
    model_pre, loss_pre, grav_pre, weight_grav, weight_shape, sigma = mcmc_initialization(initial_model)
    
    # McMC
    loss_array, model_cache, grav_cache = [], [], []
    for ii in tqdm(np.arange(iter_num)):
        # input parameters for simulating velocity field using GRF
        GRF_params = GRF_params_smpl()
        
        # velocity perturbation 
        model_next, model_discrete_next = velocity_perturb(model_pre, dim, GRF_params, step_size)

        ### Loss Function ###
        # calculate the distnce between new model and geological realism
        shploss_next = dist_multsec(realism_idxes, model_discrete_next, 
                                 geo_realism, rot_angle=0, scale=1, translate=False)
        # calculate (L2) mismatch betwen geophysical data
        gravloss_next, grav_next = grav_loss(model_discrete_next, dens_gabbro, grav_obs)

        # calculate the total loss: graivty loss + geological shape loss
        loss_next = weight_grav*gravloss_next + weight_shape*shploss_next

        # 1. calculating Hastings ratio
        alpha1 = min(1,np.exp((loss_pre**2-loss_next**2)/(sigma**2)))
        # acceptance prob
        if model_discrete_next[:,:,56:].sum()>0:
            alpha = 0 # Not accepting models above the surface.
        else:
            alpha = min(1, alpha1)
        # 2. accept or not
        u = np.random.uniform(0, 1)
        if (u <= alpha): # accept 
            model_pre = model_next
            loss_pre =loss_next
            grav_pre = grav_next

        model_cache.append(model_pre)
        loss_array.append(loss_pre)
        grav_cache.append(grav_pre)
    
    return model_catche, loss_array, grav_cache

def mcmc_initialization(initial_model):
    '''This is the function to create the initial states for running McMC
    
    Input
    initial_model: initial intrusion model, 2D or 3D array.
    '''
    model_pre = skfmm.distance(initial_model.copy()-0.5)
    model_discrete_pre = (model_pre>=0)*1
    #### Loss function ###
    # geological shape loss
    shploss_pre = dist_multsec(realism_idxes, model_discrete_pre, geo_realism, rot_angle=0, scale=1)

    # gravity loss
    gravloss_pre, grav_pre = grav_loss(model_discrete_pre, dens_gabbro, grav_obs)
    
    # Total loss: w1*graivty loss + w2*geological shape loss
    ## Loss weight: use the first loss to calculated the weights for gravity and weight loss, in order to have the same magnitude
    weight_grav, weight_shape = 1/gravloss_pre, 1/shploss_pre
    loss_pre = weight_grav*gravloss_pre + weight_shape*shploss_pre

    # calculate the hyperparameter sigma
    sigma = loss_pre*sigma_coeff
    
    return model_pre, loss_pre, grav_pre, weight_grav, weight_shape, sigma


def grav_loss(intrusion_model, dens_gabbro, grav_obs):
    '''Calculate l2 loss (mismatch) of geophysical data'''
    # run simpeg simulation
    dens_ = dens_gabbro * intrusion_model.reshape(-1, order='F')
    grav_ = gravity_sim.dpred(dens_[act_grid])
    
    # calcuate gravity loss, L2
    gravloss = np.linalg.norm(grav_ - grav_obs)
    return gravloss, grav_

def GRF_params_smpl(): 
    '''This is the function to sample velocity field parameters'''
    GRF_params_smpl = [[np.random.uniform(vel_range_x[0], vel_range_x[1]),
               np.random.uniform(vel_range_y[0], vel_range_y[1]),
               np.random.uniform(vel_range_z[0], vel_range_z[1])],
              [np.random.uniform(anisotropy_ang[0], anisotropy_ang[1])]]
    return GRF_params_smpl



def generate_velocity(theta, xyz_dim, seed = None):
    '''Generate 2D/3D velocity field'''
    if len(xyz_dim)==2:
        model = gs.Gaussian(dim=len(xyz_dim), # 2D model
                            var= theta[1], # variance
                            len_scale = [theta[2],theta[3]], # ranges
                            angles = [theta[4]*np.pi/180] # angle
                           )
    elif len(xyz_dim)==3:
        model = gs.Gaussian(dim=len(xyz_dim), # 2D model
                            var= theta[1], # variance
                            len_scale = [theta[2],theta[3], theta[4]], # ranges
                            angles = [theta[4]*np.pi/180] # angle
                           )        
    if seed:
        srf = gs.SRF(model,seed = seed)
    else: 
        srf = gs.SRF(model)
    if len(xyz_dim)==2:
        field = srf.structured([np.arange(xyz_dim[0]), 
                                np.arange(xyz_dim[1])]) + theta[0]        
    elif len(xyz_dim)==3:
        field = srf.structured([np.arange(xyz_dim[0]), 
                                np.arange(xyz_dim[1]), 
                                np.arange(xyz_dim[2])]) + theta[0]
    return field 


def dist_multsec(realism_idxes, model, georealism, rot_angle=0, scale=1, translate=False):
    '''
    calculate distance between model and realism
    realism_idxes: model section indexes of for distance calclation
    model: model, 3D array
    georealism: realistic geological sections, list of 2D arrays,
    '''
    pa_prev = 0
    for i in range(len(realism_idxes)):
        m_pa, real_pa, pa_ = uncertainPA_2D(model[realism_idxes[i]], georealism[i], rot_angle, scale, translate=translate)
        pa_prev += pa_
    return pa_prev



def uncertainPA_2D(litho_m_array, realism_array, rot_angle=0, scale=1, translate=False):
    '''
    Procrustes distance analysis of 2D array of litho indicators between model and geological realism.
    for a given uncertain angle and scaling factor. 
    Step 1. translate both arrays in the same center.
    Step 2. rotate and scale the realism array using the given uncertain rot angle and scling factor.
    Step 3. calcualte the distance between the two array.
    parameters:
        litho_m_array: modelled litho indicators, 2D array of 0/1 indicator. 
        realism_array: geological realism litho indicators, 2D array of 0/1 indicator. 
        rot_angle: given a uncertain rotation angle for realism, float, betwen (0-360) degree
        scale: given a uncertain scaling factor for realism, float, between (0,+)
    return:
        [litho_array_new, realism_array_new, pa_dist]
        litho_array_new: new litho model array after traslation
        realism_array_new: new realism litho array after scaling and rotation
    
    '''
    
    # calculate realism indictor centers
    real_centr_yx = np.argwhere(realism_array==1).mean(axis=0)
    
    if translate: # if moving the model and realism to the same center locations for comparison
        # calculate litho model indictor centers
        m_centr_yx = np.argwhere(litho_m_array==1).mean(axis=0)
        
        # translate litho model to the same center as realism
        m_translate_yx =  real_centr_yx - m_centr_yx

        m_trans = affine_trans_2D(litho_m_array, rot_center_yx=m_centr_yx,
                                  rot_angle = 0,  scale = 1, 
                                  shear = 0.0, translate_yx=m_translate_yx)
        
    else:  # if not moving the model and realism to the same center locations for comparison, default.
        
        m_trans = litho_m_array
    # rotate and scale the realism with the uncertain angles and scaling factor
#     real_r_s = affine_trans_2D(realism_array, rot_center_yx=real_centr_yx,
#                                rot_angle = rot_angle,  scale = scale, 
#                                shear = 0.0, translate_yx=0)
    real_r_s = realism_array
    # calculate new distance (L2) as the uncertain PA distance
    pa_dist = np.sqrt(((real_r_s-m_trans)**2).sum())
    [litho_array_new, realism_array_new, pa_dist]  = [m_trans, real_r_s, pa_dist]
    return [litho_array_new, realism_array_new, pa_dist]

def velocity_perturb(model, dim, GRF_params, step_size):
    '''
    GRF_params: input parameters for velocity simulation using gaussian random field (GRF)
                [[vel_range_x_, vel_range_y_, vel_range_z_], [anisotropy_ang_]]
    '''
    #  velocity fields using GRF
    if dim ==2:
        theta_i = np.array([0, 1, 
                            GRF_params[0][0], GRF_params[0][1], GRF_params[1][0]]
                          ) # you can change the range and anisotropy
        # create 2D velocity fields
        nx, ny = model.shape
        velocity = generate_velocity(theta_i, xyz_dim=[nx, ny])
    
    elif dim ==3:
        theta_i = np.array([0, 1, GRF_params[0][0], GRF_params[0][1], GRF_params[0][2], 
                            GRF_params[1][0]
                           ]) # you can change the range and anisotropy
        # create 3D velocity fields
        nx, ny, nz = model.shape
        velocity = generate_velocity(theta_i,xyz_dim=[nx, ny, nz])
    # perturbation
    _, F_eval = skfmm.extension_velocities(model, velocity, dx=1, order = 1)
    
    # levelset stochastic perturbation
    dt = step_size/np.max(abs(F_eval))
    model_next = model - dt * F_eval # Advection
    
    model_discrete_next = (model_next>=0)*1
    return model_next, model_discrete_next

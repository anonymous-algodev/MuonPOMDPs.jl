import matplotlib.pyplot as plt
import skfmm
import numpy as np
def orebody_plt_3D(orebody_3d, plt_title='', z_ticks = None):
    '''3D plot of orebody, by []
    input:
    orebody_3D: a 3D array of 0-non ore,1-ore for the orebody'''
    m_sdf = skfmm.distance(orebody_3d-0.5)
    plt_contour = np.argwhere((m_sdf<2) & (m_sdf>=-0.5))
    if z_ticks ==None:
        z_ticks=np.arange(0,orebody_3d.shape[2]+1, orebody_3d.shape[2]/4).astype(int)


    ax = plt.figure(figsize=(7,7)).add_subplot(projection='3d')

    ax.scatter(plt_contour[:,0],
               plt_contour[:,1],
               plt_contour[:,2],
               c = 'y',
               linewidth=0.1,edgecolor='k',
               marker='s', vmax=1,
               s=25)
    ax.set_title(plt_title, fontsize=18)
    ax.set_xlim(0, m_sdf.shape[0]), ax.set_ylim(0, m_sdf.shape[1]), ax.set_zlim(0,m_sdf.shape[2])
    ax.set_xlabel('X dim', fontsize = 15); ax.set_ylabel('Y dim', fontsize = 15); ax.set_zlabel('Z dim', fontsize = 15)
    ax.xaxis.pane.set_edgecolor('k')
    ax.yaxis.pane.set_edgecolor('k')
    ax.zaxis.pane.set_edgecolor('k')
#     ax.set_zticks(-z_ticks)
#     plt.gca().invert_zaxis()
    return
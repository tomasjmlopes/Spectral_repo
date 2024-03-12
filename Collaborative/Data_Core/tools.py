import numpy as np
from matplotlib import *
from matplotlib.pyplot import * 
from scipy.spatial.transform import Rotation as Rot

def kabsch_umeyama(A, B): # Find transformation parameteres to transform B --> A
    assert A.shape == B.shape
    n, m = A.shape

    EA = np.mean(A, axis = 0)
    EB = np.mean(B, axis = 0)
    VarA = np.mean(np.linalg.norm(A - EA, axis = 1) ** 2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])

    R = U @ S @ VT
    c = VarA / np.trace(np.diag(D) @ S)
    t = EA - c * R @ EB

    return R, c, t

def align_maps(map1, map2, vm1 = 2000, vm2 = 0.8, fs = (12, 8)):
    """
    Find the transformation parameters to go from map1
    to map2. As an example if we use LIBS in map1 and Hyper 
    in map2, we can obtain a transformed map2, the same size
    as the libs map.
    Vm1 and Vm2 set the the vmax for each of the images plotted
    """
    fig = figure(tight_layout = True, figsize = fs)
    gs = gridspec.GridSpec(1, 2)
    ax1=fig.add_subplot(gs[0, 0])
    ax2=fig.add_subplot(gs[0, 1])

    ig = ax1.imshow(map2, cmap='terrain', vmax = vm2)
    ig = ax2.imshow(map1.T, cmap = 'inferno', interpolation = 'gaussian', vmax = vm1)
    plot1, = ax1.plot([],[],color='r',marker='o',ms=5)
    plot2, = ax2.plot([],[],color='r',marker='o',ms=5)

    left_plot_x=[]
    left_plot_y=[]
    left_plot_z=[]

    right_plot_x=[]
    right_plot_y=[]
    right_plot_z=[]

    mmap_aligned = []
    R_opt = []
    T_opt = []
    S_opt = []

    colorrs=[0,'c','r','k']

    def right_click(event):
        """
        Point Selection
        """
        if event.inaxes==ax1: #warrants that we are clicking the correct axes
            
            left_plot_x.append(event.xdata)
            left_plot_y.append(event.ydata)
            left_plot_z.append(0)
            plot1.set_data(left_plot_x, left_plot_y)

                
        if event.inaxes==ax2: #warrants that we are clicking the correct axes
            
            right_plot_x.append(event.xdata)
            right_plot_y.append(event.ydata)
            right_plot_z.append(0)
            plot2.set_data(right_plot_x, right_plot_y)

    def left_click(event):
        if len(right_plot_x) != len(left_plot_x):
            print("The number of points selected should be the same for both images")
        else:
            right_plot = np.array(list(zip(right_plot_x, right_plot_y, right_plot_z)))
            left_plot = np.array(list(zip(left_plot_x, left_plot_y, left_plot_z)))

            R, S, T = kabsch_umeyama(left_plot, right_plot)
            """
            We want align the hyperspectral image with the LIBS image. For that we have to
            fill an array with the LIBS shape with the hyperspectral data and for that we need 
            to find how to go from LIBS to Hyper.
            """

            R_opt.append(R)
            T_opt.append(T)
            S_opt.append(S)

            print('R: ', Rot.from_matrix(R).as_euler('zyx', degrees = True)[0])
            print('S: ', S)
            print('T: ', T)
            return R, S, T

            # (rows, cols) = im1.shape[:2]
            # M = np.float32([[1, 0, T[0]], [0, 1, T[1]]])
            # res = cv2.warpAffine(im1, M, (cols, rows))

            # M = cv2.getRotationMatrix2D((0, 0), Rot.from_matrix(R).as_euler('zyx', degrees = True)[0], 1)
            # res = cv2.warpAffine(res, M, (cols, rows))
    
            # mmap_aligned.append(res)
            

    def on_click(event):
        if event.button == MouseButton.RIGHT:
            right_click(event)

        elif  event.button == MouseButton.LEFT:
            left_click(event)

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.tight_layout()

def rotate_data(data2, map1, R, S, T):

    new_hyper = np.zeros((map1.shape[0], map1.shape[1], data2.shape[-1]))
        
    for i in range(0, map1.shape[0]):
        hyper_aligned = data2[100:250, 100:320, :]
        for j in range(0, map1.shape[1]):
            vec = np.array(T) + S * R @ np.array([i, j, 0])
            new_i,new_j = int(vec[0]), int(vec[1])
            try:
                new_hyper[i, j, :] = hyper_aligned[new_j, new_i]
            except:
                pass

    return new_hyper

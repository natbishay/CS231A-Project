import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy
import matplotlib.gridspec as gridspec
from epipolar_utils import *

'''
FACTORIZATION_METHOD The Tomasi and Kanade Factorization Method to determine
the 3D structure of the scene and the motion of the cameras.
Arguments:
    points_im1 - N points in the first image that match with points_im2
    points_im2 - N points in the second image that match with points_im1

    Both points_im1 and points_im2 are from the get_data_from_txt_file() method
Returns:
    structure - the structure matrix
    motion - the motion matrix
'''
def center_points(points):
    N = points.shape[0]
    tx = np.mean(points[:,0])
    ty = np.mean(points[:,1])
    C = np.zeros((N, points.shape[1]))
    for i in range(N):
        C[i,:] = points[i,:] - np.array([tx, ty, 0])

    return C

def factorization_method(points_im1, points_im2):
    p1_center = center_points(points_im1)
    p2_center = center_points(points_im2)

    # put the rows in s.t. W = [x1;x2;y1;y2]
    # W is 4x37
    W = np.vstack((p1_center[:,0].T, p2_center[:,0].T, p1_center[:,1].T, p2_center[:,1].T))
    U, S, V = np.linalg.svd(W, full_matrices=True)
    S_rank3 = S[:3]
    motion = U[:,:3]#@np.sqrt(np.diag(S_rank3))
    structure = np.diag(S_rank3)@V[:3, :]

    return structure, motion
    
def g(a, b): # triple checked
    return np.array([a[0]*b[0], a[0]*b[1]+a[1]*b[0], a[0]*b[2]+a[2]*b[0], a[1]*b[1], a[1]*b[2]+a[2]*b[1], a[2]*b[2]])

def metric_transformation(S, M): # M is an 2F by 3 matrix
    # need to compute G
    F = int(M.shape[0]/2)
    G = np.zeros((3*F, 6))

    for i in range(F):
        ik = M[i, :]
        jk = M[F+i, :]
        G[i,:] = g(ik, ik)
        G[F+i,:] = g(jk, jk)
        G[2*F+i,:] = g(jk, ik)

    c = np.concatenate((np.ones(2*F), np.zeros(F)))
    l = np.linalg.lstsq(G,c)[0]
    L = np.array([[l[0], l[1], l[2]],
                 [l[1], l[3], l[4]],
                 [l[2], l[4], l[5]]])

    Q = np.linalg.cholesky(L)
    M_actual = M@Q
    S_actual = np.linalg.pinv(Q)@S

    return S_actual, M_actual


def sequential_factorization(points_im1, points_im2):
    p1_center = center_points(points_im1)
    p2_center = center_points(points_im2)

    # put the rows in s.t. W = [x1;x2;y1;y2]
    # W is 4x37
    W = np.vstack((p1_center[:,0].T, p2_center[:,0].T, p1_center[:,1].T, p2_center[:,1].T))
    
    F = 2
    P = W.shape[1]
    Z = np.zeros((P,P))
    A = np.random.randn(P,3)
    Q,_ = scipy.linalg.qr(A)
    Q_bar,_ = scipy.linalg.qr(A)
    
    for f in range(F):
        Z = Z + np.outer(W[f,:], W[f,:]) + np.outer(W[f+F,:], W[f+F,:])
        Y = Z@Q
        Q,_ = scipy.linalg.qr(Y)
        H = Q@Q.T
        Y = H@Q_bar
        Q_bar,_ = scipy.linalg.qr(Y)

    
    # this part could potentially go into
#    for f in range(F):
#        H = Q@Q.T
#        Y = H@Q_bar
#        Q_bar,_ = scipy.linalg.qr(Y)
#
    
    return Q, W
    


    
if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set1_subset']:
        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points_im1 = get_data_from_txt_file(im_set + '/pt_2D_1.txt')
        points_im2 = get_data_from_txt_file(im_set + '/pt_2D_2.txt')
        points_3d = get_data_from_txt_file(im_set + '/pt_3D.txt')
        assert (points_im1.shape == points_im2.shape)

        # Run the Factorization Method
        structure, motion = factorization_method(points_im1, points_im2)
        
        s, m = metric_transformation(structure, motion)
        
        Q, W = sequential_factorization(points_im1, points_im2)
        _, e_q = np.linalg.eig(Q)
        
        dist1 = np.linalg.norm(Q@Q.T - structure.T@structure)
        print(dist1)
        _,_,Vt = np.linalg.svd(W)
        
        dist2 = np.linalg.norm(Vt.T@Vt - structure.T@structure)
        print(dist2-dist1)
        dE = np.linalg.norm(Vt.T@Vt - Q@Q.T)
        print(dE)
        
        

        # Plot the structure
        fig = plt.figure()
        ax = fig.add_subplot(121, projection = '3d')
        scatter_3D_axis_equal(structure[0,:], structure[1,:], structure[2,:], ax)
        ax.set_title('Factorization Method')
#        ax = fig.add_subplot(132, projection = '3d')
#        scatter_3D_axis_equal(points_3d[:,0], points_3d[:,1], points_3d[:,2], ax)
#        ax.set_title('Ground Truth')
        ax = fig.add_subplot(122, projection = '3d')
        scatter_3D_axis_equal(Q[:,0], Q[:,1], Q[:,2], ax)
        ax.set_title('Sequential Factorization')

        plt.show()

"""
Active Shape Model for point cloud deformation
 Shuichi Akizuki, Chukyo Univ.
 Email: s-akizuki@sist.chukyo-u.ac.jp
"""
import numpy as np
from sklearn.decomposition import PCA
import copy
import os.path as osp
from math import *

import open3d as o3
import cv2


class ActiveShapeModel():
    """
        Required packages:
            import numpy as np
            from sklearn.decomposition import PCA
            import copy
    """
    def __init__( self, clouds ):
        """
        clouds: a list of open3d point cloud
        """
        self.clouds = clouds
        self.mean_size = 0.0
        
        flag, n_points = self.check_n_points()
        if flag == False:
            print("Error!! Number of points in the training set are not same.")
            print( n_points )
            exit()
        
        # M: number of models
        M = self.get_n_pcd()
        
        self.X = np.zeros([M,len(self.clouds[0].points)*3])
        for i, c in enumerate(self.clouds):    
            self.X[i] = self.to_vec(c)
        
        # Standardization
        self.mean_size = self.X.mean()
        self.std = self.X.std()
        self.X2 = (self.X - self.mean_size)/self.std
        
        self.pca = self.PCA(self.X2)
        
        
    def get_X(self):
        """
        Return features (Nx3, M)
        """
        return self.X
        
    def check_n_points( self ):
        """
        Check number of points in the training set. Assume same
        Return:
            flag: True(same) or False (not same)
            list: a list that stored points
        """
        n_points = []
        for c in self.clouds:
            n_points.append(len(c.points))
            
        n_points = np.array(n_points)
        
        return n_points.all(), n_points
        
    def get_n_pcd( self ):
        """
        Return:
            # point clouds in the training set
        """
        
        return len(self.clouds)
    
    def get_mean_size( self ):
        """
        Calc the mean size of shapes
        """
        # Normalize by mean size        
        M = self.get_n_pcd()
        norms = np.zeros(M)
        for i, x in enumerate(self.X):      
            norms[i] = np.linalg.norm(x,ord=2)
        
        
        return np.mean(norms)
    
    def to_pcd( self, vec ):
        """
        Convert (Nx3,) vector to open3d pcd
        """
        vec3d = vec.reshape((-1,3))
        vec3d = (vec3d * self.std )+self.mean_size
        pcd = o3.geometry.PointCloud()
        pcd.points = o3.utility.Vector3dVector(vec3d)
    
        return pcd
    
    def to_vec( self, pcd ):
        """
        Convert open3d pcd to (Nx3,) vector
        """
        tmp = np.asarray(pcd.points)
        vec = tmp.reshape(tmp.shape[0]*3)
        
        return vec
    
    def PCA( self, X ):
        """
        Apply PCA
        """
        pca = PCA()
        x_pca = pca.fit(X)
        
        return pca
    
    def get_components(self):
        
        return self.pca.components_
    
    def get_mean(self):
        
        return self.pca.mean_
    
    def get_explained_variance_ratio(self):
        
        return self.pca.explained_variance_ratio_
    
    def deformation( self, param, n_dim=10 ):
        """
        Shape defomation by deformation parameter 
        
        Input:
            param: deformation param (ndim,)
            n_dim: # dimension
        """
        weight = self.get_components()
        
        deformed = copy.deepcopy(self.get_mean())
        cnt = 0
        for w,p in zip(weight, param):

            if n_dim == cnt:
                break

            deformed += w*p
            cnt+=1
        
        cloud_deformed = self.to_pcd(deformed)
        return cloud_deformed
    
    def projection(self,data):
        """
            Projection data which is converted by to_vec() 
            to the latent space.
        """
        # Standardization
        data2 = (data - self.mean_size)/self.std
        return self.pca.transform([data2])
    
    def get_all_projection(self):
        """
            Get all projection at a time
            each row indicates a projection of data
        """
        projections = np.zeros((self.X.shape[0],self.X.shape[0]))
        for i, x in enumerate(self.X):
            p = self.projection(x)
            projections[i] = p
        
        return projections
    
    def get_asm_info( self ):
        """
            Get a dictionary data consist of
            the mean shape, components, and size info.
        """
        
        asm_info = {}
        asm_info["mean_shape"] = self.get_mean()        
        asm_info["components"] = self.get_components()
        asm_info["size_mean"] = self.mean_size
        asm_info["size_std"] = self.std
        
        return asm_info 
    
    def save_asm_info( self, name ):
        
        info = self.get_asm_info()
        
        np.savez( name, 
                  mean_shape=info["mean_shape"],
                  components=info["components"],
                  size_mean=info["size_mean"],
                  size_std=info["size_std"]
                )
    
def load_asmds( root, synset_names ):
    """ 複数のSSMの読み込み
    Args:
      root(str):　データセットのルートディレクトリ
      synset_names(str):　クラスのリスト．冒頭はBGなので無視する．
    Return:
      dict:SSMDeformationの辞書変数
    """
    print("Root dir:", root )    
    asmds = {}
    for s in range(len(synset_names)-1):
        paths = set_paths( root, synset_names[s+1] )
        trainset_path = paths["trainset_path"]
        info = np.load( osp.join(trainset_path,"info.npz"))
        asmd = ASMdeformation( info )
        asmds[synset_names[s+1]] = asmd
    
    return asmds
    
    
# 変形パラメータの平均と標準偏差に基づく確率でパラメータをサンプリング
# 入力
# 　 params: 変形パラメータが1行ずつ積み重なった行列
def generate_parameter( params ):
    param_mean = np.mean(params,axis=0)
    param_std = np.std(params,axis=0)
    b = np.random.normal(param_mean, param_std)
    return b

# 変形パラメータのMIN-MAXの範囲を超えないようなサンプリング
# 入力
# 　 params: 変形パラメータが1行ずつ積み重なった行列
def generate_parameter_minmax( params ):
    param_min = np.min(params,axis=0)
    param_max = np.max(params,axis=0)
    b = np.random.uniform(param_min, param_max)
    return b


class ASMdeformation():
    def __init__( self, asm_info ):
        
        self.mean_shape = asm_info['mean_shape'] # load mean shape
        self.component = asm_info['components' ] # load components
        self.mean = asm_info['size_mean'] # size mean
        self.std = asm_info['size_std'] # size std

    def get_dp_dim( self ):
        
        return self.component.shape[0]
        
    def to_pcd( self, vec ):
        """
        Convert (Nx3,) vector to open3d pcd
        """
        vec3d = vec.reshape((-1,3))
        vec3d = (vec3d * self.std )+self.mean
        pcd = o3.geometry.PointCloud()
        pcd.points = o3.utility.Vector3dVector(vec3d)

        return pcd
    
    def deformation( self, dp ):
        """
        Deformation
        """

        deformed = copy.deepcopy( self.mean_shape )
        for c,p in zip( self.component, dp):
            deformed += c*p
        cloud_deformed = self.to_pcd( deformed )
        
        return cloud_deformed


    
#####################
# Visualization tool
#####################
def generate_latent_space_image( ap, im_size=200 ):
    """ Visualization function for latent spase as image. (use top 2 dimensions) 
    Args:
      ap(ndarray): Eigen vectors generated by ASM.get_all_projection()
      im_size(int): image size
      
    Return:
      ndarray(uint8,3ch): image of latent space
    """
    im_size = im_size
    im_latent = np.zeros([im_size,im_size,3]).astype(np.uint8)
    offset = np.array([im_size/2, im_size/2])
    cv2.line( im_latent, (int(im_size/2),0),(int(im_size/2),im_size),
             (100,100,100),1 )
    cv2.line( im_latent, (0,int(im_size/2)),(im_size,int(im_size/2)),
             (100,100,100),1 )
    
    for i in range(ap.shape[0]):
        pix = ap[i,0:2] + offset
        cv2.circle( im_latent, 
                    (int(pix[0]),int(im_size-pix[1])), 
                    2, (0,255,0), -1, cv2.LINE_AA )
    return im_latent


def continuous_shape_deformation( asm, pcd ):

    param = asm.projection(asm.get_mean())

    d_param = []
    d_param = copy.deepcopy(param)
    direction = np.zeros(d_param.shape) 
    direction[0:2] = 1.0 
    print("copy")
    
    d_param_id = 0
    d_param_id2 = 1
    cnt = 0
    
    def deformation( vis, param ):
        deformed = asm.deformation( param, asm.get_n_pcd() )
        pcd.points = deformed.points
        vis.update_geometry( pcd )
    
    def shape_edit( vis ):
        nonlocal param
        nonlocal direction
        nonlocal d_param
        nonlocal d_param_id
        nonlocal cnt
        upper = 0.10
        lower = -0.10
        dim = 0
        
        if upper < d_param[d_param_id]:
            direction[d_param_id] = -1.0
        elif d_param[d_param_id] < lower:
            direction[d_param_id] = 1.0
        if upper < d_param[d_param_id2]:
            direction[d_param_id2] = -1.0
        elif d_param[d_param_id2] < lower:
            direction[d_param_id2] = 1.0
        """    
        if cnt == 300:
            d_param_id +=1
            if d_param_id == 5: 
                d_param_id = 0
            cnt=0
        cnt+=1
        """
        step = 0.001*direction
        d_param += step
        print(cnt, d_param_id, " step", step )
        print(d_param)
        deformation( vis, d_param )
        
        return False
    
    
    o3.visualization.draw_geometries_with_animation_callback([pcd], shape_edit, width=640, height=500)


def deformation_with_key_callback( asm, pcd,p):
    
    param = copy.deepcopy(p)
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    
    def show_image( img_, param_ ):

        param = str(param_)
        img = np.array( 255.0*img_, np.uint8 )
        img = cv2.putText( img, param, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (20,0,0), 1, cv2.LINE_AA )
        #cv2.imwrite( "hoge.png", img )
        cv2.imshow( "Result", img )   
        cv2.waitKey(5) #
    
    def deformation( vis, param ):
        deformed = asm.deformation( param, param.shape[0] )
        pcd.points = deformed.points
        vis.update_geometry( pcd )
        buf = vis.capture_screen_float_buffer(do_render=False)
        np_buf = np.asarray( buf )
        show_image( np_buf, param )

    def pc1p(vis):
        nonlocal param
        step = np.zeros(param.shape[0])
        step[0] = 1.0
        param += step
        deformation( vis, param )
        
        return False

    def pc1m(vis):
        nonlocal param
        step = np.zeros(param.shape[0])
        step[0] = 1.0
        param -= step
        deformation( vis, param )
        
    def pc2p(vis):
        nonlocal param
        step = np.zeros(param.shape[0])
        step[1] = 1.0
        param += step
        deformation( vis, param )
        
        return False

    def pc2m(vis):
        nonlocal param
        step = np.zeros(param.shape[0])
        step[1] = 1.0
        param -= step
        deformation( vis, param )

    def pc3p(vis):
        nonlocal param
        step = np.zeros(param.shape[0])
        step[2] = 1.0
        param += step
        deformation( vis, param )
        
        return False

    def pc3m(vis):
        nonlocal param
        step = np.zeros(param.shape[0])
        step[2] = 1.0
        param -= step
        deformation( vis, param )
        
        return False
    
    def reset(vis):
        nonlocal param
        param = np.zeros(param.shape[0])
        deformation( vis, param )
        
        return False

    key_to_callback = {}
    key_to_callback[ord("K")] = pc1p
    key_to_callback[ord("L")] = pc1m
    key_to_callback[ord(",")] = pc2p
    key_to_callback[ord(".")] = pc2m
    key_to_callback[ord("N")] = pc3p
    key_to_callback[ord("M")] = pc3m
    key_to_callback[ord("R")] = reset
    o3.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback, width=640, height=500 )
    cv2.destroyAllWindows()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import abel
import scipy.spatial
import multiprocessing as mp
from timeit import default_timer as timer
from joblib import Parallel, delayed

def tetra_volume(p1,p2,p3,p4):
     return np.abs(np.linalg.det(np.array([p1-p4,p2-p4,p3-p4])))/6

def cut(points,mask,n,d):
        #Determine cuts of edges with plane -> polyeder
        pairs = [(i,j) for i in points[mask==True] for j in points[mask==False]] #Pair vertices
        is_points = np.zeros((len(pairs),3)) #intersection points with plane
        for i, pair in enumerate(pairs):
            p1 = pair[0]
            r = pair[1]-pair[0]
            pis = p1-(np.dot(n,p1)-d)/np.dot(n,r)*r #intersection point
            is_points[i] = pis
        points_upper = np.concatenate((points[mask==True],is_points)) #original points above + intersection points
        return points_upper


#Circular isolines in the right half-plane      
def sample(number_of_lines_per_layer,number_of_points_per_line,phi_min,phi_max):
    sample_points = np.zeros(((number_of_lines_per_layer-1)*self.number_of_points_per_line+1,2))
    iso_indices = np.zeros(((number_of_lines_per_layer-1)*number_of_points_per_line+1))
    angles = np.linspace(phi_min,phi_max,number_of_points_per_line)
    sample_points[0] = [0,0]
    iso_indices[0] = 0
    for i in range(1,number_of_lines_per_layer):
        r = R[i]
        sample_points[number_of_points_per_line*(i-1)+1:number_of_points_per_line*i+1] = [[r,phi] for phi in angles]
        iso_indices[number_of_points_per_line*(i-1)+1:number_of_points_per_line*i+1] = i
    sample_points_cylindrical = np.array([[sample_points[p,0],sample_points[p,1],h[l]] for l in range(number_of_layers) for p in range(sample_points.shape[0])])
    get_cartesian()
    point_to_isoline = np.array([[l,iso_indices[p]] for l in range(number_of_layers) for p in range(sample_points.shape[0])],dtype=int) #first h, then r

def get_cartesian(self):
    self.sample_points_cartesian = np.zeros((self.sample_points_cylindrical.shape[0],3))
    for i in range(self.sample_points_cylindrical.shape[0]):
        r = self.sample_points_cylindrical[i][0]
        phi = self.sample_points_cylindrical[i][1]
        h = self.sample_points_cylindrical[i][2]
        self.sample_points_cartesian[i][0] = r*np.cos(phi)
        self.sample_points_cartesian[i][1] = r*np.sin(phi)
        self.sample_points_cartesian[i][2] = h
        
def tetrahedralize(self):
    self.tetrahedra = scipy.spatial.Delaunay(self.sample_points_cartesian).simplices
    self.number_of_tetrahedra = self.tetrahedra.shape[0]

def calculate_matrix_elements(self,image_mesh):
    I = np.zeros((image.rows,image.cols,self.number_of_layers,number_of_lines_per_layer))
    #self.calculate_tetrahedra_volumes()
    #self.group_tetrahedra_by_isolines()
    #self.group_tetrahedra_by_pixels(image)
    """Think about this again tomorrow!"""
    for i in range(self.number_of_tetrahedra):
        rc = self.pixel_indices[i]
        lis = self.isoline_indices[i]
        if isinstance(rc[0],list):
            r = rc[0][0]
            if isinstance(rc[1],list):
                c = rc[1][0]
                for li in lis:
                    I[r,c,li[0],li[1]] = self.tetra_cuts[i][0,0]
                    I[r,c+1,li[0],li[1]] = self.tetra_cuts[i][0,1]
                    I[r+1,c,li[0],li[1]] = self.tetra_cuts[i][1,0]
                    I[r+1,c+1,li[0],li[1]] = self.tetra_cuts[i][1,1]
            else:
                c = rc[1]
                for li in lis:
                    I[r,c,li[0],li[1]] = self
        
    
def calculate_tetrahedra_volumes(self):
    self.tetrahedra_volumes = np.zeros(self.number_of_tetrahedra)
    for i in range(self.number_of_tetrahedra):
        self.tetrahedra_volumes[i] = tetra_volume(*self.sample_points_cartesian[self.tetrahedra[i]])
    self.remove_degenerates()

def remove_degenerates(self):
    self.tetrahedra = self.tetrahedra[self.tetrahedra_volumes>1e-15]
    self.tetrahedra_volumes = self.tetrahedra_volumes[self.tetrahedra_volumes>1e-15]
    print(len(self.tetrahedra_volumes)/self.number_of_tetrahedra)
    self.number_of_tetrahedra = len(self.tetrahedra_volumes)
        
def group_tetrahedra_by_isolines(self): #first h, then r
    self.isoline_indices = np.zeros((self.number_of_tetrahedra,4,2),dtype=int)
    for i in range(self.number_of_tetrahedra):
        self.isoline_indices[i] = self.point_to_isoline[self.tetrahedra[i]]
        
def group_tetrahedra_by_pixels(self,image_mesh):
    self.pixel_indices = [None] * self.number_of_tetrahedra
    for i in range(self.number_of_tetrahedra):
        self.pixel_indices[i] = self.find_pixels(i, self.tetrahedra[i], image_mesh)
        
def group_2(self,image_mesh):
    self.pixels2 = [self.find_pixels(i,self.tetrahedra[i],image_mesh) for i in range(self.number_of_tetrahedra)]
    
    #num_cores = mp.cpu_count()
    #self.pixels2 = Parallel(n_jobs=num_cores)(delayed(circs.find_pixels)(i,circs.tetrahedra[i],frame) for i in range(circs.number_of_tetrahedra))                                      
def find_pixels(self, tetrahedron_index, tetrahedron, image_mesh): #x-axis as integration axis
    points = self.sample_points_cartesian[tetrahedron]
    y_coords = points[:,1]
    z_coords = points[:,2]
    ymin = -image_mesh.cols/2.0*image_mesh.pixelsize
    zmax = +image_mesh.rows/2.0*image_mesh.pixelsize
    y_indices = np.floor((y_coords-ymin)/image_mesh.pixelsize)
    y_indices = y_indices.astype(int)
    z_indices = np.floor((zmax-z_coords)/image_mesh.pixelsize)
    z_indices = z_indices.astype(int)
    
    yindmax = np.max(y_indices)
    yindmin = np.min(y_indices)
    zindmax = np.max(z_indices)
    zindmin = np.min(z_indices)
    
    if yindmax-yindmin == 0 and zindmax-zindmin == 0:
        return [zindmax,yindmax]
    elif yindmax-yindmin > 1 or zindmax-zindmin > 1:
        print("WARNING!")
        return [np.nan,np.nan]
    else:
        volumes = self.subdivide_volume(tetrahedron_index,tetrahedron,zindmax,yindmax,image_mesh)
        rows = [zindmin,zindmax] if zindmax - zindmin == 1 else zindmax
        cols = [yindmin,yindmax] if yindmax - yindmin == 1 else yindmax
        self.tetra_cuts.update({tetrahedron_index:volumes})
        return [rows,cols]
        
    
def subdivide_volume(self,tetrahedron_index,tetrahedron,zindmax,yindmax,image_mesh):
    points = self.sample_points_cartesian[tetrahedron]
    y_coords = points[:,1]
    z_coords = points[:,2]
    
    #Step 1 Cut by z (Upper Half):
    z_plane = image_mesh.upper_boundaries[zindmax,yindmax-1] #upper boundary of lower pixels (row axis showing down)
    z_mask = np.ones(4,bool) #point in upper half?
    z_mask[z_coords-z_plane < 1e-15] = False
    points_upper = cut(points,z_mask,np.array([0,0,1]),z_plane)
    volume_upper = scipy.spatial.ConvexHull(points_upper).volume if np.size(points_upper) > 0 else 0
    
    #Step 2 Cut by y (Right Half):
    y_plane = image_mesh.right_boundaries[zindmax,yindmax-1] #right boundary of left pixels (column axis showing right)
    y_mask = np.ones(4,bool) #point in right half?
    y_mask[y_coords-y_plane < 1e-15] = False
    points_right = cut(points,y_mask,np.array([0,1,0]),y_plane)
    volume_right = scipy.spatial.ConvexHull(points_right).volume if np.size(points_right) > 0 else 0
    
    #Step 3 Cut upper points for upper right quarter:
    y_mask2 = np.ones(len(points_upper),bool)
    y_coords2 = points_upper[:,1]
    y_mask2[y_coords2-y_plane < 1e-15] = False
    points_upper_right = cut(points_upper,y_mask2,np.array([0,1,0]),y_plane)
    volume_upper_right = scipy.spatial.ConvexHull(points_upper_right).volume if np.size(points_upper_right) > 0 else 0
    
    #Step 4: Rearrange
    volume_total = self.tetrahedra_volumes[tetrahedron_index]
    volumes = np.zeros((2,2))
    volumes[0,0] = volume_upper - volume_upper_right 
    volumes[0,1] = volume_upper_right
    volumes[1,0] = volume_total - volume_right - volumes[0,0]
    volumes[1,1] = volume_right - volume_upper_right
    
    return volumes

###################################
######### Actual Program ##########
###################################



N=3
M=7

#Set up image frame
rows = N
cols = N
pixelsize = 2.0/(N-1)
upper_boundaries = np.zeros((rows,cols))
right_boundaries = np.zeros((rows,cols))
pixel_centers = np.zeros((rows,cols,2))
vert_center = rows//2
hori_center = cols//2
for i in range(rows):
    for j in range(cols):
        pixel_centers[i,j] = [(vert_center-i)*pixelsize,(j-hori_center)*pixelsize]
        upper_boundaries[i,j] = pixel_centers[i,j][0]+0.5*pixelsize
        right_boundaries[i,j] = pixel_centers[i,j][1]+0.5*pixelsize

#Set up isoline parameters
#radii, angles and heights
r_min = 0
r_max = 1
h_min = -1
h_max = 1
R = np.linspace(r_min,r_max,N)
h = np.linspace(h_min,h_max,N)
phi_min = -np.pi/2
phi_max = np.pi/2
#numbers
number_of_lines = N**2
number_of_lines_per_layer = N
number_of_layers = N
number_of_points_per_line = M
#cuts
tetra_cuts = dict()

sample(R,h,phi_min,phi_max)
sam = timer()
print('Sampled in', (sam-start)/60, "min")
circs.tetrahedralize()
tet = timer()
print('Tetrahedralized in', (tet-sam)/60, "min")
circs.calculate_tetrahedra_volumes()
vol = timer()
print('Volumes in', (vol-tet)/60, "min")
circs.group_tetrahedra_by_isolines()
iso = timer()
print('Iso in', (iso-tet)/60, "min")
circs.group_tetrahedra_by_pixels(frame)
end = timer()
print('Grouped in', (end-iso)/60, "min")
circs.group_2(frame)
end2=timer()
print((end2-end)/60)
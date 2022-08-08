import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
#import abel

def tri_area(p1,p2,p3):
    area = 0.5*np.abs(p1[0]*p2[1]+p2[0]*p3[1]+p3[0]*p1[1]-p2[0]*p1[1]-p3[0]*p2[1]-p1[0]*p3[1]) #Shoelace formula
    return area

    #Pixel and corresponding line of sight
#Assume 
class Image:
    def __init__(self,img,pixelsize):
        self.img = img
        self.size = np.shape(img)
        self.pixelsize = pixelsize
        self.pixels = self.set_pixels()
        self.boundaries = self.get_boundaries()
        
    def set_pixels(self):
        pixels = []
        for i in range(self.size[0]):
                pixels.append(Pixel(self.img[i],i,self.size[0],self.pixelsize))
        return pixels
    
    def get_boundaries(self):
        boundaries = np.zeros(self.size[0]+1)
        for i in range(self.size[0]):
            boundaries[i] = self.pixels[i].upper_boundary
        boundaries[self.size[0]] = self.pixels[-1].lower_boundary
        return boundaries
                
class Pixel:
    
    def __init__(self,value,index,imgsize,width):
        self.value = value
        self.index = index
        self.lower_boundary, self.upper_boundary = self.get_boundaries(imgsize,width)
        
    def get_boundaries(self,imgsize,width): #Coordinate boundaries of centered image, assume odd number of pixels
        #center = int(imgsize/2)
        #upper_boundary = (center-self.index+0.5)*width
        #lower_boundary = (center-self.index-0.5)*width
        upper_boundary = (imgsize-self.index)*width
        lower_boundary = (imgsize-self.index-1)*width
        return lower_boundary, upper_boundary    

#Circular isolines in the right half-plane
class Isolines:
    
    def __init__(self,r_min,r_max,N,M):
        self.R = np.linspace(r_min,r_max,N)
        self.number_of_lines = N
        self.number_of_points_per_line = M
        self.phi_min = 0
        self.phi_max = np.pi/2
        self.sample_points_polar = np.empty((0,2))
        self.sample_points_cartesian = np.empty((0,2))
        self.triangles = None
        self.triangle_areas = None
        self.triangle_cuts = dict()
                
    def sample(self):
        sample_points = [[0,0]] #origin r = 0, phi = 0
        for r in self.R[1:]:
            angles = np.linspace(self.phi_min,self.phi_max,self.number_of_points_per_line)
            sample_points.extend([r,phi] for phi in angles) #fix later, extend is slow
        self.sample_points_polar = np.array(sample_points)
        self.get_cartesian()
    
    def triangulate(self):
        self.triangles = tri.Triangulation(self.sample_points_cartesian[:,0], self.sample_points_cartesian[:,1])
        
    def calculate_matrix_elements(self,image):
        matrix = np.zeros((image.size[0],self.number_of_lines))
        self.calculate_triangle_areas()
        print("Areas calculated")
        self.group_by_isolines()
        print("Grouped by isolines")
        self.group_by_pixels(image)
        print("Grouped by pixels")
        for i in range(self.triangles.triangles.shape[0]):
            m = self.pixel_indices[i]
            n = self.isoline_indices[i]
            if isinstance(m,list):
                if len(m)==2:
                    for j in n:
                        matrix[m[0],j] += 1/3.0*(self.triangle_cuts[i])
                        matrix[m[1],j] += 1/3.0*(self.triangle_areas[i] - self.triangle_cuts[i])
                else:
                    for j in n:
                        matrix[m[0],j] += 1/3.0*(self.triangle_cuts[i][0])
                        matrix[m[1],j] += 1/3.0*(self.triangle_areas[i]-self.triangle_cuts[i][0]-self.triangle_cuts[i][1])
                        matrix[m[2],j] += 1/3.0*(self.triangle_cuts[i][1])
            else:
                for j in n:
                    matrix[m,j] += 1/3.0*(self.triangle_areas[i])
        return 2*matrix/image.pixelsize
        
        
    def calculate_triangle_areas(self):
        self.triangle_areas = np.zeros(self.triangles.triangles.shape[0])
        for i in range(self.triangles.triangles.shape[0]):
            self.triangle_areas[i] = tri_area(*self.sample_points_cartesian[self.triangles.triangles[i]])
    
    def group_by_isolines(self):
        self.isoline_indices = [None] * self.triangles.triangles.shape[0]
        for i in range(self.triangles.triangles.shape[0]):
            radii = self.sample_points_polar[self.triangles.triangles[i]][:,0]
            #base attribution scheme
            #unique, counts = np.unique(radii, return_counts=True)
            #attributed_radius = unique[counts==2][0]
            #self.isoline_indices[i] = int((self.number_of_lines-1) * attributed_radius/(self.R[self.number_of_lines-1]-self.R[0]))
            
            #simple average scheme
            indices = (self.number_of_lines-1) * radii/(self.R[self.number_of_lines-1]-self.R[0])
            self.isoline_indices[i] = list(map(round,indices))
            
            
    def group_by_pixels(self,image):
        self.pixel_indices = [None] * self.triangles.triangles.shape[0]
        for i in range(self.triangles.triangles.shape[0]):
            self.pixel_indices[i] = self.find_pixel(i, self.triangles.triangles[i], image)
        
    def find_pixel(self, triangle_index, triangle, image):
        y_coords = self.sample_points_cartesian[triangle][:,1]
        bounds = -image.boundaries #flip order from descending to ascending
        indices = np.searchsorted(bounds,-y_coords) #flip sign as above
        indices = np.where(indices != 0, indices, 1)
        unique_indices, index_positions, counts = np.unique(indices,return_index=True,return_counts=True)
        if unique_indices.shape[0]==1:
            return unique_indices[0]-1
        elif unique_indices.shape[0] == 2:
            if unique_indices[1]-unique_indices[0] > 1:
                print("Error") #If that error occurs, increase M
            else:
                tip_pos = index_positions[counts==1][0]
                area = self.cut_by_pixel(triangle_index, triangle, unique_indices[0], tip_pos, image) #cut by pixel and calculate area of subtriangle
                self.triangle_cuts.update({triangle_index:area})
                return [unique_indices[counts==1][0]-1,unique_indices[counts==2][0]-1] #put tip index first so that the mapping of the subarea to the pixel is clear and doesn't need to be stored as extra information
        else:
            #return 0
            if np.any(np.diff(unique_indices)>1):
                print("Error") #If that error occurs, increase M
            else:
                tip_pos1 = index_positions[0]
                tip_pos2 = index_positions[2]
                area1 = self.cut_by_pixel(triangle_index, triangle, unique_indices[0], tip_pos1, image)
                area2 = self.cut_by_pixel(triangle_index, triangle, unique_indices[1], tip_pos2, image)
                #print((self.triangle_areas[triangle_index]-area1-area2)/self.triangle_areas[triangle_index]*100)
                self.triangle_cuts.update({triangle_index:[area1,area2]})
                return list(unique_indices-1)
        
    def cut_by_pixel(self, triangle_index, triangle, border_index, tip_pos, image):
        #Identify tip point and base points
        mask = np.ones(3,bool)
        mask[tip_pos] = False
        tip = self.sample_points_cartesian[triangle][tip_pos]
        base1 = self.sample_points_cartesian[triangle][mask][0]
        base2 = self.sample_points_cartesian[triangle][mask][1]
        
        #Calculate intersection points
        bound = image.boundaries[border_index]
        t1 = (bound - base1[1])/(tip[1]-base1[1])
        t2 = (bound - base2[1])/(tip[1]-base2[1])
        p1 = np.array([t1 * tip[0] + (1-t1) * base1[0],bound])
        p2 = np.array([t2 * tip[0] + (1-t2) * base2[0],bound])
        
        #calculate area of tip-intersection-intersection triangle
        return tri_area(tip,p1,p2)

    def get_cartesian(self):
        self.sample_points_cartesian = np.zeros((self.sample_points_polar.shape[0],2))
        for i in range(self.sample_points_polar.shape[0]):
            r = self.sample_points_polar[i][0]
            phi = self.sample_points_polar[i][1]
            self.sample_points_cartesian[i][0] = r*np.cos(phi)
            self.sample_points_cartesian[i][1] = r*np.sin(phi)
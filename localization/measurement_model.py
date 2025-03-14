import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation
from skimage.feature import corner_harris, corner_peaks
from skimage.morphology import skeletonize
from cv2 import HoughLinesP
import cv2

class MeasurementModel:
    def __init__(self):
        self.corner_weight = 0.4
        self.line_weight = 0.4
        self.endpoint_weight = 0.2
        self.corner_distance_threshold = 0.3  # meters
        self.line_distance_threshold = 0.2    # meters
        self.cached_features = None
        
    def extract_map_features(self, map_data, map_resolution):
        """Extract features from occupancy grid"""
        if self.cached_features is not None:
            return self.cached_features
            
        # Convert occupancy grid to binary image
        binary_map = (map_data == 100).astype(np.uint8)
        
        # Extract corners
        corners = corner_peaks(corner_harris(binary_map), min_distance=5)
        
        # Extract lines using probabilistic Hough transform
        edges = cv2.Canny(binary_map * 255, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, 
                               minLineLength=10, maxLineGap=5)
        
        # Extract endpoints (dead ends in thinned map)
        skeleton = skeletonize(binary_map)
        endpoints = self.find_endpoints(skeleton)
        
        self.cached_features = {
            'corners': corners * map_resolution,
            'lines': lines[0] if lines is not None else np.array([]),
            'endpoints': endpoints * map_resolution
        }
        
        return self.cached_features
    
    def find_endpoints(self, skeleton):
        """Find endpoints in skeletonized image"""
        endpoints = []
        for i in range(1, skeleton.shape[0]-1):
            for j in range(1, skeleton.shape[1]-1):
                if skeleton[i,j]:
                    neighbors = np.sum(skeleton[i-1:i+2, j-1:j+2]) - 1
                    if neighbors == 1:
                        endpoints.append([j, i])  # x,y format
        return np.array(endpoints)
    
    def compute_likelihood(self, particle, scan_points, map_features, map_info):
        """Compute likelihood of scan points matching map features"""
        likelihood = 0.0
        
        # Transform scan points to map frame
        scan_points_map = self.transform_points(scan_points, particle)
        
        # Corner matching
        corner_likelihood = self.match_corners(scan_points_map, map_features['corners'])
        
        # Line matching
        line_likelihood = self.match_lines(scan_points_map, map_features['lines'])
        
        # Endpoint matching
        endpoint_likelihood = self.match_endpoints(scan_points_map, map_features['endpoints'])
        
        # Combine likelihoods
        likelihood = (self.corner_weight * corner_likelihood +
                     self.line_weight * line_likelihood +
                     self.endpoint_weight * endpoint_likelihood)
        
        return likelihood
    
    def transform_points(self, points, particle):
        """Transform points from particle frame to map frame"""
        cos_theta = np.cos(particle[2])
        sin_theta = np.sin(particle[2])
        R = np.array([[cos_theta, -sin_theta],
                     [sin_theta, cos_theta]])
        
        transformed = np.dot(points, R.T) + particle[:2]
        return transformed
    
    def match_corners(self, scan_points, map_corners):
        """Match scan points to map corners"""
        if len(map_corners) == 0 or len(scan_points) == 0:
            return 0.0
            
        distances = np.min(np.linalg.norm(
            scan_points[:, np.newaxis] - map_corners, axis=2), axis=1)
        matches = distances < self.corner_distance_threshold
        return np.mean(matches) if len(matches) > 0 else 0.0
    
    def match_lines(self, scan_points, map_lines):
        """Match scan points to map lines"""
        if len(map_lines) == 0 or len(scan_points) == 0:
            return 0.0
            
        distances = []
        for line in map_lines:
            point_line_distances = self.point_to_line_distance(
                scan_points, line[:2], line[2:])
            distances.append(point_line_distances)
        
        min_distances = np.min(distances, axis=0)
        matches = min_distances < self.line_distance_threshold
        return np.mean(matches) if len(matches) > 0 else 0.0
    
    def match_endpoints(self, scan_points, map_endpoints):
        """Match scan endpoints to map endpoints"""
        if len(map_endpoints) == 0 or len(scan_points) == 0:
            return 0.0
            
        distances = np.min(np.linalg.norm(
            scan_points[:, np.newaxis] - map_endpoints, axis=2), axis=1)
        matches = distances < self.corner_distance_threshold
        return np.mean(matches) if len(matches) > 0 else 0.0
    
    def point_to_line_distance(self, points, line_start, line_end):
        """Calculate perpendicular distance from points to line segment"""
        line_vec = line_end - line_start
        line_length = np.linalg.norm(line_vec)
        if line_length == 0:
            return np.linalg.norm(points - line_start, axis=1)
            
        line_unit_vec = line_vec / line_length
        point_vec = points - line_start
        
        # Project point onto line vector
        projection = np.dot(point_vec, line_unit_vec)
        projection = np.clip(projection, 0, line_length)
        
        # Calculate closest point on line segment
        closest_points = line_start + projection[:, np.newaxis] * line_unit_vec
        
        return np.linalg.norm(points - closest_points, axis=1)
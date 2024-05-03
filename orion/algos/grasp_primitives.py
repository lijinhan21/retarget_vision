import numpy as np
from scipy.stats import chi2
from retarget.utils.configs import load_config

class HandPose:
    def __init__(self):
        self.thumb0 = None
        self.thumb = None
        self.index = None
        self.middle = None
        self.ring = None
        self.pinky = None

    def set_pose(self, thumb0, thumb, index, middle, ring, pinky):
        self.thumb0 = thumb0
        self.thumb = thumb
        self.index = index
        self.middle = middle
        self.ring = ring
        self.pinky = pinky

class GraspPrimitive:
    def __init__(self, config_file='configs/grasp_primitives.yaml'):
        self.name_to_pose = {}

        config = load_config(config_file)

        self.thumb0_config = config.thumb0_config
        self.thumb_config = config.thumb_config
        self.finger_config = config.finger_config

        for key, value in self.thumb0_config.items():
            self.thumb0_config[key] = value / 180 * np.pi
        for key, value in self.thumb_config.items():
            self.thumb_config[key] = value / 180 * np.pi
        for key, value in self.finger_config.items():
            self.finger_config[key] = value / 180 * np.pi

        self.add_from_dict(config.grasp_primitives)

    def add_primitive(self, name, pose):
        self.name_to_pose[name] = pose

    def add_from_dict(self, dict):
        for name, pose in dict.items():
            hand_pose = HandPose()
            hand_pose.set_pose(pose["thumb0"], pose["thumb"], pose["index"], pose["middle"], pose["ring"], pose["pinky"])
            self.add_primitive(name, hand_pose)

    def point_map_to_primitive(self, joint_angles):
        def euclidean_distance(point1, point2):
            return np.linalg.norm(point1 - point2)
        def nearest_neightbor(data_points, query_point):
            return min(data_points, key=lambda x: euclidean_distance(x[0], query_point))
        
        data = []
        for name, pose in self.name_to_pose.items():
            data.append((self.get_joint_angles(name), name))

        return nearest_neightbor(data, joint_angles)
    
    def sequence_map_to_primitive(self, joint_angles_list):
        def remove_outliers_mahalanobis(points):
            points = np.array(points)
            covariance_matrix = np.cov(points, rowvar=False)
            covariance_matrix_inv = np.linalg.pinv(covariance_matrix)
            mean = np.mean(points, axis=0)

            # Mahalanobis distance for each point
            distances = [((point - mean) @ covariance_matrix_inv @ (point - mean)) for point in points]
            threshold = chi2.ppf((1-0.01), df=points.shape[1])  # 99% confidence interval

            # Filter points within the threshold
            non_outliers = [distance <= threshold for distance in distances]
            return points[non_outliers]
        
        def find_best_anchor(points, anchors):
            # Initialize minimum distance to a large value
            min_distance = float('inf')
            best_anchor = (None, None)

            # Iterate over each anchor
            for anchor in anchors:
                # Calculate the total distance from all points to this anchor
                total_distance = sum(np.linalg.norm(np.array(point) - np.array(anchor[0])) for point in points)
                
                # If this total distance is less than the current minimum, update
                if total_distance < min_distance:
                    min_distance = total_distance
                    best_anchor = anchor

            # Return the best anchor and the minimum distance
            return best_anchor

        def find_robust_anchor(points, anchors):
            min_median_distance = float('inf')
            best_anchor = (None, None)

            for anchor in anchors:
                distances = [np.linalg.norm(point - anchor[0]) for point in points]
                median_distance = np.median(distances)
                if median_distance < min_median_distance:
                    min_median_distance = median_distance
                    best_anchor = anchor
            return best_anchor
        
        def find_mode_anchor(points, anchors):
            votes = []
            for point in points:
                votes.append(self.point_map_to_primitive(point)[1])
            mode = max(set(votes), key=votes.count)
            return (self.get_joint_angles(mode), mode)

        anchors = []
        for name, pose in self.name_to_pose.items():
            anchors.append((self.get_joint_angles(name), name))

        # points = remove_outliers_mahalanobis(joint_angles_list)
        points = joint_angles_list
        ret = find_mode_anchor(points, anchors)
        return ret[0], ret[1]

    def save_as_dict(self):
        dict = {}
        for name, pose in self.name_to_pose.items():
            dict[name] = {
                "thumb0": pose.thumb0,
                "thumb": pose.thumb,
                "index": pose.index,
                "middle": pose.middle,
                "ring": pose.ring,
                "pinky": pose.pinky,
            }

        ret = {"thumb0_config": {}, 
               "thumb_config": {}, 
               "finger_config": {}, 
               "grasp_primitives": dict}
        for key, value in self.thumb0_config.items():
            print(key, value)
            ret["thumb0_config"][key] = int(value * 180 / np.pi)
        for key, value in self.thumb_config.items():  
            ret["thumb_config"][key] = int(value * 180 / np.pi)
        for key, value in self.finger_config.items():
            ret["finger_config"][key] = int(value * 180 / np.pi)

        return ret

    def get_joint_angles(self, name):
        pose = self.name_to_pose[name]
        thumb0 = self.thumb0_config[pose.thumb0]
        thumb = self.thumb_config[pose.thumb]
        index = self.finger_config[pose.index]
        middle = self.finger_config[pose.middle]
        ring = self.finger_config[pose.ring]
        pinky = self.finger_config[pose.pinky]

        return np.array([thumb0, thumb, index, middle, ring, pinky])

    def real_control_list(self):
        def map_to_real_control(value):
            # original: 0 - np.pi/2, real: 1000 - 0
            return 1000 - 1000 * value / (np.pi / 2)

        ret = {}
        for name, pose in self.name_to_pose.items():
            thumb0 = map_to_real_control(self.thumb0_config[pose.thumb0])
            thumb = map_to_real_control(self.thumb_config[pose.thumb])
            index = map_to_real_control(self.finger_config[pose.index])
            middle = map_to_real_control(self.finger_config[pose.middle])
            ring = map_to_real_control(self.finger_config[pose.ring])
            pinky = map_to_real_control(self.finger_config[pose.pinky])
            ret[name] = [thumb0, thumb, index, middle, ring, pinky]
        
        return ret

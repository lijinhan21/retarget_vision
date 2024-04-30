import json
import numpy as np
from retarget.retargeter import SMPLGR1Retargeter
from retarget.utils.configs import load_config

class Retargeter:
    def __init__(self, config='GR1_retarget/configs/smpl_gr1.yaml', vis=False, example_data=None):
        config = load_config(config)
        self.retargeter = SMPLGR1Retargeter(config, vis=vis)
        
        with open("GR1_retarget/configs/ik_weight_template.json", "r") as f:
            self.weights = json.load(f)
        
        if example_data is not None:
            self.calibrate(example_data)

    def calibrate(self, example_data):
        self.retargeter.calibrate(example_data)
    
    def retarget(self, smplh_data, offset={"link_RArm7":[0, 0, 0]}):
        return self.retargeter(smplh_data, offset=offset)

    def update_weight(self, weights):
        for i, link in enumerate(self.weights["GR1_body"]):
            self.weights["GR1_body"][link]["position_cost"] = weights[i]
        self.retargeter.update_weight(self.weights)
    
    def get_weight(self):
        return self.weights
    
    def control(self, link_name='link_LArm7', relative_trans=np.eye(4)):
        return self.retargeter.control({link_name: self.weights["GR1_body"][link_name]}, relative_trans)

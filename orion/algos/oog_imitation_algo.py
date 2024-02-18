"""This is a file that defines the imitation algorithm for the OOG manipulation."""


class OOGImitationAlgo():
    def __init__(
            human_video_path,
    ):
        raise NotImplementedError
    

    def init_rollout(self):
        raise NotImplementedError
    
    def find_matching_oog(self):
        return matched_oog, next_subgoal_oog
    
    def compute_action_seq(self):
        raise NotImplementedError
    
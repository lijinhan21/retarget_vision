
from enum import Enum


class AnnotationMode(Enum):
    SCRIBBLE = 1
    GROUNDED_SAM = 2
    NAIVE_SAM = 3

# define a mapping from string to AnnotationMode
AnnotationModeMapping = {
    "scribble": AnnotationMode.SCRIBBLE,
    "grounded-sam": AnnotationMode.GROUNDED_SAM,
    "naive-sam": AnnotationMode.NAIVE_SAM,
}
def get_annotation_mode_from_string(mode_str):
    assert(mode_str in AnnotationModeMapping)
    return AnnotationModeMapping[mode_str]


class OOGPreprocessor():
    def __init__(self):
        raise NotImplementedError
    
    def frist_frame_annotation(self,
                               mode="scribble"):
        assert(mode in AnnotationModeMapping), f"mode {mode} not supported, only support {AnnotationModeMapping.keys()}"

        annotation_mode = get_annotation_mode_from_string(mode)

        if annotation_mode == AnnotationMode.SCRIBBLE:
            raise NotImplementedError
        elif annotation_mode == AnnotationMode.GROUNDED_SAM:
            raise NotImplementedError
        elif annotation_mode == AnnotationMode.NAIVE_SAM:
            raise NotImplementedError
        
    def temporal_segmentation(self,
                              mode="buds+tap"):
        raise NotImplementedError
        
    def vos_annotation(self,
                       model="xmem"):
        raise NotImplementedError
        
    def tap_annotation(self,
                       model="cotracker"):
        raise NotImplementedError
    
    def hand_annotation(self,
                        model="hamer"):
        raise NotImplementedError
    


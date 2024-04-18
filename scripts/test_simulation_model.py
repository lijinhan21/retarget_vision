import mujoco
from mujoco import viewer
import time
mjmodel = mujoco.MjModel.from_xml_path('orion/assets/franka_emika_panda/scene_droid.xml')

start_time = time.time()
mjdata = mujoco.MjData(mjmodel)
# mjdata.qpos[:] = [0, 0, 0,-1.57079, 0, 1.57079, -0.7853, 0.04, 0.04, 0, 0, 0, 0]
mjdata.qpos[:] = [      0.09162008114028396,
        -0.19826458111314524,
        -0.01990020486871322,
        -2.4732269941140346,
        -0.01307073642274261,
        2.30396583422025,
        0, # 0.8480939705504309,
         0.04, 0.04, 0, 0, 0, 0
        ]
print("Time to load model: ", time.time() - start_time)

viewer.launch(mjmodel, mjdata)

exit()

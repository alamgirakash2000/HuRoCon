import sys
import os
import numpy as np
from dm_control import mjcf
import random
import string

JVRC_DESCRIPTION_PATH="models/jvrc_mj_description/xml/scene.xml"

def builder(export_path):

    print("Modifying XML model...")
    mjcf_model = mjcf.from_path(JVRC_DESCRIPTION_PATH)

    # set njmax and nconmax
    mjcf_model.size.njmax = -1
    mjcf_model.size.nconmax = -1
    mjcf_model.statistic.meansize = 0.1
    mjcf_model.statistic.meanmass = 2

    # modify skybox
    for tx in mjcf_model.asset.texture:
        if tx.type=="skybox":
            tx.rgb1 = '1 1 1'
            tx.rgb2 = '1 1 1'

    # remove all collisions
    mjcf_model.contact.remove()

    waist_joints = ['WAIST_Y', 'WAIST_P', 'WAIST_R']
    head_joints = ['NECK_Y', 'NECK_R', 'NECK_P']
    hand_joints = ['R_UTHUMB', 'R_LTHUMB', 'R_UINDEX', 'R_LINDEX', 'R_ULITTLE', 'R_LLITTLE',
                   'L_UTHUMB', 'L_LTHUMB', 'L_UINDEX', 'L_LINDEX', 'L_ULITTLE', 'L_LLITTLE']
    arm_joints = ['R_SHOULDER_Y', 'R_ELBOW_Y', 'R_WRIST_R', 'R_WRIST_Y',
                  'L_SHOULDER_Y', 'L_ELBOW_Y', 'L_WRIST_R', 'L_WRIST_Y']
    leg_joints = ['R_HIP_P', 'R_HIP_R', 'R_HIP_Y', 'R_KNEE', 'R_ANKLE_R', 'R_ANKLE_P',
                  'L_HIP_P', 'L_HIP_R', 'L_HIP_Y', 'L_KNEE', 'L_ANKLE_R', 'L_ANKLE_P']

    # remove actuators except for leg joints
    for mot in mjcf_model.actuator.motor:
        if mot.joint.name not in leg_joints:
            mot.remove()

    # remove unused joints
    for joint in waist_joints + head_joints + hand_joints + arm_joints:
        mjcf_model.find('joint', joint).remove()

    # remove existing equality
    mjcf_model.equality.remove()

    #add equality for arm joints
    arm_joints = ['R_SHOULDER_P', 'R_SHOULDER_R', 'R_ELBOW_P',
                'L_SHOULDER_P', 'L_SHOULDER_R', 'L_ELBOW_P']
    # mjcf_model.equality.add('joint', joint1=arm_joints[0], polycoef='-0.052 0 0 0 0')
    # mjcf_model.equality.add('joint', joint1=arm_joints[1], polycoef='-0.169 0 0 0 0')
    # mjcf_model.equality.add('joint', joint1=arm_joints[2], polycoef='-0.523 0 0 0 0')
    # mjcf_model.equality.add('joint', joint1=arm_joints[3], polycoef='-0.052 0 0 0 0')
    # mjcf_model.equality.add('joint', joint1=arm_joints[4], polycoef='0.169 0 0 0 0')
    # mjcf_model.equality.add('joint', joint1=arm_joints[5], polycoef='-0.523 0 0 0 0')

    SHOULDER_P = -0.52
    ELBOW_P= -0.973  #3
    SHOULER_R = -0.26 #1.7

    mjcf_model.equality.add('joint', joint1='R_SHOULDER_P', polycoef=f'{SHOULDER_P} 0 0 0 0')
    mjcf_model.equality.add('joint', joint1='R_SHOULDER_R', polycoef=f'{-SHOULER_R} 0 0 0 0')
    mjcf_model.equality.add('joint', joint1='R_ELBOW_P', polycoef=f'{ELBOW_P} 0 0 0 0')
    mjcf_model.equality.add('joint', joint1='L_SHOULDER_P', polycoef=f'{SHOULDER_P} 0 0 0 0')
    mjcf_model.equality.add('joint', joint1='L_SHOULDER_R', polycoef=f'{SHOULER_R} 0 0 0 0')
    mjcf_model.equality.add('joint', joint1='L_ELBOW_P', polycoef=f'{ELBOW_P} 0 0 0 0')
    
    # collision geoms
    collision_geoms = [
        'R_HIP_R_S', 'R_HIP_Y_S', 'R_KNEE_S',
        'L_HIP_R_S', 'L_HIP_Y_S', 'L_KNEE_S',
    ]

    # remove unused collision geoms
    for body in mjcf_model.worldbody.find_all('body'):
        for idx, geom in enumerate(body.geom):
            geom.name = body.name + '-geom-' + repr(idx)
            if hasattr(geom, 'dclass') and geom.dclass is not None and geom.dclass.dclass == "collision":
                if (geom.dclass.dclass=="collision"):
                    if body.name not in collision_geoms:
                        geom.remove()

    # move collision geoms to different group
    mjcf_model.default.default['collision'].geom.group = 3

    # manually create collision geom for feet
    mjcf_model.worldbody.find('body', 'R_ANKLE_P_S').add('geom', dclass='collision', size='0.1 0.05 0.01', pos='0.029 0 -0.09778', type='box')
    mjcf_model.worldbody.find('body', 'L_ANKLE_P_S').add('geom', dclass='collision', size='0.1 0.05 0.01', pos='0.029 0 -0.09778', type='box')

    # ignore collision
    mjcf_model.contact.add('exclude', body1='R_KNEE_S', body2='R_ANKLE_P_S')
    mjcf_model.contact.add('exclude', body1='L_KNEE_S', body2='L_ANKLE_P_S')

    # remove unused meshes
    meshes = [g.mesh.name for g in mjcf_model.find_all('geom') if g.type=='mesh' or g.type==None]
    for mesh in mjcf_model.find_all('mesh'):
        if mesh.name not in meshes:
            mesh.remove()

    # fix site pos
    mjcf_model.worldbody.find('site', 'rf_force').pos = '0.03 0.0 -0.1'
    mjcf_model.worldbody.find('site', 'lf_force').pos = '0.03 0.0 -0.1'



#### Changing the number of the visible box
# add box geoms
    for idx in range(80):  # Changed from 20 to 50
        name = 'box' + repr(idx+1).zfill(2)
        mjcf_model.worldbody.add('body', name=name, pos=[0, 0, -0.2])
        mjcf_model.find('body', name).add('geom',
                                          name=name,
                                          dclass='collision',
                                          group='0',
                                          size='0.2 0.4 0.1',  # Updated size for better visibility
                                          type='box',
                                          material='')

    # wrap floor geom in a body
    mjcf_model.find('geom', 'floor').remove()
    mjcf_model.worldbody.add('body', name='floor')
    mjcf_model.find('body', 'floor').add('geom', name='floor', type="plane", size="0 0 0.25", material="groundplane")

#### Adding Construction Beams START
    # Find the right elbow body
    right_elbow = mjcf_model.worldbody.find('body', 'R_ELBOW_P_S')
    
    # Beam parameters
    NUM_BEAMS = 2  # Can be changed to desired number of beams
    BEAM_MASS_PER_UNIT = 0.5  # 1kg per beam
    
    # Beam dimensions
    BEAM_LENGTH = 0.8  # Length of each beam
    BEAM_WIDTH = 0.05  # Width of each beam
    BEAM_HEIGHT = 0.05  # Height of each beam
    
    # Position offsets
    BEAM_FORWARD_OFFSET = 0.02  # Centered relative to elbow
    BEAM_HEIGHT_OFFSET = -0.3  # Position of the sticks in hand (Forward or backwar)
    BEAM_SIDE_OFFSET = 0.15  # Centered horizontally
    
    # Create main beam carrier body
    beam_carrier = right_elbow.add('body',
        name='beam_carrier',
        pos=[BEAM_FORWARD_OFFSET, BEAM_SIDE_OFFSET, BEAM_HEIGHT_OFFSET],
        euler=[0, 0, 1.7])
    
    # Add multiple beams in a bundle
    for i in range(NUM_BEAMS):
        # Calculate vertical offset for each beam in the bundle
        vertical_offset = (i - (NUM_BEAMS-1)/2) * (BEAM_HEIGHT * 1.2)  # 1.2 factor for slight spacing
        
        beam = beam_carrier.add('body',
            name=f'beam_{i+1}',
            pos=[0, 0, vertical_offset])
        
        beam.add('geom',
            name=f'beam_{i+1}_geom',
            type='box',
            size=[BEAM_LENGTH/2, BEAM_WIDTH/2, BEAM_HEIGHT/2],  # MuJoCo uses half-sizes
            mass=BEAM_MASS_PER_UNIT,  # Each beam is 1kg
            rgba=[0.6, 0.6, 0.6, 1],  # Grey color for construction beam
            dclass='collision',
            group='1')
        
        # Add visual bindings between beams (small cylinders)
        if i < NUM_BEAMS-1:
            spacing = BEAM_LENGTH/5  # Space between binding points
            for j in range(3):  # Add 3 binding points along the length
                position = -BEAM_LENGTH/2 + (j+1)*spacing
                beam.add('geom',
                    name=f'binding_{i+1}_{j+1}',
                    type='cylinder',
                    size=[BEAM_WIDTH/4, BEAM_HEIGHT/2],  # radius and height
                    pos=[position, 0, BEAM_HEIGHT],
                    rgba=[0.3, 0.3, 0.3, 1],  # Darker grey for bindings
                    dclass='visual')
#### Adding Construction Beams END

    # export model
    mjcf.export_with_assets(mjcf_model, out_dir=os.path.dirname(export_path), out_file_name=export_path, precision=5)
    print("Exporting XML model to ", export_path)
    return

if __name__=='__main__':
    builder(sys.argv[1])

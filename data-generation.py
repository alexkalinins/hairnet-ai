import bpy
import random
import math
import mathutils
import csv
import os

#GLOBAL_OUTPUT_PATH = 'C:\\Users\\ALEX\\Desktop\\HairNet\\output\\'  # where files are saved
GLOBAL_OUTPUT_PATH = os.path.dirname(os.path.abspath(__file__))+'\\output'  # where files are saved
START_AT = 2400  # first index e.g. img1.png
DATA_SIZE = 600  # how many images to create
EEVEE_RENDER_SAMP = 16 # how many eevee samples to use


# CAMERA_SPHERE_RADIUS = 12.538
CAMERA_SPHERE_RADIUS = 14
SYSTEM_DIAMETER = 2

C_OFFSET_LOWER, C_OFFSET_UPPER = 0.1, 20
P_OFFSET_LOWER, P_OFFSET_UPPER = 0.1, 20
COUNT_LOWER, COUNT_UPPER = 20, 300
P_SCALE_RAND_LOWER, P_SCALE_RAND_UPPER = 0, 0.3

GAUSS_PARTICLE_WIDTH = True  # use gaussian distribution for particle with randomization
# Gaussian Params (only needed when using gausian)
MU = 0.1
SIGMA = 0.1

# deletes default cube
bpy.ops.object.delete()

# deletes default light
bpy.data.objects['Light'].select_set(True)
bpy.ops.object.delete()


# returns random point on half sphere of radius r:
def pick_point_on_sphere(r):
    # pick x, y, z such that: x^2 + y^2 + z^2 = r^2

    # -r <= z <= r; uniform dist
    z = random.uniform(-r, r)

    xsq = random.uniform(0, r ** 2 - z ** 2)
    ysq = r ** 2 - z ** 2 - xsq

    x = math.sqrt(xsq)
    y = math.sqrt(ysq)

    # randomizing sign (otherwise all choices in one quadrant)
    x *= random.choice([-1, 1])
    y *= random.choice([-1, 1])

    return (x, y, z)


# method for making a roundy cube scaled to size.
def make_roundy(offset, x_size, y_size, z_size, material, name='Obj'):
    # create a new cube
    bpy.ops.mesh.primitive_cube_add()
    obj = bpy.context.active_object
    obj.name = 'TempObj'  # renames

    # change to edit mode
    bpy.ops.object.mode_set(mode='EDIT')

    # subdivides active object
    bpy.ops.mesh.subdivide(number_cuts=8, smoothness=0.0, ngon=True, quadcorner='STRAIGHT_CUT', fractal=0.0,
                           fractal_along_normal=0.0, seed=0)

    # change to object mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # duplicating central
    old_obj = obj
    bpy.ops.object.duplicate()
    obj = bpy.context.active_object

    obj.scale = (5, 5, 5)

    # add shrinkwrap modifier
    bpy.ops.object.modifier_add(type='SHRINKWRAP')
    obj.modifiers['Shrinkwrap'].target = old_obj
    obj.modifiers['Shrinkwrap'].offset = offset
    bpy.ops.object.modifier_apply(modifier='Shrinkwrap')

    # delete old_obj
    bpy.ops.object.select_all(action='DESELECT')
    old_obj.select_set(True)  # selects old obj
    bpy.ops.object.delete()  # deletes old_obj

    obj.select_set(True)  # selects obj
    bpy.ops.object.shade_smooth()

    # change to edit mode
    bpy.ops.object.mode_set(mode='EDIT')

    # subdivides active object
    bpy.ops.mesh.subdivide(number_cuts=8, smoothness=0.0, ngon=True, quadcorner='STRAIGHT_CUT', fractal=0.0,
                           fractal_along_normal=0.0, seed=0)

    # change to object mode
    bpy.ops.object.mode_set(mode='OBJECT')

    obj.dimensions = (x_size, y_size, z_size)
    obj.name = name
    obj.data.materials.append(material)

    return obj


# rotates camera to look at point
def look_at(camera, point):
    direction = camera.location - point
    rot_quat = direction.to_track_quat('Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()


# Camera Initialization
camera_obj = bpy.data.objects['Camera']
camera_set = bpy.data.cameras[0]
camera_set.sensor_width = 26
camera_set.lens = 150
camera_set.lens_unit = 'MILLIMETERS'

# Scene Initialization
scene = bpy.data.scenes['Scene']
scene.render.resolution_x = 200
scene.render.resolution_y = 200
scene.render.image_settings.color_mode = 'BW'
scene.eevee.taa_render_samples=EEVEE_RENDER_SAMP

# Material Initialization
central_mat = bpy.data.materials.new(name='CentralMaterial')
particle_mat = bpy.data.materials.new(name='ParticleMaterial')

# Lighting Initialization
light_data = bpy.data.lights.new(name="Plight1000W", type='POINT')
light_data.energy = 1000

light1 = bpy.data.objects.new(name="Light1", object_data=light_data)
bpy.context.collection.objects.link(light1)

light2 = bpy.data.objects.new(name="Light2", object_data=light_data)
bpy.context.collection.objects.link(light2)

light3 = bpy.data.objects.new(name="Light3", object_data=light_data)
bpy.context.collection.objects.link(light3)

# World Color Initialization
world = bpy.data.worlds['World']
world.use_nodes = True
bg = world.node_tree.nodes['Background']
bg.inputs[0].default_value = (1, 1, 1, 1)

# DataFrame initialization
column_names = ['file_name',
                'c_x', 'c_y', 'c_z', 'c_offset',
                'p_l', 'p_wh', 'p_offset',
                'count', 'scale_rand']
data = []

# Randomization:
for i in range(START_AT, START_AT+DATA_SIZE):
    # Variables RNG
    c_x = random.uniform(1, 1.5)
    c_y = random.uniform(1, 1.5)
    c_z = random.uniform(1, 1.5)
    c_max = max([c_x, c_y, c_z])

    # p_l: particle length, p_wh: particle width-height
    p_l = 2 - c_max
    p_wh = 0
    if GAUSS_PARTICLE_WIDTH:
        p_wh = abs(random.gauss(MU, SIGMA))
    else:
        p_wh = random.uniform(0, p_l)

    c_offset = random.uniform(C_OFFSET_LOWER, C_OFFSET_UPPER)
    p_offset = random.uniform(P_OFFSET_LOWER, P_OFFSET_UPPER)

    particle_count = random.randint(COUNT_LOWER, COUNT_UPPER)
    scale_rand = random.uniform(P_SCALE_RAND_LOWER, P_SCALE_RAND_UPPER)

    # Mesh Generation
    central = make_roundy(c_offset, c_x, c_y, c_z, central_mat, 'Central' + str(i))
    particle = make_roundy(p_offset, p_wh, p_l, p_wh, particle_mat, 'Particle' + str(i))

    bpy.ops.object.select_all(action='DESELECT')

    # Particle System Generation
    bpy.context.view_layer.objects.active = central
    bpy.ops.object.particle_system_add()

    particle_sys_settings = central.particle_systems[0].settings

    # changing settings
    particle_sys_settings.type = 'HAIR'
    particle_sys_settings.render_type = 'OBJECT'
    particle_sys_settings.instance_object = particle
    particle_sys_settings.particle_size = 1
    particle_sys_settings.size_random = scale_rand  # how random size is
    particle_sys_settings.hair_length = p_l * 2  # length of particle
    particle_sys_settings.count = particle_count  # how many particles

    # Material Randomization
    c_tone = random.uniform(0.1, 1)
    p_tone = random.uniform(0.1, 1)

    central_mat.diffuse_color = (c_tone, c_tone, c_tone, 1)
    particle_mat.diffuse_color = (p_tone, p_tone, p_tone, 1)

    c_rough = random.uniform(0.2, 1)
    p_rough = random.uniform(0.2, 1)

    central_mat.roughness = c_rough
    particle_mat.roughness = p_rough

    # Camera Randomization
    camera_obj.location = pick_point_on_sphere(CAMERA_SPHERE_RADIUS)
    look_at(camera_obj, mathutils.Vector((0, 0, 0)))

    # Lights Randomization
    light1.location = pick_point_on_sphere(CAMERA_SPHERE_RADIUS)
    light2.location = pick_point_on_sphere(CAMERA_SPHERE_RADIUS)
    light3.location = pick_point_on_sphere(CAMERA_SPHERE_RADIUS)

    particle.location = (10000000, 0, 0)  # moving it out of view

    # World Background Color Randomization
    bg.inputs[1].default_value = random.uniform(0, 0.4)

    # Updates scene just in case
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()

    # Render; not sure if it's good
    file_name = 'img' + str(i) + '.png'
    bpy.context.scene.render.filepath = GLOBAL_OUTPUT_PATH + file_name
    bpy.ops.render.render(write_still=True)

    # Delete central and particle
    central.select_set(True)
    bpy.ops.object.delete()

    particle.select_set(True)
    bpy.ops.object.delete()

    # Appending to parameters to DataFrame
    row = {'file_name': file_name,
           'c_x': c_x, 'c_y': c_y, 'c_z': c_z, 'c_offset': c_offset,
           'p_l': p_l, 'p_wh': p_wh, 'p_offset': p_offset,
           'count': particle_count, 'scale_rand': scale_rand}
    data.append(row)

keys = data[0].keys()
with open('out.csv', 'w', encoding='utf8', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, fieldnames=keys)
    dict_writer.writeheader()
    dict_writer.writerows(data)
# end

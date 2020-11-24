import bpy

import sys
import os

args = sys.argv
args = args[args.index("--") + 1:]  # blender ignores arguments after a separate '--'

assert len(args) == 10  # NEEDS to be 9

output_name = args.pop(0)  # e.g. 'my_cactus.obj'

args = [float(a) for a in args]  # converting strings to float

# getting particle system parameters
c_x = args[0]
c_y = args[1]
c_z = args[2]
c_offset = args[3]
p_l = args[4]
p_wh = args[5]
p_offset = args[6]
count = args[7]
scale_rand = args[8]

# deletes default cube
bpy.ops.object.delete()

# deletes default light
bpy.data.objects['Light'].select_set(True)
bpy.ops.object.delete()


# method for making a roundy cube scaled to size.
def make_roundy(offset, x_size, y_size, z_size, material=None, name='Obj'):
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

    if material is not None:
        obj.data.materials.append(material)

    return obj


# making objects:
central = make_roundy(c_offset, c_x, c_y, c_z, None, 'Central')
particle = make_roundy(p_offset, p_wh, p_l, p_wh, None, 'Particle')

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
particle_sys_settings.count = count  # how many particles

bpy.ops.object.select_all(action='DESELECT')
bpy.context.view_layer.objects.active = central

# saving to obj
blend_file_path = bpy.data.filepath
directory = os.path.dirname(blend_file_path)
output_name = output_name if output_name.endswith('.obj') else output_name + '.obj'
target_file = os.path.join(directory, output_name)

bpy.ops.export_scene.obj(filepath=target_file, check_existing=True, use_selection=True)

from model import Model
import sys
import subprocess

BLENDER_PATH = '/Applications/Blender.app/Contents/MacOS/Blender'  # CHANGE IF BLENDER IS NOT IN PATH
GENERATOR_SCRIPT_PATH = 'generator.py'

BLENDER_PATH = 'blender' if BLENDER_PATH == '' else BLENDER_PATH

args = sys.argv
args = args[1:] # to exclude 'hairnet.py'

if len(args) != 2:
    print('Invalid arguments: need image path and output name')
    print(args)
    sys.exit(1)

model = Model()  # encapsulates PyTorch
params = model.pass_image(args[0])  # image to params

print('model params: ', str(params))

# building the command for calling blender.
# cli_args = [BLENDER_PATH, '--background', '--python', GENERATOR_SCRIPT_PATH, '--', args[1]]

# remove background option for debug:
cli_args = [BLENDER_PATH, '--python', GENERATOR_SCRIPT_PATH, '--', args[1]]

cli_args.extend([str(p) for p in params])

print('executing: ', str(cli_args))

# executing the command
subprocess.run(cli_args)

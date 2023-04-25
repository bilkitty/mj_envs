import sys
from mj_envs_vision.utils import helpers


DPI = 200
if len(sys.argv) != 3:
    print("Usage:\n\t<full_path_to_gif_list_txt> <grid length>")

paths = [x.replace('\n', '') for x in open(sys.argv[1], 'r').readlines()]
helpers.grid_gif(paths, int(sys.argv[2]), False).savefig(sys.argv[1].replace('txt', 'png'), bbox_inches='tight', dpi=DPI)

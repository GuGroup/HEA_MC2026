import matplotlib.pyplot as plt
import json
import os

dir_path = '../Data/'

exp_map_viz = json.load(open(os.path.join(dir_path, 'exp_scaled_activity_2D.json'), 'r'))
MC_patch_viz = json.load(open(os.path.join(dir_path, 'MC_activity_patch_2D.json'), 'r'))
RS_patch_viz = json.load(open(os.path.join(dir_path, 'RS_activity_patch_2D.json'), 'r'))

# Figure 1a
plt.figure()
plt.imshow(exp_map_viz, cmap='bwr_r', origin='upper', vmin=0, vmax=1)
# plt.title('Experiment')
plt.axis('off')
# plt.savefig('Experiment_1000K_dpi_300', dpi=300, bbox_inches='tight')
plt.show()

plt.figure()
plt.imshow(MC_patch_viz, cmap='bwr_r', origin='upper', vmin=0, vmax=1)
# plt.title('MC Annealed Slab')
plt.axis('off')
# plt.savefig('MC_1000K_dpi_300', dpi=300, bbox_inches='tight')
plt.show()

plt.figure()
plt.imshow(RS_patch_viz, cmap='bwr_r', origin='upper', vmin=0, vmax=1)
# plt.title('Homogeneous Slab')
plt.axis('off')
# plt.savefig('RS_1000K_dpi_300', dpi=300, bbox_inches='tight')
plt.show()
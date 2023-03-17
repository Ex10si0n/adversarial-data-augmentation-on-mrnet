import os

curve_dir = './curve/baseline_adv_1_adv_2'
plot_task = curve_dir.split('/')[-1] + '/'

graphs = ['train-accuracy', 'train-loss', 'train-specificity', 'val-auc', 'val-sensitivity', 'train-auc', 'train-sensitivity', 'val-accuracy', 'val-loss', 'val-specificity', 'auc', 'sensitivity', 'accuracy', 'loss', 'specificity']
dim = ['coronal', 'sagittal', 'axial']
task = ['acl', 'meniscus', 'abnormal']

curves = os.listdir(curve_dir)
folders = os.listdir(curve_dir)
folders.remove('.DS_Store')

from PIL import Image
import os

def combine_images(images, output_folder='./curve/combined/' + plot_task, output_name='combined.png'):
    resol = (640, 480)
    res_resol = (640*3, 480*3)
    combined_image = Image.new('RGB', res_resol)

    for i in range(3):
        for j in range(3):
            index = i * 3 + j
            if index >= len(images):
                break
            image = Image.open(images[index])
            combined_image.paste(image, (j * resol[0], i * resol[1]))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    combined_image.save(output_folder + output_name + '.png', format='png')
    print('Combined image saved to ' + output_folder + output_name + '.png')

for g in graphs:
    g_name = g + '.png'
    g_files = []
    for folder in folders:
        t, d = folder.split('_')
        curve_path = os.path.join(curve_dir, folder)
        if g_name in os.listdir(curve_path):
            g_files.append(os.path.join(curve_path, g_name))
    combine_images(g_files, output_name=g)

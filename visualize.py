import os
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import scipy.ndimage
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plotly.figure_factory import create_trisurf
from plotly.graph_objects import Layout
from tqdm import tqdm


IMAGE_DIR = '89601'
NPZ_FILE = './89601.npy'


def load_scan(dir_path):
    dicomfiles = os.listdir(dir_path)
    slices = [pydicom.read_file(os.path.join(dir_path, df)) for df in dicomfiles]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    return slices


def get_hu_array(slices):
    npimg = np.stack([s.pixel_array for s in tqdm(slices)]).astype(np.int16)
    npimg[npimg == -2000] = 0

    for i, s in enumerate(slices):
        intercept = s.RescaleIntercept
        slope = s.RescaleSlope
        npimg[i] = (slope * npimg[i] + intercept).astype(np.int16)

    return npimg


def get_spacing(slices):
    s = slices[0]
    sxy = s.PixelSpacing
    sz = s.SliceThickness
    return np.array([sz] + list(sxy), dtype=np.float32)


def resample(image, old_spacing, new_spacing):
    resize_factor = old_spacing / new_spacing
    new_shape = np.round(image.shape * resize_factor)

    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    return image, new_spacing


def generate_mesh(image, threshold=-300, step_size=1):
    p = image.transpose(0, 2, 1)
    p = p[:, ::-1, ::-1]

    # p = image.transpose(2, 1, 0)
    # p = p[::-1, ::-1, :]

    verts, faces, _, _ = measure.marching_cubes(p, threshold, step_size=step_size, allow_degenerate=True)
    return verts, faces


def segment_lung_mask(image, fill_lung_structures=True):
    binary_image = np.array(image > -320, dtype=np.int8) + 1        # make 1 and 2
    labels = measure.label(binary_image)

    background_label = labels[0, 0, 0]
    binary_image[labels == background_label] = 2

    if fill_lung_structures:
        for i in range(binary_image.shape[0]):
            axial_slice = binary_image[i] - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, background=0)
            if l_max is not None:
                binary_image[i, labeling != l_max] = 1

    binary_image = binary_image - 1     # make it binary (0 and 1)
    binary_image = 1 - binary_image     # invert it --> lung is 1

    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, background=0)
    if l_max is not None:
        binary_image[labels != l_max] = 0

    return binary_image


def largest_label_volume(image, background=-1):
    vals, counts = np.unique(image, return_counts=True)

    counts = counts[vals != background]
    vals = vals[vals != background]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def plot_3d(verts, faces):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces], alpha=0.50)
    face_color = [0.7, 0.7, 1.0]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    # x, y, z = zip(*verts)
    # ax.set_xlim(min(x), max(x))
    # ax.set_ylim(min(y), max(y))
    # ax.set_zlim(min(z), max(z))

    mi, ma = np.min(verts), np.max(verts)
    ax.set_xlim(mi, ma)
    ax.set_ylim(mi, ma)
    ax.set_zlim(mi, ma)

    plt.show()


def plotly_3d(verts, faces, outfile=None):
    x, y, z = zip(*verts)

    colormap = ['rgba(190, 190, 235, 0.5)', 'rgba(190, 190, 235, 0.5)']

    layout = Layout(
        width=1024, height=1024,
        scene=dict(
            xaxis=dict(range=[0, max(x)]),
            yaxis=dict(range=[0, max(y)]),
            zaxis=dict(range=[0, max(z)])
        )
    )

    fig = create_trisurf(
        x=x, y=y, z=z, simplices=faces,
        colormap=colormap, backgroundcolor='rgba(64, 64, 64, 1.0)', plot_edges=False
    )

    fig.update_layout(layout)

    if outfile is not None:
        fig.write_html(outfile)

    fig.show()


if __name__ == '__main__':

    # 1. load pixel array
    if not os.path.exists(NPZ_FILE):
        slices = load_scan(IMAGE_DIR)
        print(slices[0].PatientID, slices[0].PatientName)
        npimg = get_hu_array(slices)
        np.save(NPZ_FILE, npimg)
    else:
        npimg = np.load(NPZ_FILE)

    # 2. plot
    plt.style.use('dark_background')
    plt.hist(npimg.flatten(), bins=50)
    plt.xlabel('Hounsfield Units (HU)')
    plt.ylabel('Frequency')
    plt.show()

    plt.imshow(npimg[150], cmap=plt.cm.gray)
    plt.show()
    print(npimg.shape)

    # 3. resample
    print('resampling ... ', end='', flush=True)
    spacing = get_spacing(slices)
    npimg_re, new_spacing = resample(npimg, spacing, [1, 1, 1])
    print('done')
    print(spacing, '-->', new_spacing)
    print(npimg.shape, '-->', npimg_re.shape)

    # 4. generate mesh
    print('generating mesh ... ', end='', flush=True)
    v, f = generate_mesh(npimg_re, 400)
    print('done')
    print(v.shape, f.shape)

    # 5. plot 3d mesh
    print('plotting ... ', end='', flush=True)
    plot_3d(v, f)
    plotly_3d(v, f)
    print('done')

    # 6. segment lung
    print('segmenting & plotting ... ', end='', flush=True)
    npimg_seg = segment_lung_mask(npimg_re, False)
    v, f = generate_mesh(npimg_seg, 0)
    plot_3d(v, f)
    print('done')

    print('segmenting & plotting ... ', end='', flush=True)
    npimg_seg_fill = segment_lung_mask(npimg_re, True)
    v, f = generate_mesh(npimg_seg_fill, 0)
    plot_3d(v, f)
    plotly_3d(v, f)
    print('done')

    print('final plotting ... ', end='', flush=True)
    v, f = generate_mesh(npimg_seg_fill - npimg_seg, 0)
    plot_3d(v, f)
    print('done')

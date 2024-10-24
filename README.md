# Gaussian Surfels for waymo

## Environment Setup
Test experiments on Ubuntu 22.04.3, CUDA 12.1 and conda environment on Python 3.9.

Default config is 5cam, set by --cameras parameter in train.py and render.py. You can change the number of cameras by changing the parameter.

> **Note:** some sky_mask of SIDE_RIGHT cameras is wrong, 5cam setting may cause depth2rgb function crash.

Clone this repository:
```shell
git clone https://github.com/turandai/gaussian_surfels.git
cd gaussian_surfels
```

Create conda environment:
```shell
conda env create --file environment.yml
conda activate gaussian_surfels
```

Need to recompile and reinstall the CUDA rasterizer, simple-knn and pytorch3d
```shell
cd submodules/diff-gaussian-rasterization
python setup.py install && pip install .

cd submodules/simple-knn
python setup.py install && pip install .

# nvcc -V版本需要显示12.1 如果为否则调整bashrc中的cuda相关环境变量与软连接
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

```

## Data
```bash
# In PJ lab
cd /home/petrelfs/wuguofeng/waymo/waymo_train/002

```

## Training
To train a scene:
```shell
python train.py -s path/to/your/data/directory
```
Trained model will be save in ```output/```.
To render images and reconstruct mesh from a trianed model:
```shell
python render.py -m path/to/your/trained/model --img --depth 10
```
We use screened Poisson surface reconstruction to extract mesh, at this line in ```render.py```:
```python
poisson_mesh(path, wpos, normal, color, poisson_depth, prune_thrsh)
```
In your output folder, ```xxx_plain.ply``` is the original mesh after Poisson reconstruction with the default depth of 10. For scenes in larger scales, you may use a higher depth level. For a smoother mesh, you may decrease the depth value.
We prune the Poisson mesh with a certain threshold to remove outlying faces and output ```xxx_pruned.ply```. This process sometimes may over-prune the mesh and results in holes. You may increase the "prune_thrsh" parameter accordingly.


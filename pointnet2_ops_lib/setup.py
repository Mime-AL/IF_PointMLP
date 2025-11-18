import glob
import os
import os.path as osp
import sys

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 自定义BuildExtension类来处理编码问题
class CustomBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 设置环境变量避免ninja编码问题
        os.environ["PYTHONIOENCODING"] = "utf-8"
        if sys.platform.startswith('win'):
            os.environ["PYTHONUTF8"] = "1"

this_dir = osp.dirname(osp.abspath(__file__))
_ext_src_root = osp.join("pointnet2_ops", "_ext-src")
_ext_sources = glob.glob(osp.join(_ext_src_root, "src", "*.cpp")) + glob.glob(
    osp.join(_ext_src_root, "src", "*.cu")
)
_ext_headers = glob.glob(osp.join(_ext_src_root, "include", "*"))

requirements = ["torch>=1.4"]

# 安全地读取版本文件，避免编码问题
version_file = osp.join("pointnet2_ops", "_version.py")
__version__ = None
try:
    with open(version_file, 'r', encoding='utf-8') as f:
        version_content = f.read()
        exec(version_content)
except UnicodeDecodeError:
    with open(version_file, 'r', encoding='latin-1') as f:
        version_content = f.read()
        exec(version_content)

# 如果仍然无法读取版本，使用默认值
if __version__ is None:
    __version__ = "3.0.0"

# 设置环境变量以避免ninja编码问题
os.environ["TORCH_CUDA_ARCH_LIST"] = "5.0;6.0;6.1;6.2;7.0;7.5;8.0;8.6"
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.platform.startswith('win'):
    # Windows特定的编码设置
    os.environ["PYTHONUTF8"] = "1"
setup(
    name="pointnet2_ops",
    version=__version__,
    author="Erik Wijmans",
    packages=find_packages(),
    install_requires=requirements,
    ext_modules=[
        CUDAExtension(
            name="pointnet2_ops._ext",
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-Xfatbin", "-compress-all"],
            },
            include_dirs=[osp.join(this_dir, _ext_src_root, "include")],
        )
    ],
    cmdclass={"build_ext": CustomBuildExtension},
    include_package_data=True,
)

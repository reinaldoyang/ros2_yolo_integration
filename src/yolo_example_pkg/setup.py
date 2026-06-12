from setuptools import find_packages, setup
from glob import glob
import os

package_name = "yolo_example_pkg"
model_files = sorted(path for path in glob("models/*.pt") if os.path.isfile(path))

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "models"), model_files),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="alianlbj23@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "yolo_node = yolo_example_pkg.object_detect:main",
            "yolo_detection_node = yolo_example_pkg.yolo_detection_node:main",
            "yolo_segmentation_node = yolo_example_pkg.yolo_segmentation_node:main",
            "semantic_costmap_node = yolo_example_pkg.semantic_costmap_node:main",
        ],
    },
)

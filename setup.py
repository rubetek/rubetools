import os
from setuptools import find_packages, setup

requires = [
    'tqdm~=4.56.0',
    'opencv-python~=4.5.1.48',
    'pandas==1.1.5',
    'xmltodict~=0.12.0',
    'pillow~=8.1.0',
    'shapely~=1.7.1',
    'matplotlib~=3.3.4'
]


def setup_package():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    about = {}
    with open(os.path.join(base_dir, 'rubetools', '__version__.py'), 'r') as f:
        exec(f.read(), about)

    setup(
        name=about['__title__'],
        version=about['__version__'],
        author=about['__author__'],
        description=about['__description__'],
        url=about['__url__'],
        license=about['__license__'],
        keywords=about['__keywords__'],
        include_package_data=True,
        python_requires='>=3.6',
        install_requires=requires,
        packages=find_packages(include=['rubetools', 'rubetools.*'])
    )


if __name__ == '__main__':
    setup_package()

# python setup.py sdist

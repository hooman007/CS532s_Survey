"""Install package."""
from setuptools import setup, find_packages
setup(
    name='Audio_Visual_Speech_Recognition',
    version='0.1',
    description=(
        'Survey of multimodal audio-visual speech recognition algorithms'
    ),
    long_description=open('README.md').read(),
    url='https://github.com/hooman007/CS532s_Survey',
    install_requires=[
        'numpy'
    ],
    packages=find_packages('.'),
    zip_safe=False,
)
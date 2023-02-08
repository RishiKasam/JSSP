from setuptools import setup, find_packages

setup(name='JsspEnvironment',
      version='0.0.9',
      author="Deepak Vivekanandan",
      author_email="Deepak.vivekanandan@th-Rosenheim",
      description="Job-Shop Scheduling problem using Deep Reinforcement Learning",
      packages=find_packages(),
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ],
      python_requires='==3.9',
      install_requires=['gym', 'pandas', 'numpy', 'imageio', 'psutil', 'requests', 'kaleido', 'pytest',
                        'codecov', 'stable_baselines3','tensorboard','scikit-optimize', 'matplotlib','pillow'],
      include_package_data=True
      )
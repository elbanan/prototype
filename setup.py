from setuptools import setup, find_packages

from datetime import datetime

now = datetime.now()

setup(
      name='radtorch',
      version=now,
      version_date='',
      description='RADTorch, The Radiology Machine Learning Framework',
      url='https://www.radtorch.com',
      author='Mohamed Elbanan, MD',
      author_email = "https://www.linkedin.com/in/mohamedelbanan/",
      license='GNU Affero General Public License v3.0 License',
      packages=find_packages(),
      install_requires=[
      'imageio==2.9.0',
      'imagesize==1.2.0',
      'matplotlib==3.4.2',
      'matplotlib-inline==0.1.2',
      'numpy==1.19.5',
      'opencv-python==4.5.3.56',
      'pandas==1.3.0',
      'Pillow==8.3.1',
      'pydicom==2.1.2',
      'scikit-image==0.18.2',
      'scikit-learn==0.24.2',
      'scipy==1.7.0',
      'seaborn==0.11.1',
      'torch==1.9.0',
      'torchinfo==1.5.3',
      'torchvision==0.10.0',
      'tqdm==4.61.2',
      'urllib3==1.26.6',
      'wrapt==1.12.1',
      'xgboost==1.4.2',
                       ],

      zip_safe=False,
      classifiers=[
      "License :: OSI Approved :: GNU Affero General Public License v3.0 License",
      "Natural Language :: English",
      "Programming Language :: Python :: 3 :: Only",
      "Topic :: Software Development :: Libraries :: Python Modules",
      ]
      )

from setuptools import setup, find_packages

setup(
    name='gamesbiz',
    version='0.0.1rc0',

    # Package data
    packages=find_packages(),
    include_package_data=True,

    # Insert dependencies list here
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'scikit-learn',
        'tensorflow',
        'flask',
        'gunicorn',
        'gevent'
    ],

    entry_points={
        "gamesbiz.training": [
           "train=gamesbiz.train:entry_point"
        ],
        "gamesbiz.hosting": [
           "serve=gamesbiz.server:start_server"
        ]
    }
)
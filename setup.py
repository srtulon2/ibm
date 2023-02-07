from setuptools import setup

setup(name='imbalanced_metrics',
      version='0.3.1',
      description='Perfromance metrics for imbalanced classification and imbalanced regression tasks',
      url='https://github.com/paobranco/ImbalancedDomainsMetrics',
      authors=['Jean-Gabriel Gaudreault','Paula Branco','Sadid Rafsun Tulon'],
      packages=['imbalanced_metrics'],
      install_requires=[
          'scikit-learn', 'hmeasure', 'pyprg','numpy','smogn',
      ],
      zip_safe=False)

import setuptools

install_requires = [
    'numpy',
    'pandas',
    'transformers',
    'seaborn',
    'nltk',
    'sentence_transformers',
    'gensim',
    'itranslate',
    'torch',
    'keybert',
    'streamlit'
]

setuptools.setup(
    name='Contextual-Advertising-Demo',
    version='0.0.2',
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    python_requires='>=3.7',
    author='Marya Dukmak',
    author_email='Marya.dukmak@groupm.com',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    include_package_data=True,
)
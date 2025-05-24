from setuptools import setup, find_packages

setup(
    name="pm_kvq",
    version="0.1.0",
    author="Tengxuan Liu",
    author_email="liutx21@mails.tsinghua.edu.cn",
    description="PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

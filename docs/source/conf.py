# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# 添加项目根目录到 Python 路径，这样 Sphinx 可以找到你的模块
sys.path.insert(0, os.path.abspath('../..'))  # 项目根目录
sys.path.insert(0, os.path.abspath('../../openddi'))  # openddi 包目录

project = 'OpenDDI'
copyright = '2025, xiaoriwuguang, bwfan-bit, ZyxAdenine'
author = 'xiaoriwuguang, bwfan-bit, ZyxAdenine'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# 添加必要的扩展
extensions = [
    'sphinx.ext.autodoc',      # 自动从代码生成文档
    'sphinx.ext.viewcode',     # 添加源代码链接
    'sphinx.ext.napoleon',     # 支持 Google/NumPy 风格的文档字符串
]

templates_path = ['_templates']
exclude_patterns = []

# -- Autodoc 配置 ---------------------------------------------------------
# 自动包含所有成员，包括未文档化的
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# 自动为每个模块生成文档
autodoc_mock_imports = []

# -- Napoleon 配置 (用于文档字符串解析) ------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

# Alabaster 主题配置 (可选)
html_theme_options = {
    'description': 'OpenDDI: 开源药物相互作用预测框架',
    'github_user': 'ZyxAdenine',
    'github_repo': 'OpenDDI',
    'fixed_sidebar': True,
}
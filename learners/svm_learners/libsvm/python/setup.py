from distutils.core import setup, Extension


svmc_module = Extension('_svmc',
                           sources=['svmc_wrap.c', '../svm.cpp'],
                           )

setup (name = 'libsvm',
       version = '0.1',
       author      = "byron wallace",
       description = """(modified) libsvm""",
       ext_modules = [svmc_module],
       py_modules = ["svmc"],
       )

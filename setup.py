# from future.utils import iteritems
import os
import sys
from os.path import join as pjoin
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import argparse

def find_in_path(name, path):
    """Find a file in a search path"""
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

def locate_cuda():
    """Locate the CUDA environment on the system.
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'.
    """
    if "CUDAHOME" in os.environ:
        home = os.environ["CUDAHOME"]
        nvcc = pjoin(home, "bin", "nvcc")
    elif "CUDA_HOME" in os.environ:
        home = os.environ["CUDA_HOME"]
        nvcc = pjoin(home, "bin", "nvcc")
    else:
        nvcc = find_in_path("nvcc", os.environ["PATH"])
        if nvcc is None:
            raise EnvironmentError(
                "The nvcc binary could not be located in your $PATH. "
                "Either add it to your path, or set $CUDAHOME"
            )
        home = os.path.dirname(os.path.dirname(nvcc))
    cudaconfig = {
        "home": home,
        "nvcc": nvcc,
        "include": pjoin(home, "include"),
        "lib64": pjoin(home, "lib64"),
    }
    for k, v in iter(cudaconfig.items()):
        if not os.path.exists(v):
            raise EnvironmentError("The CUDA %s path could not be located in %s" % (k, v))
    return cudaconfig

def customize_compiler_for_nvcc(self):
    self.cuda_object_files = []
    self.src_extensions.append(".cu")
    default_compiler_so = self.compiler_so
    super_compile = self._compile
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if src == "zzzzzzzzzzzzzzzz.cu":
            self.set_executable("compiler_so", CUDA["nvcc"])
            postargs = extra_postargs["nvcclink"]
            cc_args = self.cuda_object_files[1:]
            src = self.cuda_object_files[0]
        elif os.path.splitext(src)[1] == ".cu":
            self.set_executable("compiler_so", CUDA["nvcc"])
            postargs = extra_postargs["nvcc"]
            self.cuda_object_files.append(obj)
        else:
            postargs = extra_postargs["gcc"]
        super_compile(obj, src, ext, cc_args, postargs, pp_opts)
        self.compiler_so = default_compiler_so
    self._compile = _compile

class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

# Proviamo a localizzare CUDA
try:
    CUDA = locate_cuda()
    run_cuda_install = True
except OSError:
    print("Unable to locate CUDA")
    print("Check if nvcc compiler is installed")
    run_cuda_install = False
run_cuda_install = False

parser = argparse.ArgumentParser()
parser.add_argument("--lapack_lib", help="Directory of the lapack lib.", default="/usr/local/opt/lapack/lib")
parser.add_argument("--lapack_include", help="Directory of the lapack include.", default="/usr/local/opt/lapack/include")
parser.add_argument("--lapack", help="Directory of both lapack lib and include. '/include' and '/lib' will be added to the end of this string.")
parser.add_argument("--gsl_lib", help="Directory of the gsl lib.", default="/usr/local/opt/gsl/lib")
parser.add_argument("--gsl_include", help="Directory of the gsl include.", default="/usr/local/opt/gsl/include")
parser.add_argument("--gsl", help="Directory of both gsl lib and include. '/include' and '/lib' will be added to the end of this string.")
args, unknown = parser.parse_known_args()
for key in [
    args.gsl_include, args.gsl_lib, args.gsl,
    "--gsl", "--gsl_include", "--gsl_lib",
    args.lapack_include, args.lapack_lib, args.lapack,
    "--lapack", "--lapack_lib", "--lapack_include",
]:
    try:
        sys.argv.remove(key)
    except ValueError:
        pass

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# Configurazione delle directory LAPACK
lapack_dir = os.environ.get("LAPACK_DIR")
if lapack_dir is None:
    print("\nWARNING: unable to find lapack\n")
    print("------------------------------------------------------------")
    print("LAPACK_DIR environment variable is not set. Please set it to the directory of the lapack installation.")
    print("For example, on CentOS, you might set it to /usr")
    print("export LAPACK_DIR=/usr")
    print("------------------------------------------------------------\n")
    raise ValueError("LAPACK_DIR environment variable is not set. Please set it to the directory of the lapack installation.")

# Determina la directory delle librerie: preferisce lib64 se esiste
if os.path.exists(pjoin(lapack_dir, "lib64")):
    lapack_lib = [pjoin(lapack_dir, "lib64")]
elif os.path.exists(pjoin(lapack_dir, "lib")):
    lapack_lib = [pjoin(lapack_dir, "lib")]
else:
    lapack_lib = [pjoin(lapack_dir, "lib")]

# Determina la directory degli header LAPACK
if os.path.exists(pjoin(lapack_dir, "include", "lapacke", "lapacke.h")):
    lapack_include = [pjoin(lapack_dir, "include", "lapacke")]
elif os.path.exists(pjoin(lapack_dir, "include", "lapacke.h")):
    lapack_include = [pjoin(lapack_dir, "include")]
elif os.path.exists(pjoin(lapack_dir, "lapacke.h")):
    lapack_include = [lapack_dir]
else:
    lapack_include = [pjoin(lapack_dir, "include")]

if args.gsl is None:
    gsl_include = [args.gsl_include]
    gsl_lib = [args.gsl_lib]
else:
    gsl_include = [args.gsl + "/include"]
    gsl_lib = [args.gsl + "/lib"]

import lisatools
path_to_lisatools = lisatools.__file__.split("__init__.py")[0]
path_to_lisatools_cutils = path_to_lisatools + "cutils/"

# Costruzione delle estensioni
if run_cuda_install:
    gpu_extension = dict(
        libraries=["cudart", "cublas", "cusparse", "gsl", "gslcblas"],
        library_dirs=[CUDA["lib64"]] + gsl_lib,
        runtime_library_dirs=[CUDA["lib64"]],
        language="c++",
        extra_compile_args={
            "gcc": ["-std=c++11", "-fpermissive"],
            "nvcc": [
                "-arch=sm_80",
                "-gencode=arch=compute_60,code=sm_60",
                "-gencode=arch=compute_61,code=sm_61",
                "-gencode=arch=compute_70,code=sm_70",
                "-gencode=arch=compute_75,code=sm_75",
                "-gencode=arch=compute_80,code=compute_80",
                "-std=c++11",
                "-c",
                "--compiler-options",
                "'-fPIC'",
            ],
        },
        include_dirs=[
            numpy_include,
            CUDA["include"],
            "bbhx/cutils/include",
            "/usr/include",
        ],
    )

    pyPhenomHM_ext = Extension(
        "bbhx.cutils.pyPhenomHM",
        sources=["bbhx/cutils/src/PhenomHM.cu", "bbhx/cutils/src/phenomhm.pyx"],
        **gpu_extension,
    )
    pyFDResponse_ext = Extension(
        "bbhx.cutils.pyFDResponse",
        sources=[
            path_to_lisatools_cutils + "src/Detector.cu",
            "bbhx/cutils/src/Response.cu",
            "bbhx/cutils/src/response.pyx",
            "zzzzzzzzzzzzzzzz.cu",
        ],
        libraries=["cudart", "cudadevrt", "cublas", "cusparse"],
        library_dirs=[CUDA["lib64"]],
        runtime_library_dirs=[CUDA["lib64"]],
        language="c++",
        extra_compile_args={
            "gcc": ["-std=c++11", "-fpermissive"],
            "nvcc": ["-arch=sm_80", "-rdc=true", "--compiler-options", "'-fPIC'"],
            "nvcclink": [
                "-arch=sm_80",
                "--device-link",
                "--compiler-options",
                "'-fPIC'",
            ],
        },
        include_dirs=[
            numpy_include,
            CUDA["include"],
            "bbhx/cutils/include",
            path_to_lisatools_cutils + "include",
            "/usr/include",
        ],
    )
    pyInterpolate_ext = Extension(
        "bbhx.cutils.pyInterpolate",
        sources=["bbhx/cutils/src/Interpolate.cu", "bbhx/cutils/src/interpolate.pyx"],
        **gpu_extension,
    )
    pyWaveformBuild_ext = Extension(
        "bbhx.cutils.pyWaveformBuild",
        sources=[
            "bbhx/cutils/src/WaveformBuild.cu",
            "bbhx/cutils/src/waveformbuild.pyx",
        ],
        **gpu_extension,
    )
    pyLikelihood_ext = Extension(
        "bbhx.cutils.pyLikelihood",
        sources=["bbhx/cutils/src/Likelihood.cu", "bbhx/cutils/src/likelihood.pyx"],
        **gpu_extension,
    )

cpu_extension = dict(
    libraries=["lapacke", "lapack", "gsl", "gslcblas"],
    language="c++",
    library_dirs=lapack_lib,
    extra_compile_args={
        "gcc": ["-std=c++11"],
    },
    include_dirs=[
        numpy_include,
        "bbhx/cutils/include",
        path_to_lisatools_cutils + "include",
        "/usr/include",
    ] + lapack_include,
)

pyPhenomHM_cpu_ext = Extension(
    "bbhx.cutils.pyPhenomHM_cpu",
    sources=["bbhx/cutils/src/PhenomHM.cpp", "bbhx/cutils/src/phenomhm_cpu.pyx"],
    **cpu_extension,
)
pyFDResponse_cpu_ext = Extension(
    "bbhx.cutils.pyFDResponse_cpu",
    sources=[
        path_to_lisatools_cutils + "src/Detector.cpp",
        "bbhx/cutils/src/Response.cpp",
        "bbhx/cutils/src/response_cpu.pyx",
    ],
    **cpu_extension,
)
pyInterpolate_cpu_ext = Extension(
    "bbhx.cutils.pyInterpolate_cpu",
    sources=["bbhx/cutils/src/Interpolate.cpp", "bbhx/cutils/src/interpolate_cpu.pyx"],
    **cpu_extension,
)
pyWaveformBuild_cpu_ext = Extension(
    "bbhx.cutils.pyWaveformBuild_cpu",
    sources=[
        "bbhx/cutils/src/WaveformBuild.cpp",
        "bbhx/cutils/src/waveformbuild_cpu.pyx",
    ],
    **cpu_extension,
)
pyLikelihood_cpu_ext = Extension(
    "bbhx.cutils.pyLikelihood_cpu",
    sources=["bbhx/cutils/src/Likelihood.cpp", "bbhx/cutils/src/likelihood_cpu.pyx"],
    **cpu_extension,
)

extensions = [
    pyPhenomHM_cpu_ext,
    pyFDResponse_cpu_ext,
    pyInterpolate_cpu_ext,
    pyWaveformBuild_cpu_ext,
    pyLikelihood_cpu_ext,
]
if run_cuda_install:
    extensions = [
        pyPhenomHM_ext,
        pyFDResponse_ext,
        pyInterpolate_ext,
        pyWaveformBuild_ext,
        pyLikelihood_ext,
    ] + extensions

setup(
    name="bbhx",
    author="Michael Katz",
    author_email="mikekatz04@gmail.com",
    ext_modules=extensions,
    packages=[
        "bbhx",
        "bbhx.utils",
        "bbhx.waveforms",
        "bbhx.response",
        "bbhx.cutils",
        "bbhx.cutils.src",
        "bbhx.cutils.include",
    ],
    cmdclass={"build_ext": custom_build_ext},
    zip_safe=False,
    version="1.1.11",
    python_requires=">=3.6",
    package_data={
        "bbhx.cutils.src": [
            "Interpolate.cu",
            "Interpolate.cpp",
            "Likelihood.cu",
            "Likelihood.cpp",
            "PhenomHM.cu",
            "PhenomHM.cpp",
            "pycppdetector.pyx",
            "Response.cu",
            "Response.cpp",
            "WaveformBuild.cu",
            "WaveformBuild.cpp",
            "interpolate.pyx",
            "likelihood.pyx",
            "phenomhm.pyx",
            "response.pyx",
            "waveformbuild.pyx",
        ],
        "bbhx.cutils.include": [
            "Interpolate.hh",
            "Likelihood.hh",
            "PhenomHM.hh",
            "Response.hh",
            "WaveformBuild.hh",
            "constants.h",
            "cuda_complex.hpp",
            "global.h",
        ],
    },
)
## 1. Cython是什么?

Cython是一个编程语言，它通过类似Python的语法来编写C扩展并可以被Python调用.既具备了Python快速开发的特点，又可以让代码运行起来像C一样快，同时还可以方便地调用C library。

## 2. 如何安装Cython?

跟大多数的Python库不同，Cython需要一个C编译器，在不同的平台上配置方法也不一样。

### 2.1 配置gcc

- **windows**
  安装MingW-w64编译器：`conda install libpython m2w64-toolchain -c msys2`
  在Python安装路径下找到\Lib\distutils文件夹，创建distutils.cfg写入如下内容：
  `[build] compiler=mingw32`
- **macOS**
  安装XCode即可
- **linux:**
  gcc一般都是配置好的，如果没有就执行这条命令： `sudo apt-get install build-essential`

### 2.2 安装cython库

- 如果没安装Anaconda： `pip install cython`
- 如果安装了Anaconda： `conda install cython`



#### 通过静态类型更快的代码

Cython 是一个 Python 编译器。这意味着它可以在不进行更改的情况下编译普通的 Python 代码（除了一些尚未支持的语言功能的一些明显例外，请参阅 [Cython 限制](https://www.bookstack.cn/read/cython-doc-zh/$userguide-limitations.html#cython-limitations) ）。但是，对于性能关键代码，添加静态类型声明通常很有用，因为它们将允许 Cython 脱离 Python 代码的动态特性并生成更简单，更快速的 C 代码 - 有时会快几个数量级。

但必须注意，类型声明可以使源代码更加冗长，从而降低可读性。因此，不鼓励在没有充分理由的情况下使用它们，例如基准测试证明它们在性能关键部分确实使代码更快。通常情况下，正确位置的一些类型会有很长的路要走。



#### 如何构建Cython 代码

与 Python 不同，Cython 代码必须编译。这发生在两个阶段：

> - A `.pyx`文件由 Cython 编译为`.c`文件，包含 Python 扩展模块的代码。
> - `.c`文件由 C 编译器编译为`.so`文件（或 Windows 上的`.pyd`），可直接`import`直接进入 Python 会话.Distutils 或 setuptools 负责这部分。虽然 Cython 可以在某些情况下为你调用它们。

要完全理解 Cython + distutils / setuptools 构建过程，可能需要阅读更多关于[分发 Python 模块](https://docs.python.org/3/distributing/index.html)的内容。

有几种方法可以构建 Cython 代码：

> - 写一个 distutils / setuptools `setup.py`。这是正常和推荐的方式。
> - 使用 [Pyximport](https://www.bookstack.cn/read/cython-doc-zh/$userguide-source_files_and_compilation.html#pyximport)，导入 Cython `.pyx`文件就像它们是`.py`文件一样（使用 distutils 在后台编译和构建）。这种方法比编写`setup.py`更容易，但不是很灵活。因此，如果您需要某些编译选项，则需要编写`setup.py`。
> - 手动运行`cython`命令行实用程序，从`.pyx`文件生成`.c`文件，然后手动将`.c`文件编译成适合从 Python 导入的共享库或 DLL。（这些手动步骤主要用于调试和实验。）
> - 使用 [[Jupyter\]](https://www.bookstack.cn/read/cython-doc-zh/docs-5.md#jupyter) 笔记本或 [[Sage\]](https://www.bookstack.cn/read/cython-doc-zh/$docs-install.html#sage) 笔记本，两者都允许 Cython 代码内联。这是开始编写 Cython 代码并运行它的最简单方法。

目前，使用 distutils 或 setuptools 是构建和分发 Cython 文件的最常用方式。其他方法在参考手册的 [源文件和编译](https://www.bookstack.cn/read/cython-doc-zh/$userguide-source_files_and_compilation.html#compilation) 部分中有更详细的描述。



```python
## hello.pyx
print("Hello World")

## setup.py
from distutils.core import setup
from Cython.Build import cythonize
setup(
    ext_modules = cythonize("hello.pyx")
)
## build
 python setup.py build_ext --inplace
    
## test.py
import hello




```



[使用cython 编译c的例子，用于图像处理中的加速](./cython_example)


Reference

[cython 加速](https://www.zhihu.com/column/c_1045627328318803968 )

[cython 中文文档](https://www.bookstack.cn/read/cython-doc-zh/README.md)



## pybind11

[pybind11 说明文档](https://pybind11.readthedocs.io/en/stable/index.html)

install  直接下载pybind11 include文件，或者 pip install pybind11

```cpp

#include <pybind11/pybind11.h>

namespace py = pybind11;


//simple function
#include <pybind11/pybind11.h>

int add(int i, int j) {
    return i + j;
}
PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function which adds two numbers");
}
//当python import 时调用PYBIND11_MODULE 宏
//模块名称（example）作为第一个宏参数给出（不应用引号引起来）。第二个参数（m）定义类型的变量，py::module_该变量是创建绑定的主要接口。该方法module_::def() 生成将代码公开add()给Python的绑定代码。

//默认参数
m.def("add", &add, "A function which adds two numbers",
      py::arg("i") = 1, py::arg("j") = 2);



//编译1  pybind11是仅标头的库，因此无需链接到任何特殊的库，并且没有中间（魔术）转换步骤。在Linux上，可以使用以下命令来编译以上示例：
$ c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` example.cpp -o example`python3-config --extension-suffix`

    
    
    
    
    
    
    
    
//编译2  使用setup 编译 
//setup.py  遇到问题，需要指定c++11
 
//   src/main.cpp
#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) {
    return i + j;
}

namespace py = pybind11;

PYBIND11_MODULE(python_example, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: python_example
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers
        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers
        Some other explanation about the subtract function.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}



//setup.py 文件
from setuptools import setup
# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import sys
__version__ = "0.0.1"

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)

    ## pip install pybind11 方式编译
ext_modules = [
    Pybind11Extension("python_example",     ## pip install pybind11 方式
        ["src/main.cpp"],
        # Example: passing in the version to the compiled code
        define_macros = [('VERSION_INFO', __version__)],
        extra_compile_args=['-std=c++11'], ## 需要指定c++11 否则会报错
        ),
]

    ## 使用pybind11头文件方式编译 
    from setuptools import setup
    from setuptools import Extension
    ext_modules = [
    Extension(name = "python_example",     ## pip install pybind11 方式
        source = ["src/main.cpp"],
              include_dirs = [r'./pybind11/incude/']  ## 指定pybind11 头文件
        # Example: passing in the version to the compiled code
        define_macros = [('VERSION_INFO', __version__)],
        extra_compile_args=['-std=c++11'], ## 需要指定c++11 否则会报错
        ),
	]
    
setup(
    name="python_example",
    version=__version__,
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)

## build
python setup.py build_ext --inplace
## 保存编译结果到文件
python setup.py build_ext --inplace  >1.txt 2>&1  

    
    
    
//编译3 使用cmake 编译
//CMakeLists.txt 
cmake_minimum_required(VERSION 3.4...3.18)
project(cmake_example)

add_subdirectory(pybind11)
pybind11_add_module(cmake_example src/main.cpp)
    
    
```





### 面向对象

```cpp
struct Pet {
    Pet(const std::string &name) : name(name) { }
    void setName(const std::string &name_) { name = name_; }
    const std::string &getName() const { return name; }

    std::string name;
};

#include <pybind11/pybind11.h>
namespace py = pybind11;
PYBIND11_MODULE(example, m) {
    py::class_<Pet>(m, "Pet")
        .def(py::init<const std::string &>())
        .def("setName", &Pet::setName)
        .def("getName", &Pet::getName);
}
```












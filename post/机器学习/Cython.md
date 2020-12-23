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




引用

<<<<<<< .mine
Reference

=======
[cython 加速]: https://www.zhihu.com/column/c_1045627328318803968
[cython 中文文档]: https://www.bookstack.cn/read/cython-doc-zh/README.md
>>>>>>> .theirs

[cython 加速](https://www.zhihu.com/column/c_1045627328318803968 )

[cython 中文文档](https://www.bookstack.cn/read/cython-doc-zh/README.md)


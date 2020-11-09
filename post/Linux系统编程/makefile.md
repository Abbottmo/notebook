## 普通版

```bash
app: main.o sum.o sub.o
     gcc main.o sum.o sub.o -o app

main.o: main.c
	gcc -c main.c

sum.o: sum.c
	gcc -c sum.c

sub.o: sub.c
	gcc -c sub.c
```



## 自动变量

```bash
$@: 规则中的目标
$<: 规则中的第一个依赖
$^: 规则中的所有依赖
```

```bash
obj = main.o sum.o sub.o
target = app
# makefile系统变量
CC = gcc

$(target):$(obj)
$(CC) $(obj) -o $(target)

%.o: %.c
$(CC) -c $< -o $@
```

makefile函数

```bash
target = app

# 得到所有.c文件
src = $(wildcard ./*.c)

# 将.c替换为.o
obj = $(patsubst ./%.c, ./%.o, $(src))

CC = gcc

$(target): $(obj)
	$(CC) $(obj) -o $(target)

%.o: %.c
	$(CC) -c $< -o $@
	
# 声明伪目标
.PHONY:clean
clean:
	rm $(obj) $(target)
```

## 静态模型

$(objects): %.o:%.c

目标从$(objects)中获取， "%.o"表示以".o"结尾的目标, 即foo.o bar.o, 依赖模式就是把"%.o"中的"%"拿来加上".c"

```bash
objects = foo.o bar.o
all: $(objects)

$(objects): %.o: %.c
	$(CC) -c $(CFLAGS) &< -o $@
```

等价于

```bash
foo.o : foo.c
$(CC) -c $(CFLAGS) foo.c -o foo.o

bar.o : bar.c
$(CC) -c $(CFLAGS) bar.c -o bar.o
```

makefile： 管理项目。

	命名：makefile	 Makefile  --- make 命令
	
	1 个规则：
	
		目标：依赖条件
		（一个tab缩进）命令
	
		1. 目标的时间必须晚于依赖条件的时间，否则，更新目标
	
		2. 依赖条件如果不存在，找寻新的规则去产生依赖条件。
	
	ALL：指定 makefile 的终极目标。


	2 个函数：
	
		src = $(wildcard ./*.c): 匹配当前工作目录下的所有.c 文件。将文件名组成列表，赋值给变量 src。  src = add.c sub.c div1.c 
	
		obj = $(patsubst %.c, %.o, $(src)): 将参数3中，包含参数1的部分，替换为参数2。 obj = add.o sub.o div1.o
	
	clean:	(没有依赖)
	
		-rm -rf $(obj) a.out	“-”：作用是，删除不存在文件时，不报错。顺序执行结束。
	
	3 个自动变量：
	
		$@: 在规则的命令中，表示规则中的目标。
	
		$^: 在规则的命令中，表示所有依赖条件。
	
		$<: 在规则的命令中，表示第一个依赖条件。如果将该变量应用在模式规则中，它可将依赖条件列表中的依赖依次取出，套用模式规则。
	
	模式规则：
	
		%.o:%.c
		   gcc -c $< -o %@
	
	静态模式规则：
	
		$(obj):%.o:%.c
		   gcc -c $< -o %@	
	
	伪目标：
	
		.PHONY: clean ALL
	
	参数：
		-n：模拟执行make、make clean 命令。
	
		-f：指定文件执行 make 命令。				xxxx.mk
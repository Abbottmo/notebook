## shell笔记

## 字符串的提取

*代表要删除的字符

#*_：     从左边开始，去第一个符号“ _ ”左边的所有字符 

% _*：   从右边开始，去掉第一个符号“ _ ”右边的所有字符

##*_：   从右边开始，去掉第一个符号“ _ ”左边的所有字符

%%_*： 从左边开始，去掉第一个符号“ _ ”右边的所有字符

```bash
# 当前目录
project_path=$(cd `dirname $0`; pwd)
# 提取文件夹
project_name="${project_path##*/}"
```


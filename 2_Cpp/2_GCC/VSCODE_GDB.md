# VSCODE配置GDB

## 1 配置launch.json文件

### 1.1 新建并打开launch.json

-  “Run-AddConfiguration-C++(GDB/LLDB)”

![](images/launch1.png)

![](images/launch2.png)

![](images/launch3.png)
              
                          

### 1.2 配置launch.json

- 主要是下面几个选项：
    - **program :** 改为 ”${workspaceFolder}/可执行文件名“
    - **cwd :** 改为 ”${workspaceFolder}“
    - **preLaunchTask :** 必须跟tasks.json文件中的”label“选项一致，这里可以填"build"

![](images/launch4.png)

## 2 配置tasks.json文件
### 1.1 新建并打开tasks.json
- "Terminal-Configure Tasks"
![](images/tasks1.png)
![](images/tasks2.png)
![](images/tasks3.png)

### 1.2 配置tasks.json

- 主要是下面几个选项
    - label : 必须跟launch.json里面的”preLaunchTasks“一致，这里写”build“
    - command : 写g++的路径，”/usr/bin/g++“
    - args : 将g++链接时的命令写进去

![](images/tasks4.png)

## 3 其它问题
### 3.1 常见错误1

![](images/debugerror.png)

解决方法 :
- 第一步 :
```
apt install glibc-source
```
- 第二步 :
```
cd /usr/src/glibc
```
- 第三步 :
```
tar xvf glibc-2.27.tar.xz
```
- 第四步 :
在launch.json文件中的 "configurations"下添加
```
"sourceFileMap" : {
    "/build/glibc-S9d2JN" : "/usr/src/glibc"
}
```
上面弄完之后看不到跳出来的错误了，但是还是有点小问题，不影响debug
https://github.com/microsoft/vscode-cpptools/issues/3831

https://github.com/Microsoft/vscode-cpptools/issues/1123
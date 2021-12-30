

# targets : prerequisities
# 		command


# 编译过程：
# 1、预处理成 .i/ii文件:
# 2、编译成 .s 文件
# 3、汇编成 .o 文件 : 
#      g++ -c [.c/.cpp][.c/.cpp]... -o [.o][.o]...   [flags/options]
# 4、链接成可执行文件 :
#      g++ [.o][.o]... -o [executable] [flags/options]





# 编译成objects文件
speak.o : speak.cpp
	g++ -c $< -o $@

hellospeak.o : hellospeak.cpp
	g++ -c $< -o $@

# 链接成可执行文件
exec : speak.o hellospeak.o
	g++ $^ -o $@


run : exec
	./exec


clean : 
	@rm -f *.o exec

.PHONY : run
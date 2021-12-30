cpp_srcs := $(shell find src -name "*.cpp")
cpp_objs := $(patsubst src/%.cpp,objs/%.o,$(cpp_srcs))

include_paths := /datav/MyLesson/2_Cpp/2_GCC/2.3_MakefileTutorial/2_SharedLibrary/include
library_paths := /datav/MyLesson/2_Cpp/2_GCC/2.3_MakefileTutorial/2_SharedLibrary/lib
link_library  := xxx

I_option := $(include_paths:%=-I%)
l_option := $(link_library:%=-l%)
L_option := $(library_paths:%=-L%)
r_option := $(library_paths:%=-Wl,-rpath=%)

compile_options := -g -O3 -fpic -std=c++11 $(I_option)
link_options    := $(l_option) $(L_option) $(r_option)


objs/%.o : src/%.cpp
	@mkdir -p $(dir $@)
	@g++ -c $^ -o $@ $(compile_options)

objs : $(cpp_objs)

lib/libxxx.so : objs/say.o objs/sayhello.o
	@mkdir -p $(dir $@)
	@g++ -shared $^ -o $@

lib : lib/libxxx.so

workspace/exec : objs/saymain.o
	@g++ $< -o $@ $(link_options)

run : workspace/exec
	@./$<

debug :
	@echo $(I_option)

clean :
	@rm -rf objs lib workspace/exec

.PHONY : clean debug




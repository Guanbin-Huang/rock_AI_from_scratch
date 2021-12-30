

cpp_srcs := $(shell find src -name "*.cpp")
cpp_objs := $(patsubst src/%.cpp,objs/%.o,$(cpp_srcs))

include_paths := 
library_paths :=
link_library  := stdc++

I_option := $(include_paths:%=-I%)
l_option := $(link_library:%=-l%)
L_option := $(library_paths:%=-L%)
r_option := $(library_paths:%=-Wl,-rpath=%)

compile_options := -m64 -std=c++11 -O3 -w -fopenmp $(I_option)
linking_options := $(l_option) $(L_option) $(r_option)



objs/%.o : src/%.cpp
	@echo Compile $^
	@mkdir -p $(dir $@)
	@g++ -c $^ -o $@ $(compile_options)


workspace/exec : $(cpp_objs)
	@echo Link $@
	@mkdir -p $(dir $@)
	@g++ $^ -o $@ $(linking_options)


run : workspace/exec
	@./$<


clean :
	@rm -rf objs workspace/exec


debug :
	@


.PHONY : clean run debug
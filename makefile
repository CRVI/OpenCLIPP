dirs = C++ OpenCLIPP

all : $(dirs)

$(dirs) : ; $(MAKE) -C $(@) $(MAKECMDGOALS)

% : $(dirs) ;

.PHONY : $(dirs) all

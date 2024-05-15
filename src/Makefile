OBJ_DIR = ./lib/

SRCS=$(wildcard *.cpp)

OBJS=$(SRCS:%.cpp=$(OBJ_DIR)/%.o)

DEPS=$(OBJS:.o=.d)


LIB_MAIN= $(OBJ_DIR)/libpyoomph_main.a

SHELL=bash


PYOOMPH_CONFIG_FILE ?= ../pyoomph_config.env
include $(PYOOMPH_CONFIG_FILE)

ifneq ($(PYOOMPH_MARCH_NATIVE),false)
#-Wdate-time
CXXFLAGS =   -fwrapv -O2 -Wall -Wformat -Werror=format-security  -D_FORTIFY_SOURCE=2 -fPIC -DVERSION_INFO="0.0.1" -DNDEBUG
ifeq ($(OS),Windows_NT)
 CXXFLAGS+=-fno-stack-protector
else
ifeq ($(shell uname),Darwin)
 CXXFLAGS+= -fstack-protector-strong    -march=native
else
 CXXFLAGS+= -fstack-protector-strong   -fvisibility=hidden -march=native
endif
endif
else
#-Wdate-time
CXXFLAGS = -fwrapv -O2 -Wall -Wformat -Werror=format-security  -D_FORTIFY_SOURCE=2 -fPIC -DVERSION_INFO="0.0.1" -DNDEBUG
ifeq ($(OS),Windows_NT)
 CXXFLAGS+=-fno-stack-protector
else
ifeq ($(shell uname),Darwin)
 CXXFLAGS+=-fstack-protector-strong
else
 CXXFLAGS+=-fstack-protector-strong -fvisibility=hidden
endif
endif
endif

ifeq ($(PYOOMPH_PARANOID),true)
CXXFLAGS+=-DPARANOID
endif

ifneq ($(PYOOMPH_USE_MPI),false)
CXXFLAGS+=  -I/usr/include/mpi -DOOMPH_HAS_MPI
CXX=mpic++
endif

ifneq ($(PYOOMPH_DEBUG_INFOS),false)
 CXXFLAGS+=-g
else
 CXXFLAGS+=-g0
endif

CXXFLAGS+=-std=c++11

CXXFLAGS+=-I. -I./thirdparty -I./thirdparty/oomph-lib/include -I./thirdparty/tinycc

ifneq ($(PYOOMPH_GINAC_INCLUDE_DIR),)
CXXFLAGS+=-I$(PYOOMPH_GINAC_INCLUDE_DIR)
endif
ifneq ($(PYOOMPH_CLN_INCLUDE_DIR),)
CXXFLAGS+=-I$(PYOOMPH_CLN_INCLUDE_DIR)
endif

ifeq ($(PYOOMPH_NO_TCC),true)
CXXFLAGS+=-DPYOOMPH_NO_TCC
endif

CXXFLAGS+=$(PYOOMPH_CXXFLAGS_EXTRA)


.PHONY: all clean
all: $(LIB_MAIN)

$(OBJ_DIR)/%.o: %.cpp
	@if [[ "$(PYOOMPH_FAST_MULTI_VERSION_BUILD)" != "true" ]]; then echo This Makefile is intendend only to be used for PYOOMPH_FAST_MULTI_VERSION_BUILD=true. Use setup.py for normal builds >&2 ; exit 1; fi
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@
	@$(CXX) -MM -MT $@ $(CXXFLAGS) $< > $(OBJ_DIR)/$*.d


$(LIB_MAIN): $(OBJS)
	@if [[ "$(PYOOMPH_FAST_MULTI_VERSION_BUILD)" != "true" ]]; then echo This Makefile is intendend only to be used for PYOOMPH_FAST_MULTI_VERSION_BUILD=true. Use setup.py for normal builds >&2 ; exit 1; fi
	echo ar rcs $@ $^
	ar rvs $@ $^

clean:
	rm -rf $(OBJ_DIR)/*.d $(OBJ_DIR)/*.o

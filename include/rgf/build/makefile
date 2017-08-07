BIN_NAME = rgf
BIN_DIR = ../bin
TARGET = $(BIN_DIR)/$(BIN_NAME)
CFLAGS = -I../src/com -I../src/tet -O2

ifeq ($(OS), Windows_NT)
	CFLAGS += -static  # MinGW fix
endif

CPP_FILES = ../src/tet/driv_rgf.cpp                \
            ../src/com/AzDmat.cpp                  \
            ../src/tet/AzFindSplit.cpp             \
            ../src/com/AzIntPool.cpp               \
            ../src/com/AzLoss.cpp                  \
            ../src/tet/AzOptOnTree_TreeReg.cpp     \
            ../src/tet/AzOptOnTree.cpp             \
            ../src/com/AzParam.cpp                 \
            ../src/tet/AzReg_Tsrbase.cpp           \
            ../src/tet/AzReg_TsrOpt.cpp            \
            ../src/tet/AzReg_TsrSib.cpp            \
            ../src/tet/AzRgf_FindSplit_Dflt.cpp    \
            ../src/tet/AzRgf_FindSplit_TreeReg.cpp \
            ../src/tet/AzRgf_Optimizer_Dflt.cpp    \
            ../src/tet/AzRgforest.cpp              \
            ../src/tet/AzRgfTree.cpp               \
            ../src/com/AzSmat.cpp                  \
            ../src/tet/AzSortedFeat.cpp            \
            ../src/com/AzStrPool.cpp               \
            ../src/com/AzSvDataS.cpp               \
            ../src/com/AzTaskTools.cpp             \
            ../src/tet/AzTETmain.cpp               \
            ../src/tet/AzTETproc.cpp               \
            ../src/com/AzTools.cpp                 \
            ../src/tet/AzTree.cpp                  \
            ../src/tet/AzTreeEnsemble.cpp          \
            ../src/tet/AzTrTree.cpp                \
            ../src/tet/AzTrTreeFeat.cpp            \
            ../src/com/AzUtil.cpp

all:
	g++ $(CPP_FILES) $(CFLAGS) -o $(TARGET)

# _____________________________________________________________________________
# Parameters

MULBERRY_PRESET=release

LLVM_COMMIT=llvmorg-22.1.0
LLVM_PRESET=Release
LLVM_SRC_DIR=${PROJECT_DIR}/llvm-project
LLVM_BUILD_DIR=${LLVM_SRC_DIR}/build/${LLVM_PRESET}
# LLVM_PYTHON_ENV=${HOME}/.venv/mlirdev
#LLVM_PYTHON_ENV=/usr
LLVM_PYTHON_ENV=${HOME}/.pyenv/shims/

# _____________________________________________________________________________
# Paths

PROJECT_DIR=${shell cd .; pwd}
MULBERRY_BUILD_DIR=./build

# _____________________________________________________________________________
# Targets

.SILENT:
.NOTPARALLEL:

.PHONY: help - Lists targets.
help:
	echo "Targets:"
	sed -nr 's/^.PHONY: (.*) - (.*)/\1|\2/p' ${MAKEFILE_LIST} | \
		awk -F '|' '{printf "* %-30s %s\n", $$1, $$2}' | sort

.PHONY: all - Execute all LLVM and Mulberry targets.
all:	llvm-all \
		mulberry-all

define format
	find ${1} -name "*.cpp" -or -name "*.h" | xargs clang-format -i
endef

.PHONY: format - Format source files.
format:
	echo "Format"
	$(call format, mulberry-opt)
	$(call format, mulberry-translate)
	$(call format, include)
	$(call format, lib)
	$(call format, test)
	$(call format, tools)
	$(call format, unittests)

# _____________________________________________________________________________
# Targets - LLVM

.PHONY: llvm-all - Execute all LLVM targets.
llvm-all: 	llvm-clone \
			llvm-checkout \
			llvm-generate-python-env \
			llvm-generate-project \
			llvm-build

.PHONY: llvm-clean - Clean LLVM Build.
llvm-clean:
	echo "LLVM - Clean"
	rm -rdf ${LLVM_BUILD_DIR}

.PHONY: llvm-clone - Clone LLVM.
llvm-clone:
	echo "LLVM - Clone"
	-git clone https://github.com/llvm/llvm-project.git

.PHONY: llvm-checkout - Checkout LLVM.
llvm-checkout:
	echo "LLVM - Checkout"
	cd ${LLVM_SRC_DIR} && git fetch && git checkout ${LLVM_COMMIT}

.PHONY: llvm-generate-python-env - Generate LLVM Python Virtual Environment.
llvm-generate-python-env:
	echo "LLVM - Generate Python Environment"
	/usr/bin/python3 -m venv ${LLVM_PYTHON_ENV} && \
		source ${LLVM_PYTHON_ENV}/bin/activate && \
		python -m pip install --upgrade pip && \
		python -m pip install -r ${LLVM_SRC_DIR}/mlir/python/requirements.txt

.PHONY: llvm-generate-project - Generate LLVM Project.
llvm-generate-project:
	echo "LLVM - Generate Project"
	cmake -G Ninja -S ${LLVM_SRC_DIR}/llvm -B ${LLVM_BUILD_DIR} \
		-DLLVM_ENABLE_PROJECTS="clang;mlir" \
		-DCLANG_ENABLE_CIR=ON \
		-DLLVM_TARGETS_TO_BUILD=host \
		-DCMAKE_BUILD_TYPE=${LLVM_PRESET} \
		-DLLVM_ENABLE_ASSERTIONS=ON \
		-DMLIR_ENABLE_BINDINGS_PYTHON=ON \
		-DPython3_EXECUTABLE=${LLVM_PYTHON_ENV}/python3 \
		-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
		-DCMAKE_C_COMPILER=clang \
		-DCMAKE_CXX_COMPILER=clang++

.PHONY: llvm-build - Build LLVM.
llvm-build:
	echo "LLVM - Build"
	cmake --build ${LLVM_BUILD_DIR} $(if $(strip $(JOBS)),-j$(strip $(JOBS)))

# _____________________________________________________________________________
# Targets - Mulberry

.PHONY: mulberry-all - Execute all Mulberry targets.
mulberry-all: mulberry-generate-presets \
			mulberry-generate-project \
			mulberry-copy-compile-commands \
			mulberry-build

.PHONY: mulberry-clean - Clean Mulberry Build.
mulberry-clean:
	echo "Mulberry - Clean"
	rm -rdf ${MULBERRY_BUILD_DIR}

.PHONY: mulberry-generate-presets - Generate Mulberry CMake Presets.
mulberry-generate-presets:
	echo "Mulberry - Generate Presets"
	echo $$CMAKE_PRESETS_TEMPLATE > ./CMakeUserPresets.json

.PHONY: mulberry-generate-project - Generate Mulberry Project.
mulberry-generate-project:
	echo "Mulberry - Generate Project"
	cmake -S ${PROJECT_DIR} --preset ${MULBERRY_PRESET}

.PHONY: mulberry-copy-compile-commands - Copy Mulberry `compile_commands.json`.
mulberry-copy-compile-commands:
	echo "Mulberry - Copy compile_commands.json"
	cp ${PROJECT_DIR}/build/${MULBERRY_PRESET}/compile_commands.json  ${PROJECT_DIR}/build

.PHONY: mulberry-build - Build Mulberry.
mulberry-build:
	echo "Mulberry - Build"
	cmake --build ${PROJECT_DIR}/build/${MULBERRY_PRESET} --target check-mulberry mlir-doc --verbose $(if $(strip $(JOBS)),-j$(strip $(JOBS)))
	# echo cmake --build ${PROJECT_DIR}/build/${MULBERRY_PRESET} --target check-mulberry mlir-doc --verbose $(if $(strip $(JOBS)),-j$(strip $(JOBS)))

# _____________________________________________________________________________
# Presets

define CMAKE_PRESETS_TEMPLATE
{
    "version": 3,
    "configurePresets": [
        {
            "name": "default",
            "hidden": true,
            "displayName": "Default configure preset",
            "description": "Default configure preset",
            "generator": "Ninja",
            "binaryDir": "./build/$${presetName}",
            "cacheVariables": {
				"LLVM_SRC_DIR": "${LLVM_SRC_DIR}",
				"MLIR_DIR": "${LLVM_BUILD_DIR}/lib/cmake/mlir",
				"Clang_DIR": "${LLVM_BUILD_DIR}/lib/cmake/clang",
				"LLVM_EXTERNAL_LIT": "${LLVM_BUILD_DIR}/bin/llvm-lit",
				"Python3_EXECUTABLE": "${LLVM_PYTHON_ENV}/python3",
				"CMAKE_C_COMPILER": "/usr/bin/clang",
                "CMAKE_CXX_COMPILER": "/usr/bin/clang++",
				"CMAKE_EXPORT_COMPILE_COMMANDS": "ON"
            }
        },
        {
            "name": "debug",
            "inherits": "default",
            "displayName": "Debug",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Debug"
            }
        },
        {
            "name": "release",
            "inherits": "default",
            "displayName": "Release",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Release"
            }
        },
        {
            "name": "relWithDebInfo",
            "inherits": "default",
            "displayName": "RelWithDebInfo",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "RelWithDebInfo"
            }
        }
    ]
}
endef
export CMAKE_PRESETS_TEMPLATE

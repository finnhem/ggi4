# Makefile for installing Python 3.11 and setting up a virtual environment

.PHONY: all update deps download extract configure build install set-default clean venv

# Configuration
PYTHON_VERSION = Python-3.11.3
PYTHON_TGZ    = ${PYTHON_VERSION}.tgz
PYTHON_URL    = https://www.python.org/ftp/python/3.11.3/${PYTHON_TGZ}
INSTALL_DIR   = /usr/local/bin
VENV_NAME     ?= .venv
VENV_DIR      = ${VENV_NAME}

all: update deps download extract configure build install set-default venv

update:
	sudo apt update && sudo apt upgrade -y

deps:
	sudo apt install -y build-essential zlib1g-dev libncurses5-dev \
	libgdbm-dev libnss3-dev libssl-dev libreadline-dev \
	libffi-dev libsqlite3-dev wget libbz2-dev python3-venv

download:
	wget -O ${PYTHON_TGZ} ${PYTHON_URL}

extract:
	tar -xzf ${PYTHON_TGZ}
	rm ${PYTHON_TGZ}

configure:
	cd ${PYTHON_VERSION} && \
	./configure --enable-optimizations

build:
	cd ${PYTHON_VERSION} && \
	make -j$(shell nproc)

install:
	cd ${PYTHON_VERSION} && \
	sudo make altinstall

set-default:
	sudo update-alternatives --install /usr/bin/python3 python3 ${INSTALL_DIR}/python3.11 1
	sudo update-alternatives --config python3

venv: install
	python3.11 -m venv ${VENV_DIR}
	@echo "Virtual environment created at '${VENV_DIR}'"
	@echo "To activate: source ${VENV_DIR}/bin/activate"
	@echo "To exit: deactivate"

clean:
	rm -f ${PYTHON_TGZ}
	rm -rf ${PYTHON_VERSION}
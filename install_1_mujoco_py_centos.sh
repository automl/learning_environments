#!/bin/bash
if [ ! -d "${HOME}/rpm" ]
then
    echo "\033[31;1mCreating directory ~/rpm and downloading and extracting packages there.\033[0m"
    mkdir -p ~/rpm
    yumdownloader --destdir ~/rpm --resolve mesa-libOSMesa.x86_64 mesa-libOSMesa-devel.x86_64 patchelf.x86_64
    cd ~/rpm
    for rpm in `ls`; do rpm2cpio $rpm | cpio -id ; done
else
    echo "\033[31;1mDirectory ~/rpm already exists. Will not download and extract OS RPMs needed for mujoco-py. Please look inside the script and debug if you face errors.\033[0m"
fi

export PATH="$PATH:$HOME/rpm/usr/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/rpm/usr/lib:$HOME/rpm/usr/lib64"
export LDFLAGS="-L$HOME/rpm/usr/lib -L$HOME/rpm/usr/lib64"
export CPATH="$CPATH:$HOME/rpm/usr/include"

echo -e '\033[31;1mWriting to ~/.bashrc\033[0m'
cat >>~/.bashrc <<EOL
# Added by install_mujoco_py_centos.sh script
export PATH="$PATH:$HOME/rpm/usr/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/rpm/usr/lib:$HOME/rpm/usr/lib64"
export LDFLAGS="-L$HOME/rpm/usr/lib -L$HOME/rpm/usr/lib64"
export CPATH="$CPATH:$HOME/rpm/usr/include"
EOL
#"\nexport PATH=\"$PATH:$HOME/rpm/usr/bin\"\nexport LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH:$HOME/rpm/usr/lib:$HOME/rpm/usr/lib64\"\nexport LDFLAGS=\"-L$HOME/rpm/usr/lib -L$HOME/rpm/usr/lib64\"\nexport CPATH=\"$CPATH:$HOME/rpm/usr/include\"\n"



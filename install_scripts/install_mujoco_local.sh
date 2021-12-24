if [ "$SHELL" = "/usr/bin/zsh" ]; then
    SHELL_TYPE="zsh"
else
    SHELL_TYPE="bash"
fi

echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/home/$USER/.mujoco/mujoco200/bin" >> ~/.$SHELL_TYPE"rc"
echo "export LD_PRELOAD=\$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libGLEW.so" >> ~/.$SHELL_TYPE"rc"

source ~/.$SHELL_TYPE"rc"
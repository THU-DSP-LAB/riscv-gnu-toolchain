# quick start
```
git clone https://git.tsinghua.edu.cn/gpu-dsplab/riscv-gnu-toolchain.git

cd riscv-gnu-toolchain
.configure --prefix=$RISCV
make linux	# build toolchain
make build-sim	# build spike
```
PS: $RISCV 是工具链的安装路径，安装完成后将安装路径加入到环境变量中，可测试看objdump等命令看是否安装成功。

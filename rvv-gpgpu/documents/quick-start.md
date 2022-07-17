# quick start
```
git clone https://git.tsinghua.edu.cn/gpu-dsplab/riscv-gnu-toolchain.git

cd riscv-gnu-toolchain
git submodule update --init --recursive
./configure --prefix=$RISCV
make linux			# build toolchain
make SIM=spike build-sim	# build spike
```
PS:
- $RISCV 是工具链的安装路径，安装完成后将安装路径加入到环境变量中，可测试看objdump等命令看是否安装成功。
- build spike 的过程中子模块init可能出问题，可以手动clone仓库
- submodule clone 中musl会clone失败，不过应该用不到这个所以不用管

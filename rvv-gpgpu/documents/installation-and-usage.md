# installation and usage

## installation

### gcc 编译器搭建

这里是基本的gcc编译器搭建，gitlab库中已经设置了更改后的riscv-binutils子模块，汇编器支持自定义的指令与csr寄存器。

```bash
git clone git@git.tsinghua.edu.cn:gpu-dsplab/riscv-gnu-toolchain.git
cd riscv-gnu-toolchain
mkdir build
cd build
../configure --prefix=${RISCV} --with-arch=rv64gv --with-abi=lp64d --with-cmodel=medany
make
```

设置环境变量，使用`riscv64-unknown-elf-gcc -v`测试是否安装成功。

### spike 安装

```bash
git clone git@git.tsinghua.edu.cn:gpu-dsplab/riscv-isa-sim-gpgpu.git
cd riscv-isa-sim-gpgpu
apt-get install device-tree-compiler
mkdir build
cd build
../configure --prefix=${RISCV}
make
sudo make install
```

PS：spike安装也可以直接在riscv-gnu-toolchain/build中运行`make SIM=spike build-sim`安装，不过这种方法会额外安装pk，时间较长。

### 关于链接修正

```
binutils可能不是gitlab上的修改版
git remote -v确认一下

git remote set-url origin 
https://git.tsinghua.edu.cn/gpu-dsplab/riscv-binutils.git


然后分支记得切到rvv-gpgpu
```

### libgloss-htif 搭建

本设计的gpgpu运行于bare metal状态，故需要安装libgloss-htif来编译bare metal代码

```bash
git clone https://github.com/ucb-bar/libgloss-htif.git
cd libgloss-htif
mkdir build
cd build
../configure --prefix=${RISCV}/riscv64-unknown-elf --host=riscv64-unknown-elf
make
make install
```

之后，运行以下命令检查是否安装成功。

```bash
make check
spike hello.riscv
```

## usage

### bare metal 代码编译

```bash
riscv64-unknown-elf-gcc -fno-common -fno-builtin-printf -specs=htif_nano.specs -c test.s
riscv64-unknown-elf-gcc -static -specs=htif_nano.specs test.o -o test.riscv
spike test.riscv
```

### spike 使用

```bash
spike -d --isa rv64gv --varch elen:256,vlen:32 test.riscv
```

PS:

1. --isa 指定使用的指令集

2. --varch 指定elen，vlen等参数

3. 进入interactive模式后会有一部分加载代码，可以使用until pc命令跳过。具体方法为在使用spike之前先使用命令`riscv64-unknown-elf-objdump -d test.riscv`反汇编可执行程序，找到`main`入口处的pc值（如0x1000），之后在spike interactive mode下运行`until pc [hartid] 0x1000`。interactive mode下可使用help命令查看更多命令的使用方法。
   
   ### link script

使用link脚本可以自己指定数据段和代码段的起始地址，但目前设置会与spike预留的地址空间存在冲突，后续可能会提供支持。
TODO



### 反汇编

```
riscv64-unknown-elf-objdump -D gaussian.riscv > objdump
code objdump
```
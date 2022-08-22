# gpgpu branch support

本文档记录了spike rvv的基本执行逻辑和为支持gpgpu分支执行和simt stack所做的修改。对gpgpu branch进行支持主要分为两部分：1）支持vector branch指令的执行、2）对simt stack进行建模仿真。

## spike原有vector指令执行逻辑
```cpp
VI_VV_LOOP(BODY)
    VI_CHECK_SSS(true)
    VI_LOOP_BASE
        VI_GENERAL_LOOP_BASE # outer for loop to iterate over all elements
        VI_LOOP_ELEMENT_SKIP # mask operation
    get the element according to sew
    BODY
    VI_LOOP_END
```
PS: 
* vm can be set to true to enable rvv-gpgpu simlation
* mask 实现可以仿照 VI_VV_LOOP_CMP

关于rvv的mask相关实现如下：
* mask的设置，有vmseq等指令设置，这些指令调用VI__VV_LOOP_CMP宏来实现
* mask的使用
  * 对一条指令来说，如果vm位set，会读取v0寄存器的值作为mask来mask当前指令
  * mask是否使用由vm决定，各种指令会调用相应的skip相关宏来进行，如果要改成默认执行的话，应该需要对相应的宏进行修改

## vector branch指令执行支持
### 基本支持
为支持自定义的vector branch指令，对spike框架做出了如下的修改：
1. 指令解码支持：通过riscv-opcodes工具生成了自定义vector branch指令的opcode、mask等信息，加入到了文件spike/riscv/encoding.h中
2. 指令执行支持：
   1. 在路径spike/riscv/insn下增加了vbeq、vblt、vbge、vbne、vbltu、vbgeu和join指令的实现
   2. 在文件spike/riscv/v_ext_macros.h中增加了vector branch instruction使用的宏定义VV_LOOP_BRANCH

下面对VV_LOOP_BRANCH的实现做一下详细说明（具体代码见spike/riscv/v_ext_macros.h:2072）。
在指令实现的spike/riscv/insn路径下的.h文件中，指令会将自定义的BODY内容传递给VV_LOOP_BRANCH宏，展开成最终执行的函数。VV_LOOP_BRANCH的实现思路如下。（注意，下面每一个缩进都代表了一级宏的展开）
```c++
VV_LOOP_BRANCH
    VI_CHECK_MSS(true); // 访问寄存器的合法性检查
    VV_LOOP_BRANCH_BODY(VV_BRANCH_PARAMS, BODY) // 循环结构的主要实现
        VV_LOOP_BRANCH_BEGIN // 循环的开始，功能为读取本条指令的vl、vs1、vs2、rd等指令，生成循环遍历元素用到的循环header
        INSNS_BASE(VV_BRANCH_PARAMS, BODY) // 执行两个向量寄存器元素之间的比较，得到if的mask相应的bit
        VV_LOOP_BRANCH_END // 循环结束，算出mask，将vstart寄存器归零
        VV_BRANCH_SS_SET_PC_MASK // simt stack相关操作
            P.simt_stack.push_branch(if_pc, if_mask,    r_mask, else_pc, else_mask); // simt stack入栈
            SET_PC(P.simt_stack.get_pc()); // 从simt stack读取下一条指令的pc值并设置
            SET_MASK(P.simt_stack.get_mask()); // 从simt stack中读取后续指令需要遵守的mask并设置
```

对于join指令，我们有实现如下(riscv/insn/join.h)
```c++
P.simt_stack.pop_join(BRANCH_TARGET); // simt stack pop，传入reconvergence point的pc地址
SET_PC(P.simt_stack.get_npc()); // 从simt stack读取下一条指令的pc值并设置
SET_MASK(P.simt_stack.get_mask()); // 从simt stack中读取后续指令需要遵守的mask并设置
```

上面部分规定了指令执行过程中访问simt stack的接口函数。
```
push_branch : branch指令执行调用的接口，将该指令if_pc、if_mask等信息入栈，改变simt-stack的状态（主要改变后续输出的npc和mask两个变量）。
pop_join : join指令执行调用的接口，将该指令的join pc传递给simt stack，改变simt-stack状态
get_npc : 从simt-stack得到下一条指令的pc值
get_mask : 从simt-stack得到后续执行需要的mask值
```
指令实现时候调用的逻辑如下。
```c++
push_branch(...) or pop_join(...) // 根据指令决定
set_pc(get_npc())
set_mask(get_mask())
```
### 与原有mask机制的兼容
属于与关系

## simt stack支持

### simt stack基本逻辑
the pseudo code listed below shows the usage of instructions related to simt stack.

```
if(condition[vs2 != vs1]) A;
else B;
C;
```

```
main :
	vbeq vs2, vs1, ELSE_CODE
	assembly for A
	join v1, v2, join_point
ELSE_CODE :
	assembly for B
    join v1, v2, join_point
join_point :
	assembly for C
```
与simt stack综合来看

* 遇到vbranch相关指令
  * 计算1）if分支的mask、2）else分支的pc地址、3）else分支的mask。
  * 将r_mask else_pc else_mask入栈。
  * 如果 if 分支的mask为全零，使用else分支的pc地址和mask；否则使用if的mask和分支继续运行
* 遇到join指令
  * 对于 if join，在栈顶中记录reconvergence point，读取栈顶的else pc地址，跳转到else分支执行。
  * 对于 else join，跳转到reconvergence point，弹出栈顶。
* 对于全0\/1mask的处理

### 接口实现
push_branch pop_join get_npc get_mask 

实现于processor.cc内

### barrier指令

实现多warp之间同步

见insn/v_macro_.h和barrier.h

### endprg指令

打印程序结束
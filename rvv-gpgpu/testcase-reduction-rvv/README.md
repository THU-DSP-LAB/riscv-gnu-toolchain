# reduction约简算法
reduction约简算法的汇编版本和c++对比版本代码，目前work for single SM.
## 约简算法
约简算法为一类通用算法，对于任意二元运算符，对于N个数据的输入，最终输出结果为x0(运算)x1(运算)x2(运算)...(运算)xN-1，本样例中运算符为+
## 汇编代码说明
CUDA版本的约简算法为二次约简，包括一个kernel函数和对该函数的两次调用。
当前版本汇编代码包括了该kernel函数和两次调用。
代码`L_FUNC_REDUCTION`之后代码段为kernel函数内容
汇编代码开头定义了具体参数，需要根据运行目标进行修改。
```
.equ OUT_ADDR, 0x40  # final output
.equ IN_ADDR, 0x0 #input
.equ N,    0x080
.equ CSR_NUMW, 0x801 #warp num
.equ CSR_NUMT, 0x802 # num thread
.equ CSR_BI, 0x805 #blockindex
.equ CSR_GD, 0x806 #griddim
.equ CSR_SM, 0x807 # shared memory addr csr id, global addr
.equ CSR_TID, 0x800 #thread id within the block
.equ sPartial, 0x100 #spartial base addr, smaller than shared mem max
.equ CSR_WID, 0x805 #warp id 
.equ partial, 0x200 # result after first time reduction
```
其中`OUT_ADDR`为最终输出结果的基地址，需为sharedmem；
`IN_ADDR`为输入地址，当前版本默认输入为ROM，所以仅需给出偏移量即可，后面会将地址转换为ROM地址；
`N`为待约简数据数量；
`sPartial`为中间结果变量存储位置，为sharedmem；
`partial`为第一次执行完kernel函数的结果保存地址，为sharedmem

## C++代码说明
从`data.txt`中读取input数据；
在执行时需要依次键入待约简数据总数N、单SM的warp数、每个warp包括的thread数；
最终将在屏幕打印每warp每次执行`vse32.v`指令时的存入数据，可与汇编执行结果进行对比。


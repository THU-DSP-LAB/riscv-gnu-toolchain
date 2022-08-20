# PTX

## 从CUDA代码编译生成PTX

现在安装CUDA后不附送官方Sample了, 需要从[Github](https://github.com/nvidia/cuda-samples)自行克隆.
```
$ git clone https://github.com/NVIDIA/cuda-samples.git
```

选取其中最简单的向量加法, 找到目录
```
/cuda-samples/Samples/0_Introduction/vectorAdd/
```
在其下运行
```
$ nvcc -I ../../../Common/ -arch sm_50 -ptx vectorAdd.cu
```
此时`vectorAdd`文件夹内会增加一个文件`vectorAdd.ptx`, 即为编译完成的ptx代码.
* 在其他目录下跑nvcc记得相应更改路径
* `-I ../../../Common/`指示nvcc引用`/cuda-samples/Common/`下的头文件, 否则直接编译NVIDIA的例程会报错
  * 应该也可以采用修改cuda代码, 去除与生成ptx无关的部分解决.
* `-arch`参数指定目标虚拟GPU架构, 此外还有`--gpu-code`参数指定目标物理GPU架构以生成cubin文件, 不过在不考虑不同GPU架构兼容性的情况下, 此处不必做过多区分. 架构对应关系大致如下, 本例选取的是Maxwell架构. 


| Arch      | Code  |
| ----      | ----  |
| Kepler    | sm_3x |
| Maxwell   | sm_5x |
| Pascal    | sm_6x |
| Turing    | sm_7x |
| Ampere    | sm_8x |

# PTX代码阅读
[PTX指令集文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)

## vectorADD.cu的PTX编译结果

```
.version 7.6
.target sm_50
.address_size 64

	// .globl	_Z9vectorAddPKfS0_Pfi

.visible .entry _Z9vectorAddPKfS0_Pfi(              \\ Kernel原型
	.param .u64 _Z9vectorAddPKfS0_Pfi_param_0,      \\ 向量A起始地址d_A的地址
	.param .u64 _Z9vectorAddPKfS0_Pfi_param_1,      \\ 向量B起始地址d_B的地址
	.param .u64 _Z9vectorAddPKfS0_Pfi_param_2,      \\ 向量C起始地址d_C的地址
	.param .u32 _Z9vectorAddPKfS0_Pfi_param_3       \\ 向量元素个数numElements的地址
)
{
	.reg .pred 	%p<2>;                              \\ 寄存器声明(*ptx并非真正的机器码*)
	.reg .f32 	%f<5>;
	.reg .b32 	%r<6>;
	.reg .b64 	%rd<11>;


	ld.param.u64 	%rd1, [_Z9vectorAddPKfS0_Pfi_param_0];  \\ %rd1=向量A起始地址
	ld.param.u64 	%rd2, [_Z9vectorAddPKfS0_Pfi_param_1];  \\ %rd2=向量B起始地址
	ld.param.u64 	%rd3, [_Z9vectorAddPKfS0_Pfi_param_2];  \\ %rd3=向量C起始地址
	ld.param.u32 	%r2, [_Z9vectorAddPKfS0_Pfi_param_3];   \\ %r2=numElements
	mov.u32 	%r3, %ntid.x;                               \\ threadsPerBlock参数, 由于是1D CTA所以只取x分量
	mov.u32 	%r4, %ctaid.x;                              \\ grid内的block编号, 同样是1D grid
	mov.u32 	%r5, %tid.x;                                \\ block内的thread编号, x分量0 ~ %ntid-1, y z分量均为0
	mad.lo.s32 	%r1, %r3, %r4, %r5;                         \\ 乘加指令计算thread的唯一编号(0 ~ totalthreads-1)
	setp.ge.s32 	%p1, %r1, %r2;                          \\ numElements不一定能够用完所有threads, 筛选出>=numElements的thread
	@%p1 bra 	$L__BB0_2;                                  \\ 这些thread直接跳转, 不参与计算

	cvta.to.global.u64 	%rd4, %rd1; // 全局地址转换
	mul.wide.s32 	%rd5, %r1, 4;   // thread编号变换为地址偏移量
	add.s64 	%rd6, %rd4, %rd5;   // 向量A元素的地址
	cvta.to.global.u64 	%rd7, %rd2; 
	add.s64 	%rd8, %rd7, %rd5;   // 向量B元素的地址
	ld.global.f32 	%f1, [%rd8];    // load向量B元素
	ld.global.f32 	%f2, [%rd6];    // load向量A元素
	add.f32 	%f3, %f2, %f1;      // 加法
	add.f32 	%f4, %f3, 0f00000000;   // 0f是IEEE754单精度格式的标记(双精度是0d), 这一步冗余的作用未知
	cvta.to.global.u64 	%rd9, %rd3;
	add.s64 	%rd10, %rd9, %rd5;  // 向量C元素的地址
	st.global.f32 	[%rd10], %f4;   // store向量C元素

$L__BB0_2:
	ret;
}
```

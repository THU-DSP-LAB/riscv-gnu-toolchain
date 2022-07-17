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
   csrr t2, CSR_TID #tid is same for t within the same w
   csrr s4, CSR_NUMT #num thread
   csrr t0, CSR_SM
    vid.v v7
    slli s2, t2, 2 #s2=s1<<2=csrtid<<2
    # param for the first time reduction
    li s3, IN_ADDR # input addr
    add s3, s3, t0
    csrr s5, CSR_NUMW # warp num for first cycle
    li s7, sPartial # where to store result
    li t1, N # num of numbers to be reducted
    li a2, partial # where to store the result after first time reduction
    jal ra, L_FUNC_REDUCTION
    #param for the second time reduction
    barrier x0, x0, x0
    li s5, 1
    csrr s4, CSR_NUMW
    li a2, OUT_ADDR
    csrr t1, CSR_NUMW
    li s3, partial
    vmv.v.i v6, 0
    jal ra, L_FUNC_REDUCTION
    # variable for reduction function is: s5 -> first step reduction stride (first time is num_warp, second time is 1, p.s for single SM)
    # t1 -> total number of numbers to be reducted (first time is N, second time is num_thread)
    # a2 -> address to save the function result (first time is partial to store intermediate result, second time is OUT_ADDR)
    # s3 -> input variable address (first time is IN_ADDR, second time is partial)
    endprg x0, x0, x0

L_FUNC_REDUCTION:
    vadd.vx v2, v7, t2 # thread id->v2
    vle32.v v1, (s3) # v1<-input s3: base addr for input of each warp
    vmv.v.x v3, t1 #v3<-N total num of input needed to be reducted
    vmv.v.x v5, s5
    vmv.v.x v4, s4 #blockdin->v4
    vmul.vv v4, v4, v5 # v4->reduction stride
    vbge v2, v3, L_BB0_3 #branch if tid_true>=N $$diverge1
    addi s1, t2, 0
    slli s5, s1, 2
    add s6, s5, s3
    add s7, s7, s5 # sPartial addr for each warp
    vle32.v v5, (s6) #input
    vadd.vv v6, v5, v6 #sum=sum+in
    vadd.vv v2, v2, v4 #i=i+reduction stride
     vmv.x.s s1, v2 # s1= base addr of input read for each warp
    vblt v2, v3, L_BB0_2
    join v1, v2, L_JOIN2 # $$ diverge2 join if (loop diverge)
L_BB0_2:
    slli s5, s1, 2
    add s6, s5, s3
    vle32.v v5, (s6) #input
    vadd.vv v6, v5, v6 #sum=sum+in
    vadd.vv v2, v2, v4#i=i+blockdim*grindim

    vmv.x.s s1, v2 #s1=blockdim*grindim 起�~K
    join v1, v2, L_JOIN2
    vblt v2, v3, L_BB0_2
    join v1, v2, L_JOIN2

L_JOIN2:# $$diverge2 join point 
   join v1, v2, L_JOIN1 # $$ diverge1 join if
L_BB0_3:
    join v1, v2, L_JOIN1 # $$ diverge1 join else
L_JOIN1: # $$ diverge1 join point
    vse32.v v6, (s7)
    barrier x0, x0, x0
    addi s8, s4, -1 #s8=blockdim-1
    and s9, s8, s4 #s9=blockdim&blockdim-1
    beq s9, x0, L_BB0_9
    addi s10, s4, 0#pow=blockdim

L_BB0_5:
    addi s4, s10, 0
    addi s8, s4, -1
    and s10, s8, s4
    bne s10, x0, L_BB0_5
    vmv.v.x v9, s10
    vblt v7, v9, L_BB0_8 #tid<floorpow2 branch
    slli s2, s10, 2
    sub s11, s7, s2
    vle32.v v10, (s7)
    vle32.v v11, (s11)
    vadd.vv v12, v11, v10
    vse32.v v12, (s11) #mem(r29)=r31  sPartials[tid - floorPow2] += sPartials[tid]
    join v1, v2, L_JOIN3

L_BB0_8:
    join v1, v2, L_JOIN3
L_JOIN3:
    barrier x0, x0, x0

L_BB0_9:
    li s9, 66
    blt s4, s9, L_BB0_14
    srli s5, s4, 1 #s5=active threads

L_BB0_11:
    vmv.v.x v8, s5
    vbge v7, v8, L_BB0_13
    slli s6, s5, 2 #addr form of AT
    add s8, s7, s6
    vle32.v v10, (s7)
    vle32.v v11, (s8)
    vadd.vv v12, v10, v11
    vse32.v v12, (s7)
L_BB0_13:
    barrier x0, x0, x0
    srli s9, s5, 31
    add s10, s9, s5
    srli s2, s10, 1
    li t0, 65
    addi s11, s5, 0
    addi s5, s2, 0
    bgt s11, t0, L_BB0_11

L_BB0_14:
    addi t1, x0, 32
    vmv.v.x v8, t1
    vbge v7, v8, L_BB0_29
    li t1, 33
    blt s4, t1, L_BB0_17 #branch if floorpow2<33
    vle32.v v3, (s7)
    addi t0, s7, 128
    vle32.v v4, (t0)
    vadd.vv v11, v3, v4
    vse32.v v11, (s7)

L_BB0_17:
    li t1, 17
    blt s4, t1, L_BB0_19
    vle32.v v3, (s7)
    addi t0, s7, 64
    vle32.v v4, (t0)
    vadd.vv v11, v3, v4
    vse32.v v11, (s7)
L_BB0_19:
    li t1, 9
    blt s4, t1, L_BB0_21
    vle32.v v3, (s7)
    addi t0, s7, 32
    vle32.v v4, (t0)
    vadd.vv v11, v3, v4
    vse32.v v11, (s7)

L_BB0_21:
    li t1, 5
    blt s4, t1, L_BB0_23
    vle32.v v3, (s7)
    addi t0, s7, 16
    vle32.v v4, (t0)
    vadd.vv v11, v3, v4
    vse32.v v11, (s7)

L_BB0_23:
    li t1, 3
    blt s4, t1, L_BB0_25
    vle32.v v3, (s7)
    addi t0, s7, 8
    vle32.v v4, (t0)
    vadd.vv v11, v3, v4
    vse32.v v11, (s7)

L_BB0_25:
    li t1, 2
    blt s4, t1, L_BB0_27
    vle32.v v3, (s7)
    addi t0, s7, 4
    vle32.v v4, (t0)
    vadd.vv v11, v3, v4
    vse32.v v11, (s7)

L_BB0_27:
    vmv.v.i v8, 0
    vbne v7, v8, L_BB0_28
    csrr t1, CSR_WID
    vle32.v v3, (s7)
    slli s8, t1, 2
    add s9, s8, a2 # a2 is the output addr read&write
    vse32.v v3, (s9)
    join v1, v2, L_JOIN5 # $$diverge 5 join if
L_BB0_28:
    join v1, v2, L_JOIN5 # $$ diverge 5 join else
L_JOIN5:
    join v1, v2, L_JOIN4 # $$ diverge4 join if
L_BB0_29:
    join v1, v2, L_JOIN4 # $$ diverge4 join else
L_JOIN4:
    jalr ra, 0(ra)

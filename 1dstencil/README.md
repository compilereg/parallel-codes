# 1D stencil
* 1D stencil is an operation that for any point in an element in an vector is the sum of the  same element in the input vector + Elements before that number with RADIUS number + Elements after that number with RADIUS number. 
**The elements before 0 counted as 0
**The elements after vector SIZE counted as 0

* Example:
As in Figure 1, vector size=10, and radius=2 

![image](https://github.com/compilereg/parallel-codes/blob/main/1dstencil/example1.png)
Figure 2
 
** To calculate b[0] = a[-2]+a[-1]+a[0]+a[1]+a[2] = 0+0+10+4+5. Because a[-1], and a[-2] before location 0, counted as 0
** To calculate b[1] = a[-1]+a[0]+a[1]+a[2]+a[3] = 0+10+4+5+6. Because a[-1] before location 0, counted as 0
** To calculate b[4] = a[2]+a[3]+a[4]+a[5]+a[6] = 5+7+9+11+13.
** To calculate b[9] = a[7]+a[8]+a[9]+a[10]+a[11] = 20+0+3+0+. Because a[10], and a[11] after vector size, counted as 0


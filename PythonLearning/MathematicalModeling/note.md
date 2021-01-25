# 规划



## 线性规划

### scipy求解

- 需要知道目标函数(一般是求最大或者最小值)和约束条件

  求解前转化为下面的标准形式

  <img src="https://img2020.cnblogs.com/blog/2134757/202101/2134757-20210125160917373-302412204.png" style="zoom:80%;" />

  

```python
from scipy import optimize
import numpy as np

# 求解函数 
res = optimize.linprog(c, A, b, Aeq, beq, LB, UB, X0, OPTIONS)
# 目标函数最小值
print(res.fun)
# 最优解 
print(res.x)
```

- 标准形式是<=，如果是>=，则在两边加上符号-

#### 举例1

<img src="https://img2020.cnblogs.com/blog/2134757/202101/2134757-20210125154651585-1863217363.png" style="zoom:80%;" />

- 使用scipy求解z的最大值

  - c是目标函数的系数矩阵

  - A是化成标准的<=式子的左边的系数矩阵

  - B是化成标准的<=式子的右边的数值矩阵

  - Aeq是所有=左边的系数矩阵，记得里面是[[]]二维

  - Beq是所有=右边的数值矩阵

  - 下面第11行-c加-，是因为此题求的是最大值，但是标准格式是求最小值，所以加负号

  - 另外上面的3个变量都大于0，这里可以使用```bounds=(0, None)```,bounds=(min,max)是范围，None代表无穷，如果不写bounds，那默认是(0, None)

    ```python
    res = optimize.linprog(-c, A, B, Aeq, Beq, bounds=(0, None))
    ```

  ```python
  from scipy import optimize
  import numpy as np
  
  # 确定c,A,b,Aeq,beq
  c = np.array([2, 3, -5])
  A = np.array([[-2, 5, -1], [1, 3, 1]])
  B = np.array([-10, 12])
  Aeq = np.array([[1, 1, 1]])
  Beq = np.array([7])
  # 求解
  res = optimize.linprog(-c, A, B, Aeq, Beq)
  print(res)
  
  ```

  - res

  ```
       con: array([1.80713222e-09])
       fun: -14.57142856564506
   message: 'Optimization terminated successfully.'
       nit: 5
     slack: array([-2.24583019e-10,  3.85714286e+00])
    status: 0
   success: True
         x: array([6.42857143e+00, 5.71428571e-01, 2.35900788e-10])
  
  Process finished with exit code 0
  ```

  - fun是目标函数最小值
  - x是最优解，即上面的x1,x2,x3的最优解

#### 举例2

<img src="https://img2020.cnblogs.com/blog/2134757/202101/2134757-20210125164051771-1805134741.png" style="zoom:80%;" />

```python
from scipy import optimize
import numpy as np

c = np.array([2, 3, 1])
A = np.array([[-1, -4, -2], [-3, -2, 0]])
B = np.array([-8, -6])
Aeq = np.array([[1, 2, 4]])
Beq = np.array([101])
# 求解
res = optimize.linprog(-c, A, B, Aeq, Beq)
print(res)

```

```
con: array([3.75850107e-09])
     fun: -201.9999999893402
 message: 'Optimization terminated successfully.'
     nit: 6
   slack: array([ 93.        , 296.99999998])
  status: 0
 success: True
       x: array([1.01000000e+02, 6.13324051e-10, 3.61350245e-10])

Process finished with exit code 0
```



### pulp求解

- 也可以使用pulp求解，见https://www.bilibili.com/video/BV12h411d7Dm?p=4

- 但是稍微繁琐



## 整数规划

### cvxpy求解

- 和线性规划差不多，但是多了个约束，那就是部分变量被约束为整数

- 目前没有一种方法可以有效地求解一切整数规划。常见的整数规划求解算法有：
  - 分支定界法：可求纯或混合整数线性规划；
  - 割平面法：可求纯或混合整数线性规划；
  - 隐枚举法：用于求解0-1整数规划，有过滤隐枚举法和分支隐枚举法；
  - 匈牙利法：解决指派问题（0-1规划特殊情形）；
  - Monte Carlo法：求解各种类型规划。

#### 举例1

<img src="https://img2020.cnblogs.com/blog/2134757/202101/2134757-20210125190009985-419409110.png" style="zoom:80%;" />

- 同理也是化成<=的标准形式
- 这里的改动只需要我们输入n,a,b,c，以及第10行的小改动，n,a,b,c含义和上面线性规划一样，如果有Aeq和Beq也是同理，加上即可，然后放进cons(下面第11行)里面
- 如果是求最大，第10行用cp.Maximize

```python
import cvxpy as cp
from numpy import array

if __name__ == '__main__':
    n = 2  # 两个变量
    c = array([40, 90])  # 定义目标向量
    a = array([[9, 7], [-7, -20]])  # 定义约束矩阵
    b = array([56, -70])  # 定义约束条件的右边向量
    x = cp.Variable(n, integer=True)  # 定义两个整数决策变量
    obj = cp.Minimize(c * x)  # 构造目标函数
    cons = [a * x <= b, x >= 0]  # 构造约束条件
    prob = cp.Problem(obj, cons)  # 构建问题模型
    prob.solve(solver='GLPK_MI', verbose=True)  # 求解问题
    # prob.solve(solver=cp.CPLEX, verbose=True)  # cp.CPLEX也可以
    print("最优值为:", prob.value)
    print("最优解为:", x.value)
```

- 运行结果会有警告，但是不影响结果

```
      0: obj =   2.700000000e+02 inf =   6.250e-01 (1)
      1: obj =   3.150000000e+02 inf =   0.000e+00 (0)
Long-step dual simplex will be used
+     1: mip =     not found yet >=              -inf        (1; 0)
Solution found by heuristic: 360
+     2: >>>>>   3.500000000e+02 >=   3.500000000e+02   0.0% (1; 0)
+     2: mip =   3.500000000e+02 >=     tree is empty   0.0% (0; 1)
最优值为: 350.0
最优解为: [2. 3.]

Process finished with exit code 0
```

- 参考：https://zhuanlan.zhihu.com/p/344215929





## 非线性规划

- 非线性规划可分为两种，目标函数是凸函数或者是非凸函数

  - 凸函数的非线性规划：如f = x^2+y^2+x*y，可以使用scipy

  - 非凸函数的非线性规划：如求极值，可以有如下方法

    - 纯数学方法，求导求极值

    - 神经网络，深度学习(bp算法链式求导)

    - scipy.optimize.minimize

      ```
      fun：求最小值的目标函数
      args：常数值
      method：求极值方法，一般默认。
      constraints：约束条件
      x0：变量的初始猜测值，注意 minimize是局部最优
      ```

#### 举例

- 计算1/x+x的最小值

  - 只需要改12行的系数，15行的初始猜测值，和8行的函数
  - 如果结果是True，则是找到局部最优解，若是False，则结果是错误的

  ```python
  from scipy.optimize import minimize
  import numpy as np
  
  
  # f = 1/x+x
  def fun(args):
      a = args
      return lambda x: a / x[0] + x[0]
  
  
  if __name__ == '__main__':
      args = (1)  # a
      # x0 = np.asarray((1.5))  # 初始猜测值
      # x0 = np.asarray((2.2))  # 初始猜测值
      x0 = np.asarray((2))  # 设置初始猜测值
  
      res = minimize(fun(args), x0, method='SLSQP')
      print('最值:', res.fun)
      print('是否是最优解', res.success)
      print('取到最值时，x的值(最优解)是', res.x)
  
  ```

  ```
  最值: 2.00000007583235
  是否是最优解 True
  取到最值时，x的值(最优解)是 [1.00027541]
  
  Process finished with exit code 0
  ```

  

  

#### 举例2

- <img src="https://img2020.cnblogs.com/blog/2134757/202101/2134757-20210125205945946-1572918851.png" style="zoom:67%;" />

- x1,x2,x3的范围都在0.1到0.9 之间

- x是变量矩阵，如x[0]即为x1

- 需要变动的是函数fun，con,27,29,32行，x0的设置要尽在要求的0.1到0.9范围内

  ```python
  from scipy.optimize import minimize
  import numpy as np
  
  
  # 计算(2+x1)/(1+x2)- 3*x1+4*x3
  def fun(args):
      a, b, c, d = args
      return lambda x: (a + x[0]) / (b + x[1]) - c * x[0] + d * x[2]
  
  
  def con(args):
      # 约束条件 分为eq 和ineq
      # eq表示 函数结果等于0
      # ineq 表示 表达式大于等于0
      x1min, x1max, x2min, x2max, x3min, x3max = args
      cons = ({'type': 'ineq', 'fun': lambda x: x[0] - x1min},
              {'type': 'ineq', 'fun': lambda x: -x[0] + x1max},
              {'type': 'ineq', 'fun': lambda x: x[1] - x2min},
              {'type': 'ineq', 'fun': lambda x: -x[1] + x2max},
              {'type': 'ineq', 'fun': lambda x: x[2] - x3min},
              {'type': 'ineq', 'fun': lambda x: -x[2] + x3max})
      return cons
  
  
  if __name__ == "__main__":
      # 定义常量值
      args = (2, 1, 3, 4)  # a,b,c,d
      # 设置参数范围/约束条件
      args1 = (0.1, 0.9, 0.1, 0.9, 0.1, 0.9)  # x1min, x1max, x2min,x2max
      cons = con(args1)
      # 设置初始猜测值
      x0 = np.asarray((0.5, 0.5, 0.5))
      res = minimize(fun(args), x0, method='SLSQP', constraints=cons)
      print('最值:', res.fun)
      print('是否是最优解', res.success)
      print('取到最值时，x的值(最优解)是', res.x)
  
  ```

  ```
  最值: -0.773684210526435
  是否是最优解 True
  取到最值时，x的值(最优解)是 [0.9 0.9 0.1]
  
  Process finished with exit code 0
  ```

- 可以看出对于这类简单函数，局部最优解与真实最优解相差不大，但是对于复杂的函数，x0的初始值设置，会很大程度影响最优解的结果

# 数值逼近

## 一维和二维插值

- 参考https://www.bilibili.com/video/BV12h411d7Dm?p=5
- 都使得图像更加光滑

## 最小二乘法拟合

- 用的是

  ```python
  from scipy.optimize import leastsq
  ```

#### 举例

- 一组数据：

  ```
  X = np.array([8.19, 2.72, 6.39, 8.71, 4.7, 2.66, 3.78])
  Y = np.array([7.01, 2.78, 6.47, 6.71, 4.1, 4.23, 4.05])
  ```

- 使用leastsq

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


# 计算以p为参数的直线与原始数据之间误差
def f(p):
    k, b = p
    return Y - (k * X + b)


if __name__ == '__main__':
    X = np.array([8.19, 2.72, 6.39, 8.71, 4.7, 2.66, 3.78])
    Y = np.array([7.01, 2.78, 6.47, 6.71, 4.1, 4.23, 4.05])
    # leastsq使得f的输出数组的平方和最小，参数初始值为[1,0]
    r = leastsq(f, [1, 0])  # 数初始值可以随便设个合理的
    k, b = r[0]
    x = np.linspace(0, 10, 1000)
    y = k * x + b

    # 画散点图，s是点的大小
    plt.scatter(X, Y, s=100, alpha=1.0, marker='o', label=u'数据点')
    # 话拟合曲线，linewidth是线宽
    plt.plot(x, y, color='r', linewidth=2, linestyle="-", markersize=20, label=u'拟合曲线')
    plt.xlabel('安培/A')  # 美赛就不用中文了
    plt.ylabel('伏特/V')
    plt.legend(loc=0, numpoints=1)  # 显示点和线的说明
    # plt.plot(X, Y)
    plt.show()

    print('k = ', k)
    print('b = ', b)

```

```
k =  0.6134953491930442
b =  1.794092543259387

Process finished with exit code 0
```

<img src="https://img2020.cnblogs.com/blog/2134757/202101/2134757-20210125231930793-2082702774.png" style="zoom:67%;" />

- 另外，这个线性拟合也可以使用sklearn求k和b，再去画图

  - 注意维度转换X = X.reshape(-1, 1)

  ```python
  from sklearn import linear_model
  import numpy as np
  
  if __name__ == '__main__':
      X = np.array([8.19, 2.72, 6.39, 8.71, 4.7, 2.66, 3.78])
      Y = np.array([7.01, 2.78, 6.47, 6.71, 4.1, 4.23, 4.05])
      X = X.reshape(-1, 1)
      model = linear_model.LinearRegression()
      model.fit(X, Y)
      print('k = ', model.coef_)
      print('b = ', model.intercept_)
  
  ```

  ```
  k =  [0.61349535]
  b =  1.7940925542916233
  
  Process finished with exit code 0
  ```

- 更多线性和非线性问题，分类问题，见我之前的sklearn blog



# 微分方程

- 微分方程是用来描述某一类函数与其导数之间关系的方程，其解是一个符合方程的函数。微分方程按自变量个数可分为 **常微分方程**和**偏微分方程**，

  前者表达通式 ：<img src="https://img2020.cnblogs.com/blog/2134757/202101/2134757-20210125234039695-755335755.png" style="zoom:50%;" />

  后者表达通式：<img src="https://img2020.cnblogs.com/blog/2134757/202101/2134757-20210125234109978-1913319984.png" style="zoom:67%;" />

- 建议稍微复习一下高数上册最后微分方程那章再看看会更好



## 解析解



- 使用sympy库，但是得到的是字符形式的格式

  如下这种,如果结果是比较复杂的，可能太丑

  如 <img src="https://img2020.cnblogs.com/blog/2134757/202101/2134757-20210126000059946-133280703.png" style="zoom: 50%;" />

  ```python
  import sympy
  
  def apply_ics(sol, ics, x, known_params):
      free_params = sol.free_symbols - set(known_params)
      eqs = [(sol.lhs.diff(x, n) - sol.rhs.diff(x, n)).subs(x, 0).subs(ics) for n in range(len(ics))]
      sol_params = sympy.solve(eqs, free_params)
      return sol.subs(sol_params)
  
  
  if __name__ == '__main__':
      sympy.init_printing()  # 初始化打印环境
      t, omega0, gamma = sympy.symbols("t, omega_0, gamma", positive=True)  # 标记参数，且均为正
      x = sympy.Function('x')  # 标记x是微分函数，非变量
      ode = x(t).diff(t, 2) + 2 * gamma * omega0 * x(t).diff(t) + omega0 ** 2 * x(t)
      ode_sol = sympy.dsolve(ode)  # 用diff()和dsolve得到通解
      ics = {x(0): 1, x(t).diff(t).subs(t, 0): 0}  # 将初始条件字典匹配
      x_t_sol = apply_ics(ode_sol, ics, t, [omega0, gamma])
      sympy.pprint(x_t_sol)
  
  ```

  - 此段解释可见：https://www.bilibili.com/video/BV12h411d7Dm?p=6

  <img src="https://img2020.cnblogs.com/blog/2134757/202101/2134757-20210125235722095-422250700.png" style="zoom:80%;" />

  - 如果最后加上```print(x_t_sol)```

    结果为：

  ```
  Eq(x(t), (-gamma/(2*sqrt(gamma**2 - 1)) + 1/2)*exp(omega_0*t*(-gamma - sqrt(gamma - 1)*sqrt(gamma + 1))) + (gamma/(2*sqrt(gamma**2 - 1)) + 1/2)*exp(omega_0*t*(-gamma + sqrt(gamma - 1)*sqrt(gamma + 1))))
  
  ```

  

- 结果较为简单的常微分方程

  - f(x)''+f(x)=0 二阶常系数齐次微分方程

  ```python
  import sympy as sy
  
  # f(x)''+f(x)=0 二阶常系数齐次微分方程
  def differential_equation(x, f):
      return sy.diff(f(x), x, 2) + f(x)  
  
  
  if __name__ == '__main__':
      x = sy.symbols('x')  # 约定变量
      f = sy.Function('f')  # 约定函数
      print(sy.dsolve(differential_equation(x, f), f(x)))  # 打印
      sy.pprint(sy.dsolve(differential_equation(x, f), f(x)))  # 漂亮的打印
  
  ```

  ```
  Eq(f(x), C1*sin(x) + C2*cos(x))
  f(x) = C₁⋅sin(x) + C₂⋅cos(x)
  
  Process finished with exit code 0
  ```

- 可以参考：https://blog.csdn.net/your_answer/article/details/79234275





## 数值解



- 当ODE(常微分方程)无法求得解析解时，可以用scipy中的integrate.odeint求数值解来探索其解的部分性质,并辅以可视化，能直观地展现ODE解的函数表达

#### 举例

- 一阶非线性常微分方程 <img src="https://img2020.cnblogs.com/blog/2134757/202101/2134757-20210126001647795-11891498.png" style="zoom:50%;" />

- plot_direction_field函数里面的参数意义：

  - y_x:也就是y(x)
  - f_xy:也就是x-y(x)^2
  - x_lim=(-5, 5), y_lim=(-5, 5)也就是在这个x，y轴的范围展示出来

- 关键需要修改的部分,29-31行，33行的y0需要适当调

  ```python
  x = sympy.symbols('x')
  y = sympy.Function('y')
  f = x - y(x) ** 2
  ```

  

```python
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import sympy


def plot_direction_field(x, y_x, f_xy, x_lim=(-5, 5), y_lim=(-5, 5), ax=None):
    f_np = sympy.lambdify((x, y_x), f_xy, 'numpy')
    x_vec = np.linspace(x_lim[0], x_lim[1], 20)
    y_vec = np.linspace(y_lim[0], y_lim[1], 20)
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    dx = x_vec[1] - x_vec[0]
    dy = y_vec[1] - y_vec[0]
    for m, xx in enumerate(x_vec):
        for n, yy in enumerate(y_vec):
            Dy = f_np(xx, yy) * dx
            Dx = 0.8 * dx ** 2 / np.sqrt(dx ** 2 + Dy ** 2)
            Dy = 0.8 * Dy * dy / np.sqrt(dx ** 2 + Dy ** 2)
            ax.plot([xx - Dx / 2, xx + Dx / 2], [yy - Dy / 2, yy + Dy / 2], 'b', lw=0.5)

    ax.axis('tight')
    ax.set_title(r'$ % s$' % (sympy.latex(sympy.Eq(y_x.diff(x), f_xy))), fontsize=18)
    return ax


if __name__ == '__main__':
    x = sympy.symbols('x')
    y = sympy.Function('y')
    f = x - y(x) ** 2
    f_np = sympy.lambdify((y(x), x), f)  # 符号表达式转隐函数
    y0 = 1 # odeint需要给个初始值
    xp = np.linspace(0, 5, 100)
    yp = integrate.odeint(f_np, y0, xp)  # 初始y0解f_np,x范围xp
    xn = np.linspace(0, -5, 100)
    yn = integrate.odeint(f_np, y0, xp)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plot_direction_field(x, y(x), f, ax=ax)  # 绘制f的场线图
    ax.plot(xn, yn, 'b', lw=2)
    ax.plot(xp, yp, 'r', lw=2)
    plt.show()

```

<img src="https://img2020.cnblogs.com/blog/2134757/202101/2134757-20210126004745972-533198965.png" style="zoom:67%;" />



## 传染病模型

- 传染病模型研究属于传染病动力学研究方向，这里只是将模型中微分方程进行了python实现

- 传染病模型包括：SI、SIS、SIR、SIRS、SEIR、SEIRS共六个模型

### SI模型

- 比如病毒传染初期，没有加防疫手段，就符合SI模型

- SI模型的表达式见网上(S:易感染，I:已感染)

- 需要修改的参数

  ```python
  N = 10000  # N为人群总数
  beta = 0.25  # β为传染率系数
  gamma = 0  # gamma为恢复率系数
  I_0 = 1  # I_0为感染者的初始人数
  S_0 = N - I_0  # S_0为易感染者的初始人数
  T = 150  # T为传播时间
  ```

  - β为传染率系数,比如现在100个人已经传染了，在一段时间内，传染新增了25人，则β为0.25
  - gamma为恢复率系数，一开始没有抗体都是为0的，如果不为0，比如是开始有100人感染，在一个传播时间T后，治愈了6个人，则gamma取0.06
  - I_0为感染者的初始人数
  - S_0为易感染者的初始人数，这个要看情况，如果都不加干预，那就是N - I_0，一般看情况需要再考虑其他因素(交通，社交群体，航线等)，S_0考虑的越多，则越完备
  - Susceptible易感染的，Infection已经感染的

- code

```python
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

N = 10000  # N为人群总数
beta = 0.25  # β为传染率系数
gamma = 0  # gamma为恢复率系数
I_0 = 1  # I_0为感染者的初始人数
S_0 = N - I_0  # S_0为易感染者的初始人数
T = 150  # T为传播时间
INI = (S_0, I_0)  # INI为初始状态下的数组


def funcSI(inivalue, _):
    Y = np.zeros(2)
    X = inivalue
    Y[0] = -(beta * X[0] * X[1]) / N + gamma * X[1]  # 易感个体变化
    Y[1] = (beta * X[0] * X[1]) / N - gamma * X[1]  # 感染个体变化
    return Y


if __name__ == '__main__':
    T_range = np.arange(0, T + 1)
    RES = spi.odeint(funcSI, INI, T_range)
    plt.plot(RES[:, 0], color='darkblue', label='Susceptible', marker='.')
    plt.plot(RES[:, 1], color='red', label='Infection', marker='.')
    plt.title('SI Model')
    plt.legend()
    plt.xlabel('Day')
    plt.ylabel('Number')
    plt.show()

```

<img src="https://img2020.cnblogs.com/blog/2134757/202101/2134757-20210126011701220-865611893.png" style="zoom: 80%;" />

- 可以看到10000个人大概在60天左右就全部感染了

### SIS模型

- 与SI区别不大，区别在于7行的gamma有初始值，以及17行的公式改变了

```python
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

N = 10000  # N为人群总数
beta = 0.25  # β为传染率系数
gamma = 0.05  # gamma为恢复率系数
I_0 = 1  # I_0为感染者的初始人数
S_0 = N - I_0  # S_0为易感染者的初始人数
T = 150  # T为传播时间
INI = (S_0, I_0)  # INI为初始状态下的数组


def funcSI(inivalue, _):
    Y = np.zeros(2)
    X = inivalue
    Y[0] = -(beta * X[0]) / N * X[1] + gamma * X[1]  # 易感个体变化
    Y[1] = (beta * X[0] * X[1]) / N - gamma * X[1]  # 感染个体变化
    return Y


if __name__ == '__main__':
    T_range = np.arange(0, T + 1)
    RES = spi.odeint(funcSI, INI, T_range)
    plt.plot(RES[:, 0], color='darkblue', label='Susceptible', marker='.')
    plt.plot(RES[:, 1], color='red', label='Infection', marker='.')
    plt.title('SIS Model')
    plt.legend()
    plt.xlabel('Day')
    plt.ylabel('Number')
    plt.show()

```

<img src="https://img2020.cnblogs.com/blog/2134757/202101/2134757-20210126012337702-518099954.png" style="zoom:80%;" />

- 可以看到60-80天之间，逐渐稳定，有的人治愈后(获得抗体)活了下来，有的没治愈的就死了

### SIR模型

- 多了R_0为治愈者的初始人数，即刚开始注射疫苗恢复的人
- 表达式也改变
- 注意恢复治愈包括自身产生抗体以及通过医疗手段获得抗体两种



```python
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

N = 10000  # N为人群总数
beta = 0.25  # β为传染率系数
gamma = 0.05  # gamma为恢复率系数
I_0 = 1  # I_0为感染者的初始人数
R_0 = 0  # R_0为治愈者的初始人数
S_0 = N - I_0 - R_0  # S_0为易感染者的初始人数
T = 150  # T为传播时间
INI = (S_0, I_0, R_0)  # INI为初始状态下的数组


def funcSIR(inivalue, _):
    Y = np.zeros(3)
    X = inivalue
    Y[0] = -(beta * X[0] * X[1]) / N  # 易感个体变化
    Y[1] = (beta * X[0] * X[1]) / N - gamma * X[1]  # 感染个体变化
    Y[2] = gamma * X[1]  # 治愈个体变化
    return Y


if __name__ == '__main__':
    T_range = np.arange(0, T + 1)
    RES = spi.odeint(funcSIR, INI, T_range)
    plt.plot(RES[:, 0], color='darkblue', label='Susceptible', marker='.')
    plt.plot(RES[:, 1], color='red', label='Infection', marker='.')
    plt.plot(RES[:, 2], color='green', label='Recovery', marker='.')
    plt.title('SIR Model')
    plt.legend()
    plt.xlabel('Day')
    plt.ylabel('Number')
    plt.show()

```

<img src="https://img2020.cnblogs.com/blog/2134757/202101/2134757-20210126012856029-403667398.png" style="zoom:80%;" />

- 可以看到感染人数出现峰值，在前期已经开始使用抗体，病人逐渐治愈，最后所有人都恢复健康



### SIRS模型

- 多了Ts为抗体持续时间，也就是说有了抗体一段时间后，抗体失效，又变成了易感染人群
- 公式也改变

```python
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

N = 10000  # N为人群总数
beta = 0.25  # β为传染率系数
gamma = 0.05  # gamma为恢复率系数
Ts = 7  # Ts为抗体持续时间
I_0 = 1  # I_0为感染者的初始人数
R_0 = 0  # R_0为治愈者的初始人数
S_0 = N - I_0 - R_0  # S_0为易感染者的初始人数
T = 150  # T为传播时间
INI = (S_0, I_0, R_0)  # INI为初始状态下的数组


def funcSIRS(inivalue, _):
    Y = np.zeros(3)
    X = inivalue
    Y[0] = -(beta * X[0] * X[1]) / N + X[2] / Ts  # 易感个体变化
    Y[1] = (beta * X[0] * X[1]) / N - gamma * X[1]  # 感染个体变化
    Y[2] = gamma * X[1] - X[2] / Ts  # 治愈个体变化
    return Y


if __name__ == '__main__':
    T_range = np.arange(0, T + 1)
    RES = spi.odeint(funcSIRS, INI, T_range)
    plt.plot(RES[:, 0], color='darkblue', label='Susceptible', marker='.')
    plt.plot(RES[:, 1], color='red', label='Infection', marker='.')
    plt.plot(RES[:, 2], color='green', label='Recovery', marker='.')
    plt.title('SIRS Model')
    plt.legend()
    plt.xlabel('Day')
    plt.ylabel('Number')
    plt.show()

```

<img src="https://img2020.cnblogs.com/blog/2134757/202101/2134757-20210126013611658-1218001757.png" style="zoom:80%;" />

- 最终达到一个平衡



### SEIR模型

- 考虑了病毒的潜伏期(潜伏人群E)，也就是感染病毒后，过了潜伏期就是感染人群了

- 多了E_0为潜伏者的初始人数，如果是0，那说明开始时有人感染，但是还没有发病，此时这类人不是易感染，但是他们携带病毒
- 只有经过潜伏期，才能被传染
- 公式有所变化

```python
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

N = 10000  # N为人群总数
beta = 0.6  # β为传染率系数
gamma = 0.1  # gamma为恢复率系数
Te = 14  # Te为疾病潜伏期
I_0 = 1  # I_0为感染者的初始人数
E_0 = 0  # E_0为潜伏者的初始人数
R_0 = 0  # R_0为治愈者的初始人数
S_0 = N - I_0 - R_0 - E_0  # S_0为易感染者的初始人数
T = 150  # T为传播时间
INI = (S_0, E_0, I_0, R_0)  # INI为初始状态下的数组


def funcSEIR(inivalue, _):
    Y = np.zeros(4)
    X = inivalue
    Y[0] = -(beta * X[0] * X[2]) / N  # 易感个体变化
    Y[1] = (beta * X[0] * X[2] / N - X[1] / Te)  # 潜伏个体变化
    Y[2] = X[1] / Te - gamma * X[2]  # 感染个体变化
    Y[3] = gamma * X[2]  # 治愈个体变化
    return Y


if __name__ == '__main__':
    T_range = np.arange(0, T + 1)
    RES = spi.odeint(funcSEIR, INI, T_range)
    plt.plot(RES[:, 0], color='darkblue', label='Susceptible', marker=
    '.')
    plt.plot(RES[:, 1], color='orange', label='Exposed', marker='.')
    plt.plot(RES[:, 2], color='red', label='Infection', marker='.')
    plt.plot(RES[:, 3], color='green', label='Recovery', marker='.')
    plt.title('SEIR Model')
    plt.legend()
    plt.xlabel('Day')
    plt.ylabel('Number')
    plt.show()

```

<img src="https://img2020.cnblogs.com/blog/2134757/202101/2134757-20210126014328064-1899684542.png" style="zoom:80%;" />

- 潜伏期的人会比感染人群先到达峰值，最终都可以治愈

### SEIRS模型

- 考虑了抗体持续时间

- 一般多了潜伏期的话，传染率系数会有所增加，上面的SEIR也是同理

```python
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

N = 10000  # N为人群总数
beta = 0.6  # β为传染率系数
gamma = 0.1  # gamma为恢复率系数
Ts = 7  # Ts为抗体持续时间
Te = 14  # Te为疾病潜伏期
I_0 = 1  # I_0为感染者的初始人数
E_0 = 0  # E_0为潜伏者的初始人数
R_0 = 0  # R_0为治愈者的初始人数
S_0 = N - I_0 - R_0 - E_0  # S_0为易感染者的初始人数
T = 150  # T为传播时间
INI = (S_0, E_0, I_0, R_0)  # INI为初始状态下的数组


def funcSEIRS(inivalue, _):
    Y = np.zeros(4)
    X = inivalue
    Y[0] = -(beta * X[0] * X[2]) / N + X[3] / Ts  # 易感个体变化
    Y[1] = (beta * X[0] * X[2] / N - X[1] / Te)  # 潜伏个体变化
    Y[2] = X[1] / Te - gamma * X[2]  # 感染个体变化
    Y[3] = gamma * X[2] - X[3] / Ts  # 治愈个体变化
    return Y


if __name__ == '__main__':
    T_range = np.arange(0, T + 1)
    RES = spi.odeint(funcSEIRS, INI, T_range)
    plt.plot(RES[:, 0], color='darkblue', label='Susceptible', marker='.')
    plt.plot(RES[:, 1], color='orange', label='Exposed', marker='.')
    plt.plot(RES[:, 2], color='red', label='Infection', marker='.')
    plt.plot(RES[:, 3], color='green', label='Recovery', marker='.')
    plt.title('SETRS Model')
    plt.legend()
    plt.xlabel('Day')
    plt.ylabel('Number')
    plt.show()

```

<img src="https://img2020.cnblogs.com/blog/2134757/202101/2134757-20210126015400873-268245591.png" style="zoom:80%;" />

- 潜伏期的人先到峰值，然后是易感染者，然后是治愈者，他们最终会达到平衡稳定



# 图论

## Dijkstra





## Floyd



## 机场航线设计
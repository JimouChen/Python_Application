import sympy as sy


def differential_equation(x, f):
    return sy.diff(f(x), x, 2) + f(x)  # f(x)''+f(x)=0 二阶常系数齐次微分方程


if __name__ == '__main__':
    x = sy.symbols('x')  # 约定变量
    f = sy.Function('f')  # 约定函数
    print(sy.dsolve(differential_equation(x, f), f(x)))  # 打印
    sy.pprint(sy.dsolve(differential_equation(x, f), f(x)))  # 漂亮的打印

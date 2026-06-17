"""
一 函数入门
"""
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 1.不使用函数
# 打印欢迎信息1
print("********************************")
print("*                              *")
print("*     欢迎来到Python世界       *")
print("*                              *")
print("********************************")

# 打印欢迎信息2
print("********************************")
print("*                              *")
print("*     欢迎来到Python世界       *")
print("*                              *")
print("********************************")

# 打印欢迎信息3
print("********************************")
print("*                              *")
print("*     欢迎来到Python世界       *")
print("*                              *")
print("********************************")

# 2.使用函数
def print_welcome():
    """打印欢迎信息"""
    print("********************************")
    print("*                              *")
    print("*     欢迎来到Python世界       *")
    print("*                              *")
    print("********************************")

# 多次调用函数打印欢迎信息
print_welcome()
print_welcome()
print_welcome()

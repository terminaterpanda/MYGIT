import numpy as np
import pandas as pd
import re

class MYMETA(type):
    def __init__(self, adr, sentence, set):
        self.adr = adr
        set = sentence.replace(".", '')
        return set

#metaprogramm -1
#1. reflection

#프로그램이 자신의 구조와 속성을 검사하고 수정할 수 있는 능력.

class Exam:
    def __init__(self, x):
        self.x = x

attr_name = "x"
example = Exam(42)

print(hasattr(example, attr_name))

value = getattr(example, attr_name)
print(value)

setattr(example, attr_name, 13)
print(example.x)

delattr(example, attr_name)

# 데코레이터 (함수 & 메서드 확장 변경)

#@interest

def my_decorater(func):
    def wrapper(): #a -> function -> b
        print("a")
        func()
        print("b")
    return wrapper

def hello():
    print("hihi")

decorated_he = my_decorater(hello)
decorated_he()

# -> 위의 코드와 아래의 코드 동일한 효과
def my_dd(func):
    def wrap():
        print("c")
        func()
        print("d")
    return wrap

@my_dd
def hello():
    print("안녕하세요")

hello()

import time

def timer_decorater(func):
    def wrapper(*args, **kwargs):
        start_time = time.time
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 실행시간 : {end_time - start_time:.5f}초")
        return result
    return wrapper

@timer_decorater
def example_func():
    time.sleep(2)

example_func()

#logging 

#function에 debugging을 잘하기 위해서 use하는 함수
import logging

logging.basicConfig(level=logging.INFO)
#__logger.setLevel(logging.DEBUG)

def limits_calls_decorater(max_calls):
    def decorator(func):
        calls = 0

        def wrapper(*args, **kwargs):
            nonlocal calls
            if calls < max_calls:
                calls += 1
                return func(*args, **kwargs)
            else:
                raise Exception("error")            
        return wrapper
    
    return decorator

@limits_calls_decorater(3)
def example_function():
    print('gogo --')

for i in range(5):
    try:
        example_function()
    except Exception as e:
        print(e)


import os

print(os.listdir())
#os.mkdir() -.> 지정된 경로에 새로운 디렉토리를 생성
new_path = "a"

if not os.path.exists(new_path):
    os.mkdir(new_path)
else:
    print("None")
#os.rmdir() -> 지정된 경로의 dict delete
#예시

dir_path = "aaa"

for root, dirs, files in os.walk(dir_path, topdown=False):
    for name in files:
        file_path = os.path.join(root, name)
        os.remove(file_path)
    for name in dirs:
        dir_path = os.path.join(root, name)
        os.rmdir(dir_path)

os.rmdir(dir_path)

#os.rename = 파일이나 디렉토리의 이름을 change

os.rename(file_path, os.path.join(os.path.dirname(file_path), new_path))
#os.stat()

file_12 = "enter//:12"

file = os.stat(file_12)

print(f"df{file.st_size}")

import functools

#함수의 일부 인자를 고정하여 새로운 함수를 생성

#ex::))
def power(base, exponent):
    return base ** exponent

square = functools.partial(power, exponent=2)
cube = functools.partial(power, exponent=3)

print(square, (4))
#이런 식으로 functools 기능은 -> 함수를 인자를 축약 해서 고정.<using partial>

#functools.reduce()

def multiply(x, y):
    return x * y
#이 함수는 list, tuple 등등의 etc:; 요소에 대해서 주어진 함수를 적용해서 결과반환
numbers = [12, 23]

product = functools.reduce(multiply, numbers)
#list 형태를 넣고, reduce를 사용해 감소줄임.

#functools.cache(), functools.lru_cach()

@functools.cache()
def calculation(x):
    time.sleep(13)
    return x // 4
#cache -> use하여 캐싱을 해놓아서 언제든지 뽑아쓸 수 있게 만듬.
@functools.lru_cache(maxsize = 999) #maxsize = 캐싱메모리를 저장해놓고, stack 형태로 저장.
def sss(x):
    time.sleep(80)
    return x**x

#@functools.wraps() # metadata? 저장

import math

class Intercept:
    def __init__(self, value):
        self.value = value
     
    def calculation(self):
        for i in range(1, int(math.sqrt(self.value)) + 1):
            if self.value % i == 0:  # 나눠 떨어지면 i는 self.value의 약수입니다.
                a = self.value / i
                if a.is_integer():
                    return int(a)  # a가 정수일 경우 정수로 변환하여 반환합니다.
        raise ValueError("소인수분해에 실패했습니다.")  # 적절한 소인수분해를 

#캐싱 
"""
캐싱 -> "파일 복사본을 캐시 혹은 임시 저장 위치에 저장하여 빠르게 access
캐싱을 해두었다가 나중에 뽑아내고, 너무 많은 양, 즉 보관소를 넘어가면 삭제.
"""

#pickle module?
import pickle

#pickle.dump(obj, file, protocol = None, fix_imports = True)

#객체를 파일에 저장하는 과정 - pickling
#파일에서 객체를 읽어오는 과정 - unpickling   

#metaclass -> class의 동작을 제어, 일관되게 속성 추가 등등 
#super()
#-> 부모 클래스의 인자들을 바로 가져올 수 있다.


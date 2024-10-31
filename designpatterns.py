#1. 싱글턴 pattern
class Singleton:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            #__new__ 메서드 == 클래스의 인스턴스를 생성, 정적이므로 따로 return 해줘야함
            #__init__ == 여기로 전달되면 초기화.
        return cls._instance


#2. 팩토리 pattern
class Factory:
    def create_inits(self, value):
        
        if isinstance(value, int):
            return Numeric()
        elif isinstance(value, str):
            return Character()
        elif isinstance(value, float):
            return Float()
        else:
            raise ValueError("this type can't be implemented")
        
class Numeric:
    def calculate(self, num):
        return num * 10
    
class Character:
    def Name_length(self, name):
        return len(name)
    
class Float:
    def floating_pass(self, num):
        return round(num)

#3. 추상적인 Factory Pattern
class AbstractFactory:
    def create_inits1(self):
        raise NotImplementedError
    
    def create_character(self):
        raise NotImplementedError

class MakeFactory(AbstractFactory):
    def create_inits1(self):
        return inits()
    
    def create_character(self):
        return characters()

class inits:
    def return_init(self):
        return "INIT"

class characters:
    def return_character(self):
        return "CHARACTER"

#4. Builder Pattern
"""
객체 생성을 단계별로 나누고, 각 단게별로 객체 생성 방법 지정.
"""
class Product:
    def __init__(self):
        self.parts = []

    def add(self, part):
        self.parts.append(part)
    
    def show(self):
        print("product parts:", ", ".join(self.parts))

class Builder:
    def __init__(self):
        self.product = Product()

    def build_part_a(self):
        self.product.add("Part A")

    def build_part_b(self):
        self.product.add("Part B")

    def get_result(self):
        return self.product

class Director:
    def __init__(self, builder):
        self.builder = builder

    def construct(self):
        self.builder.build_part_a()
        self.builder.build_part_b()



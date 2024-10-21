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
        
        



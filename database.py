# database define.

# 1. key - value database

class KEY():
    def __init__(self, name, *args):
        self.name = name
        if not args:
            raise ValueError("nothing is imported")
        self.lists = {}
        
        for i, value in enumerate(args, start=1):
            self.lists[i] = value

    
    def get_value(self, key):
        return self.lists.get(key, "key not found")
    
    """ class key-value 함수 define 완료."""
    
    def show_table(self):
        for key, value in self.lists.items():
            print(f"key: {key}-> value : {value} ")

# 2. document - based database
class Document:
    def __init__(self):
        self.documents = []

    def insert(self, document):
        self.documents.append(document)

    def find(self, critera):
        return [doc for doc in self.documents if all(doc.get(k) == v for k, v in critera.items())]


class Column:
    def __init__(self):
        self.data = {}

    def insert(self, row_key, column_key, value):
        if row_key not in self.data:
            self.data[row_key] = {}
        self.data[row_key][column_key] = value
        #self.data[row_key][col_key] 이런 식으로 dataframe 저장.

    def select(self, row_key):
        return self.data.get(row_key, {})




        
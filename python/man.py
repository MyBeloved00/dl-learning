class man:
    def __init__(self,name):
        self.name = name
        print("initialized")
    def hello(self):
        print("hello " + self.name)

m = man("jack")
m.hello() 
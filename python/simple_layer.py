class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y

        return self.x * self.y

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        return dout, dout


# test
# apple = 100
# apple_num = 2
# tax = 1.1
#
# l1 = MulLayer()
# l2 = MulLayer()
# apple_price = l1.forward(apple, apple_num)
# final_price = l2.forward(apple_price, tax)
# print(final_price)
# _final_price = 1
# _apple_price, _tax = l2.backward(_final_price)
# _apple, _apple_num = l1.backward(_apple_price)

# 图5-17

l1_1, l1_2 = MulLayer(), MulLayer()
l2 = AddLayer()
l3 = MulLayer()

apple = 100
orange = 150

apple_num = 2
orange_num = 3

tax = 1.1

# forward
apple_price = l1_1.forward(apple, apple_num)
orange_price = l1_2.forward(orange, orange_num)

fruit_price = l2.forward(apple_price, orange_price)

price = l3.forward(fruit_price, tax)
print(apple_price)
print(orange_price)
print(fruit_price)
print(price)
print("++++++backward+++++++")
# backward
dprice = 1
dfruit_price, dtax = l3.backward(dprice)
dapple_price, dorange_price = l2.backward(dfruit_price)
dapple, dapple_num = l1_1.backward(dapple_price)
dorange, dorange_num = l1_2.backward(dorange_price)

print(dfruit_price)
print(dtax)
print(dapple_price)
print(dorange_price)
print(dapple)
print(dapple_num)
print(dorange)
print(dorange_num)

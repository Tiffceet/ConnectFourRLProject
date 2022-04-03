from ConnectFourEnv import ConnectFourEnv

e = ConnectFourEnv()
e.step(1)
e.step(2)
e.step(1)
e.step(2)
e.step(1)
e.step(2)
print(e.check_win())
e.step(1)
print(e.check_win())
e.render()
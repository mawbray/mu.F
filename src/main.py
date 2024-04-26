

"""
TODO :
- visualisation module - tidy up methods into distinct plot functions and integrate with constructor
- integration of all code module
- test and debugging
- documentation
"""



class obj:
    def __init__(self, x):
        self.x = x
    
    def modify_x(self, x):
        self.x = x
        return 


x = 1
t = obj(x)
t.modify_x(2)

print(x)
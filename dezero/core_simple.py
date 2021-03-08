# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 06:53:23 2021

@author: jooji
"""
import weakref
import numpy as np
import contextlib

#--------------------------------------------------------------------------------------
# Config
#-------------------------------------------------------------------------------------- 
class Config:
    enable_backprop = True
    
#--------------------------------------------------------------------------------------
# Variable
#-------------------------------------------------------------------------------------- 
class Variable:
    __array_priority = 200
    
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray): # ndarray 인스턴스만 취급
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))
        self.data = data # 인스턴스 변수 
        self.name = name
        self.grad = None 
        self.creator = None
        self.generation = 0 # 세대 수를 기록하는 변수 
     
    @property # 한 줄 덕분에 메서드를 인스턴스 변수처럼 사용 
    def shape(self): # Variable은 ndarray만 취급한다. 다차원배열용 인스턴스들을 만들어준다.
        return self.data.shape # 차원의 수
    
    @property
    def size(self):
        return self.data.size # 원소 수
    
    @property
    def dtype(self):
        return self.data.dtype # 데이터 타입
        
    def __len__(self): # 특수메서드를 구현하면 Variable 인스턴스에 대해서도 len함수를 사용할 수 있게 된다.
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n','\n' + ' ' * 9)
        return 'variable(' + p + ')'
    
    def set_creator(self, func):
        self.creator = func #함수와변수를연결  
        self.generation = func.generation + 1 # 세대를 기록한다(부모 세대 + 1)
       
    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data) #self.data가 스칼라면 self.grad도 스칼라 
            
        funcs = []
        seen_set = set()
        
        def add_func(f): #중첩함수로 정의 
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x : x.generation)
        
        add_func(self.creator) # generation을 추가 
        
        while funcs:
            f = funcs.pop()          

            gys = [output().grad for output in f.outputs] # 출력변수인 outputs에 담겨 있는 미분값들을 리스트에 담음 
            gxs = f.backward(*gys) # 함수 f의 역전파 호출(gys리스트 언팩)
            
            if not isinstance(gxs, tuple): # 튜플이 아니면 튜플로 반환 
                gxs = (gxs,)
            
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None: # 미분값을 그냥 덮어쓰게 되는 것을 방지 
                    x.grad = gx 
                else:
                    x.grad = x.grad + gx
                
                if x.creator is not None:
                    add_func(x.creator) # 수정 전 : funcs.append(x.creator)
        
        if not retain_grad: 
            for y in f.outputs:
                y().grad = None # y는 약한 참조(weakref) # 각 함수의 출력 변수의 미분값을 유지하지 않도록 
                    
    def cleargrad(self):# Variable에 담기 x를 재사용하는 코드에서 미분값이 누적되어 잘못 출력되는 것을 방지 
        self.grad = None                    
                
#--------------------------------------------------------------------------------------
# Function
#--------------------------------------------------------------------------------------     
class Function:
    def __call__(self, *inputs): # __call__ 파이썬의 특수 메서드 # 인수에 별표를 붙이면 호출할 때 넘긴 인수들을 하나로 모아서 받음. 
        inputs = [as_variable(x) for x in inputs] # 3/8
        xs = [x.data for x in inputs] # 데이터를 꺼낸다. 

        ys = self.forward(*xs) # 구체적인 계산은 forward 메서드에서 한다. 

        if not isinstance(ys, tuple): #튜플이 아닌 경우 추가 지원
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys] # Variable 형태로 되돌린다. 
        
        if Config.enable_backprop: # True 일때만 역전파코드 실행   
            self.generation = max([x.generation for x in inputs]) #세대 설정
            for output in outputs:
                output.set_creator(self) # 연결 설정 
            self.inputs = inputs #입력 변수를 기억(보관)한다.
            self.outputs = [weakref.ref(output) for output in outputs] # 출력도 저장한다. output에 약한 참조방식을 적용(참조카운드가올라가지 않음)
        return outputs if len(outputs) > 1 else outputs[0] # 출력 또한 여러개 
    
    def forward(self, xs):  # 순전파
        raise NotImplementedError() # 예외 발생 
    
    def backward(self, gys):  # 역전파
        raise NotImplementedError()
 
#--------------------------------------------------------------------------------------
# 연산지원 클래스 
#-------------------------------------------------------------------------------------- 
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return (y,)  
    
    def backward(self, gy):
        return gy, gy

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0        
    
#--------------------------------------------------------------------------------------
# 연산지원 클래스 
#--------------------------------------------------------------------------------------   
@contextlib.contextmanager #문맥을 판단하는 함수 
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop',False) #역전파가 필요없을때 with블록에서 순전파코드만실행       

def as_array(x): # 항상 ndarray인스턴스라고 가정했기 때문에 output에 사용
    if np.isscalar(x): # 입력데이터가 스칼라인지 판단 
        return np.array(x)
    return x

def as_variable(obj): #ndarray int float를 Variable인스턴스로 변환
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

def add(x0, x1):
    x1 = as_array(x1) # float, int, np.float64, np.int64와 같은 타입을 ndarray로 변경 
    return Add()(x0, x1)

def mul(x0, x1):
    x1 = as_array(x1) # float, int, np.float64, np.int64와 같은 타입을 ndarray로 변경 
    return Mul()(x0, x1)

#--------------------------------------------------------------------------------------
# 연산자를 그대로 사용할 수 있게 추가 
#--------------------------------------------------------------------------------------
def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
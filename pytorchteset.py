import torch 
print(torch.backends.mps.is_built())
#mps를 use가 가능하다.
print(torch.backends.mps.is_available())
#torch.cuda.is_avaliable()과 동일.

#m1 gpu 가속 연산 use way.
mps_device = torch.device("mps")

x = torch.ones(5, device=mps_device)

y = x*2

#다른 way.

#이런 식으로 use.
class mymodelname():
    pass

model = mymodelname()

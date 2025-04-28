#특정 목표를 달성하기 위해 주어진 환경에서 관찰 및 행동하는 자율적 소프트웨어

#cognitive archictecture
import os

agent_dir = "MTGIT/agents/"
models_dir = "models"
models_dir_1 = "general"
models_dir_2 = "multimodels"
models_dir_3 = "fine_tuned_model"
os.mkdir(agent_dir)
os.path.join(agent_dir, models_dir)
os.path.join(models_dir, models_dir_1)
os.path.join(models_dir, models_dir_2)
os.path.join(models_dir, models_dir_3)


tools_dir = "tools"
os.path.join(agent_dir, tools_dir)

tools_dir_1 = "extension"
#표준화된 interface 제공
tools_dir_2 = "functions"
#데이터 생성 및 작업 실행
tools_dir_3 = "data stores"
#구조화된 data를 혹은 비구조화 data를 에이전트가 활용할 수 있게 해주는 system
os.path.join(tools_dir, tools_dir_1)
os.path.join(tools_dir, tools_dir_2)
os.path.join(tools_dir, tools_dir_3)
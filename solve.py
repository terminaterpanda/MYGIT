
def sum1():
    sum = 0
    for i in range(1, 201):
        if i % 4 == 0 and i % 6 ==0:
            sum += i
            if sum > 1000:
                break
    return sum
sum1()

def find_max_min(a,b,c):
    if a > b and a > c:
        max = a
    elif b > a and b > c:
        max = b
    else:
        max = c
    if a < b and a < c:
        min = a
    elif b < a and b < c:
        min = b
    else:
        min = c
    if min==max:
        return 'same'
    else:
        return max, min
find_max_min(5,9,10)

payment = [1000,2000,3500,4000,45,67,86]

def calculate(cost_list):
    total = sum(cost_list)
    average=total/len(cost_list)
    return total, average

calculate(payment)
#saving.index(a) a의 index를 확인할 수 있음.
def cal(saving, year, month):
    total = sum(saving)
    min_save = min(saving)
    min_index = saving.index(min_save)
    min_year = year[min_index]
    min_month = month[min_index]
    return total, min_year, min_month

#soving.5
#for i in range(1,10) -> 이런 식으로 하면 list가 생성
def gogogo():
    go_list = []
    for i in range(0, 9):
        go_list.append([])
        for j in range(0,9):
            go_list[i].append((i+1)*(j+1))
    return go_list

def conut_str(list):
    food_count = {}
    for x in list:
        if x in food_count:
            food_count[x] +=1
        else:
            food_count[x] = 1
    return food_count

def get_worktime_weekbyweek(w_time):
    avg = []
    for i in range(0, 4):
        d = w_time[7*i:7*(i+1)]
        avg.append(sum(d) / len(d)*60)
    return avg

def cal_sum(a_list, obs):
    sum_2 = []
    for i in range(0, 4):
        d = {}
        d["name"] = a_list[i]
        d['total'] = sum(obs[i])
        d["max"] = max(obs[i])
        d['date'] = obs[i].index(d['max']) + 1
        sum_2.append(d)
    return sum_2
